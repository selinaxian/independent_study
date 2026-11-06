import json
import numpy as np
import pandas as pd
from iclab.data import dump_json, sha256
from risk_v4.core import replay, matched_exposure, block_interval
from finance_v8.core import quantities, shifted_opportunities
from finance_v8.controls import safe_corr,covered_matched_exposure
from eligibility_v9.train import ROOT,HERE,OUT,OLD


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    assert json.loads((OUT/'training_status.json').read_text())['status']=='complete'
    predictions=pd.read_parquet(OUT/'forecasts.parquet').set_index('bar_id')
    allbars=pd.read_parquet(ROOT/'independent_study/results/v1/bars_features.parquet')
    folds=json.loads((ROOT/'independent_study/results/v2/folds.json').read_text())
    calendar=[d for f in folds for d in f['forecast_dates']]
    bars=allbars[allbars.trading_date.isin(calendar)].copy()
    base=pd.read_csv(ROOT/'independent_study/results/v2/backtests/rolling_all__raw_history__base_trades.csv')
    risk=predictions.loc[base.signal_bar_id].reset_index(drop=True)
    assert np.array_equal(risk.trading_date,base.trading_date)
    assert np.array_equal(risk.contract_id,base.contract_id)
    base['fold']=risk.fold.to_numpy()
    trading0=json.loads((ROOT/'independent_study/configs/experiment.json').read_text())['trading']
    rules={f['fold']:json.loads((OUT/'folds'/f'fold_{f["fold"]:02d}'/'lock.json').read_text())['rules'] for f in folds}
    dest=OUT/'backtests';dest.mkdir(exist_ok=True)
    summaries=[];all_daily=[];path_metrics=[];comparisons=[];audit=[]
    covered=risk.covered.to_numpy(dtype=bool)
    for scenario in cfg['execution_scenarios']:
        opportunities=shifted_opportunities(base,allbars,scenario['extra_delay'])
        path_rv=[];mae=[];absolute=[]
        for row in opportunities.itertuples():
            entry,end=int(row.entry_bar_id),int(row.exit_bar_id_actual)
            opens=allbars.iloc[entry:end+1].open.to_numpy()
            path_rv.append(float(np.sqrt(np.sum(np.diff(np.log(opens))**2))*10000))
            path=allbars.iloc[entry:end]
            adverse=(row.entry_open-path.low.min()) if row.side>0 else (path.high.max()-row.entry_open)
            mae.append(float(max(0.,adverse)/row.entry_open*10000))
            absolute.append(abs(row.exit_open/row.entry_open-1)*10000)
        for model in cfg['models']:
            values=risk[model].to_numpy()
            for name,truth in [('held_rv',path_rv),('held_absolute_return',absolute),('direction_mae',mae)]:
                for scope,mask in [('covered',covered),('all_opportunities',np.ones(len(risk),bool))]:
                    a=pd.Series(values[mask]);b=pd.Series(np.asarray(truth)[mask])
                    path_metrics.append(dict(scenario=scenario['name'],model=model,scope=scope,target=name,
                                              n=int(mask.sum()),pearson=safe_corr(a,b),spearman=safe_corr(a,b,rank=True)))
        for application in cfg['applications']:
            base_lots=2 if application=='sizing' else 1
            trading={**trading0,'capital':cfg['capital_per_base_lot']*base_lots,'slippage_ticks':scenario['ticks']}
            variants={'baseline':np.full(len(risk),base_lots)}
            for model in cfg['models']:
                q60=np.array([rules[f][model]['q60'] for f in risk.fold]);q80=np.array([rules[f][model]['q80'] for f in risk.fold])
                variants[model]=quantities(risk[model],covered,q60,q80,application)
            daily_lookup={}
            for model,q in variants.items():
                summary,tr,eq,daily=replay(bars,opportunities,q,trading,calendar)
                # Independent cash-flow reconstruction, without calling accounting helpers.
                qty=tr.lots.to_numpy();side=tr.side.to_numpy();entry=tr.entry_open.to_numpy();exit=tr.exit_open.to_numpy()
                slip=trading['tick']*trading['slippage_ticks'];mult=trading['multiplier']
                gross=(exit-entry)*side*mult*qty
                fees=((entry+side*slip)*(trading['open_fee_rate']+trading['broker_extra_rate'])
                      +(exit-side*slip)*(trading['close_today_fee_rate']+trading['broker_extra_rate']))*mult*qty
                fees+=2*trading['order_fee']*(qty>0)
                expected_net=gross-fees-2*slip*mult*qty
                np.testing.assert_allclose(tr.net_pnl,expected_net,atol=1e-7,rtol=0)
                if scenario['name']=='base' and application=='filter' and model=='baseline':
                    np.testing.assert_allclose(tr.net_pnl,base.net_pnl,atol=1e-7,rtol=0)
                    oldeq=pd.read_parquet(ROOT/'independent_study/results/v2/backtests/rolling_all__raw_history__base_equity.parquet')
                    np.testing.assert_allclose(eq.equity,oldeq.equity,atol=1e-6,rtol=0)
                if model=='unsupervised':
                    oldtr=pd.read_csv(OLD/'backtests'/f'{scenario["name"]}__{application}__teacher_opportunities.csv')
                    np.testing.assert_array_equal(tr.lots,oldtr.lots)
                    np.testing.assert_allclose(tr.net_pnl,oldtr.net_pnl,atol=1e-7,rtol=0)
                control,expected=matched_exposure(tr,trading,cfg['bootstrap_repeats'],cfg['seed'])
                tr['covered']=covered
                conditional,conditional_expected=covered_matched_exposure(tr,trading,cfg['bootstrap_repeats'],cfg['seed'])
                conditional_daily=pd.Series(conditional_expected).groupby(tr.trading_date).sum().reindex(calendar,fill_value=0).to_numpy()
                conditional_excess=daily.pnl.to_numpy()-conditional_daily
                clo,chi=block_interval(conditional_excess,cfg['bootstrap_repeats'],cfg['bootstrap_block_days'],cfg['seed'])
                expected_daily=pd.Series(expected).groupby(tr.trading_date).sum().reindex(calendar,fill_value=0).to_numpy()
                excess=daily.pnl.to_numpy()-expected_daily
                elo,ehi=block_interval(excess,cfg['bootstrap_repeats'],cfg['bootstrap_block_days'],cfg['seed'])
                baseline=daily_lookup.get('baseline',daily)
                change=daily.pnl.to_numpy()-baseline.pnl.to_numpy()
                lo,hi=block_interval(change,cfg['bootstrap_repeats'],cfg['bootstrap_block_days'],cfg['seed'])
                common=dict(scenario=scenario['name'],application=application,model=model)
                summaries.append({**common,**summary,**control,'selection_excess_low':elo,'selection_excess_high':ehi,
                                  'net_change':float(change.sum()),'net_change_low':lo,'net_change_high':hi,
                                  'covered_control_expected':conditional['matched_expected_net'],
                                  'covered_selection_excess':conditional['selection_excess'],
                                  'covered_selection_excess_low':clo,'covered_selection_excess_high':chi,
                                  'lot_fraction':float(qty.sum()/(len(qty)*base_lots)),
                                  'covered_opportunities':int(covered.sum())})
                tr['risk_score']=np.nan if model=='baseline' else risk[model].to_numpy()
                tr['covered']=covered;tr['held_rv']=path_rv;tr['direction_mae_bp']=mae;tr['held_absolute_bp']=absolute
                tr['matched_expected_pnl']=expected
                tr['covered_matched_expected_pnl']=conditional_expected
                name=f'{scenario["name"]}__{application}__{model}'
                tr.to_csv(dest/f'{name}_opportunities.csv',index=False)
                daily['matched_expected_pnl']=expected_daily;daily['selection_excess']=excess
                daily['covered_matched_expected_pnl']=conditional_daily;daily['covered_selection_excess']=conditional_excess
                daily.to_csv(dest/f'{name}_daily.csv',index=False)
                if scenario['name']=='base':eq.to_parquet(dest/f'{name}_equity.parquet',index=False)
                all_daily.append(daily.assign(**common));daily_lookup[model]=daily
                audit.append(dict(**common,opportunities=len(tr),accounting_max_error=float(np.max(abs(expected_net-tr.net_pnl))),
                                  margin_rejections=int(summary['margin_rejections'])))
            for other in cfg['models']:
                if other=='true_feedback':continue
                diff=daily_lookup['true_feedback'].pnl.to_numpy()-daily_lookup[other].pnl.to_numpy()
                lo,hi=block_interval(diff,cfg['bootstrap_repeats'],cfg['bootstrap_block_days'],cfg['seed'])
                comparisons.append(dict(scenario=scenario['name'],application=application,other=other,
                                         true_feedback_minus_other=float(diff.sum()),lower=lo,upper=hi))
            print('traded',scenario['name'],application,flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'trading_summary.csv',index=False)
    daily=pd.concat(all_daily,ignore_index=True);daily.to_csv(OUT/'trading_daily.csv',index=False)
    daily['month']=daily.trading_date.str[:7]
    daily.groupby(['scenario','application','model','month'])[['pnl','selection_excess']].sum().reset_index().to_csv(OUT/'trading_monthly.csv',index=False)
    pd.DataFrame(path_metrics).to_csv(OUT/'held_path_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_trading_intervals.csv',index=False)
    dump_json(OUT/'trading_verification.json',dict(cases=len(audit),opportunity_records=sum(a['opportunities'] for a in audit),
       maximum_accounting_error=max(a['accounting_max_error'] for a in audit),
       maximum_margin_rejections=max(a['margin_rejections'] for a in audit),baseline_reproduced=True,
       covered_opportunities=int(covered.sum()),total_opportunities=len(risk),source_sha256=sha256(HERE/'trade.py')))


if __name__=='__main__':main()

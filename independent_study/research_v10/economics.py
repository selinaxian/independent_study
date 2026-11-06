import json
import numpy as np
import pandas as pd
from iclab.data import dump_json,sha256
from risk_v4.core import replay,matched_exposure,block_interval
from finance_v8.core import shifted_opportunities
from research_v10.state_run import ROOT,HERE,load
from research_v10.budget_core import budget_lots,matched_risk

OUT=ROOT/'independent_study/results/v10_economics'
MODELS=['raw40','raw50','old40_stdp','readout50_stdp','full50_stdp','full50_random','full50_shuffled','gated40_stdp']


def main():
    if OUT.exists():raise RuntimeError('Refuse to overwrite economic evaluation')
    OUT.mkdir(parents=True)
    cfg=json.loads((HERE/'config.json').read_text());bars,meta,folds,y,base,current=load()
    source=ROOT/'independent_study/results/v10_refit'
    pred=pd.concat([pd.read_parquet(source/f'{s}_forecasts.parquet') for s in ['development','audit']],ignore_index=True).set_index('bar_id')
    opportunities=pd.read_csv(ROOT/'independent_study/results/v2/backtests/rolling_all__raw_history__base_trades.csv')
    signals=pred.loc[opportunities.signal_bar_id].reset_index(drop=True)
    np.testing.assert_array_equal(signals.contract_id,opportunities.contract_id)
    np.testing.assert_array_equal(signals.trading_date,opportunities.trading_date)
    opportunities['fold']=signals.fold.to_numpy()
    opportunities['session']=bars.session.iloc[opportunities.signal_bar_id].to_numpy()
    calendar=[d for f in folds for d in f['forecast_dates']];live=bars[bars.trading_date.isin(calendar)].copy()
    trading=json.loads((ROOT/'independent_study/configs/experiment.json').read_text())['trading']
    trading={**trading,'capital':2000000.}
    budgets={};budget_rows=[]
    price=bars.close.to_numpy();unit_historical=price[meta.bar_id.to_numpy()]*trading['multiplier']*np.sqrt(y)/10000
    for fold in folds:
        dates=fold['train_dates']+fold['validation_dates'];assert max(dates)<min(fold['forecast_dates'])
        mask=meta.trading_date.isin(dates).to_numpy();budgets[fold['fold']]=float(np.median(unit_historical[mask]))
        budget_rows.append(dict(fold=fold['fold'],budget=budgets[fold['fold']],fit_last_date=max(dates),forecast_first=fold['forecast_dates'][0]))
    pd.DataFrame(budget_rows).to_csv(OUT/'budgets.csv',index=False)
    budget=np.array([budgets[f] for f in signals.fold]);known=price[opportunities.signal_bar_id.to_numpy()]
    quantities={'fixed_one':np.ones(len(signals),int)}
    for m in MODELS:quantities[m],_=budget_lots(signals[m],known,budget,trading['multiplier'])
    summaries=[];all_daily=[];audit=[];pairs=[]
    for scenario,ticks,delay in [('base',1,0),('two_ticks',2,0),('extra_minute',1,1)]:
        tc={**trading,'slippage_ticks':ticks};ops=shifted_opportunities(opportunities,bars,delay)
        unit=[]
        for row in ops.itertuples():
            path=bars.open.iloc[int(row.entry_bar_id):int(row.exit_bar_id_actual)+1].to_numpy()
            variance=float(np.sum(np.diff(np.log(path))**2)*1e8)
            unit.append(row.entry_open*tc['multiplier']*np.sqrt(variance)/10000)
        unit=np.array(unit);daily_by_model={}
        for model,q in quantities.items():
            summary,tr,eq,daily=replay(live,ops,q,tc,calendar)
            qty=tr.lots.to_numpy();side=tr.side.to_numpy();en=tr.entry_open.to_numpy();ex=tr.exit_open.to_numpy()
            slip=tc['tick']*ticks;mult=tc['multiplier']
            gross=(ex-en)*side*mult*qty
            fees=((en+side*slip)*(tc['open_fee_rate']+tc['broker_extra_rate'])+(ex-side*slip)*(tc['close_today_fee_rate']+tc['broker_extra_rate']))*mult*qty+2*tc['order_fee']*(qty>0)
            net=gross-fees-2*slip*mult*qty
            np.testing.assert_allclose(tr.net_pnl,net,rtol=0,atol=1e-7)
            actual_risk=(qty*unit)**2;expected_risk,expected_exceed=matched_risk(tr,unit,budget)
            # The same strict groups also govern the matched cash-flow control.
            control_tr=tr.copy();keys=list(zip(tr.fold,tr.contract_id,tr.session))
            control_tr['fold']=pd.factorize(keys)[0]
            control,expected_net=matched_exposure(control_tr,tc,cfg['bootstrap_repeats'],cfg['seed'])
            tr['risk_budget']=budget;tr['predicted_variance']=np.nan if model=='fixed_one' else signals[model].to_numpy()
            tr['unit_actual_risk']=unit;tr['actual_risk_squared']=actual_risk;tr['matched_risk_squared']=expected_risk
            tr['matched_net_pnl']=expected_net;tr['exceeded_budget']=(qty*unit>budget)
            agg=tr.groupby('trading_date')[['actual_risk_squared','matched_risk_squared','matched_net_pnl']].sum().reindex(calendar,fill_value=0)
            daily=daily.merge(agg,left_on='trading_date',right_index=True,how='left')
            risk_delta=daily.actual_risk_squared.to_numpy()-daily.matched_risk_squared.to_numpy()
            net_delta=daily.pnl.to_numpy()-daily.matched_net_pnl.to_numpy()
            rlo,rhi=block_interval(risk_delta,cfg['bootstrap_repeats'],5,cfg['seed']);nlo,nhi=block_interval(net_delta,cfg['bootstrap_repeats'],5,cfg['seed'])
            expected_total=float(expected_risk.sum());actual_total=float(actual_risk.sum())
            summaries.append(dict(scenario=scenario,model=model,**summary,**control,
                observed_path_variance=actual_total,matched_path_variance=expected_total,
                variance_saving_fraction=1-actual_total/expected_total if expected_total else 0.,
                variance_difference_low=rlo,variance_difference_high=rhi,
                observed_budget_exceedance=float(np.mean(qty*unit>budget)),matched_budget_exceedance=float(expected_exceed.mean()),
                net_selection_low=nlo,net_selection_high=nhi))
            name=f'{scenario}__{model}';tr.to_csv(OUT/f'{name}_opportunities.csv',index=False)
            daily.to_csv(OUT/f'{name}_daily.csv',index=False)
            if scenario=='base':eq.to_parquet(OUT/f'{name}_equity.parquet',index=False)
            all_daily.append(daily.assign(scenario=scenario,model=model));daily_by_model[model]=daily
            audit.append(dict(scenario=scenario,model=model,n=len(tr),accounting_error=float(abs(net-tr.net_pnl).max()),margin_rejections=summary['margin_rejections']))
        for other in ['raw50','old40_stdp','full50_stdp','full50_random','gated40_stdp']:
            a=daily_by_model['readout50_stdp'];b=daily_by_model[other]
            for variable in ['pnl','actual_risk_squared']:
                diff=a[variable].to_numpy()-b[variable].to_numpy();lo,hi=block_interval(diff,cfg['bootstrap_repeats'],5,cfg['seed'])
                pairs.append(dict(scenario=scenario,left='readout50_stdp',right=other,metric=variable,difference=float(diff.sum()),lower=lo,upper=hi,
                    note='Different quantities possible; matched-exposure attribution separately reported'))
        print('economic scenario',scenario,flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.concat(all_daily,ignore_index=True).to_csv(OUT/'daily.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_intervals.csv',index=False)
    dump_json(OUT/'verification.json',dict(status='complete',opportunities=len(signals),cases=len(audit),
        accounting_records=sum(a['n'] for a in audit),maximum_error=max(a['accounting_error'] for a in audit),
        margin_rejections=sum(a['margin_rejections'] for a in audit),all_signals_before_execution=True,
        matched_groups=['fold','contract_id','session'],forecast_variance_units='bp squared over five executable minutes',
        sources={str(p.relative_to(ROOT)):sha256(p) for p in [HERE/'economics.py',HERE/'budget_core.py',HERE/'ECONOMIC_PROTOCOL.md']}))
    print(pd.DataFrame(summaries).query('scenario == "base"')[['model','trades','lots','net_pnl','max_drawdown_pct','variance_saving_fraction','selection_excess']].to_string(index=False))


if __name__=='__main__':main()

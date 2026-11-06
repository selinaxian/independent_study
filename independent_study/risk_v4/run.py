"""Evaluate fixed risk applications, without choosing winners on observed returns."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from iclab.data import sha256
from diagnostics_v2.common import config,save_json
from .prepare import settings,ROOT
from .core import risk_lots,replay,matched_exposure,block_interval


def run():
    cfg=settings();_,v1=config();out=Path(cfg['output']);source=Path(cfg['source'])
    assert (out/'prepared.json').exists()
    lock=json.loads((out/'source_lock.json').read_text())
    assert sha256(ROOT/'PROTOCOL.md')==lock['protocol_sha256']
    assert sha256(ROOT/'config.json')==lock['config_sha256']
    meta=pd.read_parquet(out/'risk_forecasts.parquet').set_index('sample_id',drop=False)
    allbars=pd.read_parquet(Path(v1['output_dir'])/'bars_features.parquet')
    folds=json.loads((source/'folds.json').read_text());calendar=[d for f in folds for d in f['forecast_dates']]
    bars=allbars[allbars.trading_date.isin(calendar)].copy()
    rules={f['fold']:json.loads((out/'calibration'/f"fold_{f['fold']:02d}"/'rules.json').read_text())['rules'] for f in folds}
    dest=out/'backtests';dest.mkdir(exist_ok=True)
    summaries=[];pred_metrics=[];daily_all=[];month_all=[];audits=[];diagnostics=[];pairrows=[]
    for direction in cfg['direction_bases']:
        opportunities=pd.read_csv(source/'backtests'/f'rolling_all__{direction}__base_trades.csv')
        indices=opportunities.sample_id.to_numpy();fm=meta.loc[indices].reset_index(drop=True)
        assert not opportunities.sample_id.duplicated().any()
        opportunities['fold']=fm.fold.to_numpy()
        # Held-path adverse excursion is evaluated after the fact, never in rules.
        mae=[]
        for r in opportunities.itertuples():
            path=allbars.iloc[r.entry_bar_id:r.exit_bar_id_actual]
            adverse=(r.entry_open-path.low.min()) if r.side>0 else (path.high.max()-r.entry_open)
            mae.append(max(0.,adverse/r.entry_open*10000))
        for model in cfg['risk_models']:
            for target in cfg['targets']:
                key=model+'__'+target;risk=fm[key]
                for label,truth in [('future_rv',fm.future_rv),('absolute_return',abs((opportunities.exit_open/opportunities.entry_open-1)*10000)),('direction_mae',pd.Series(mae))]:
                    pred_metrics.append({'direction':direction,'model':model,'target':target,'evaluation_target':label,
                        'pearson_ic':risk.corr(truth),'rank_ic':risk.rank().corr(truth.rank()),'n':len(risk)})
        daily_lookup={}
        for ticks in cfg['slippage_scenarios']:
            for application in ['filter','sizing']:
                base_lots=1 if application=='filter' else cfg['sizing_max_lots']
                trading={**v1['trading'],'capital':cfg['capital_per_base_lot']*base_lots,'slippage_ticks':ticks}
                variants={'baseline':np.full(len(fm),base_lots,dtype=int)}
                for model in cfg['risk_models']:
                    for target in cfg['targets']:
                        key=model+'__'+target;q=np.zeros(len(fm),dtype=int)
                        for fold in sorted(fm.fold.unique()):
                            mask=fm.fold.to_numpy()==fold
                            q[mask]=risk_lots(fm.loc[mask,key].to_numpy(),rules[fold][key],application)
                        variants[key]=q
                for name,quantities in variants.items():
                    prefix=f'{direction}__{application}__{name}__ticks{ticks:g}'
                    summary,tr,eq,daily=replay(bars,opportunities,quantities,trading,calendar)
                    if name=='baseline' and application=='filter' and ticks==1:
                        oldeq=pd.read_parquet(source/'backtests'/f'rolling_all__{direction}__base_equity.parquet')
                        np.testing.assert_array_equal(eq.datetime,oldeq.datetime)
                        delta=float(np.max(abs(eq.equity.to_numpy()-oldeq.equity.to_numpy())))
                        np.testing.assert_allclose(eq.equity,oldeq.equity,atol=1e-6,rtol=0)
                        np.testing.assert_allclose(tr.net_pnl,opportunities.net_pnl,atol=1e-7,rtol=0)
                        audits.append({'direction':direction,'baseline_equity_max_diff':delta,'same_opportunity_ids':True})
                    common={'direction':direction,'application':application,'risk':name,'ticks':ticks}
                    control,expected=matched_exposure(tr,trading,cfg['permutation_repetitions'],cfg['seed'])
                    expected_daily=pd.Series(expected).groupby(tr.trading_date).sum().reindex(calendar,fill_value=0)
                    excess=daily.pnl.to_numpy()-expected_daily.to_numpy()
                    lo,hi=block_interval(excess,cfg['bootstrap_repetitions'],cfg['bootstrap_block_days'],cfg['seed'])
                    base=daily_lookup.get((ticks,application,'baseline'))
                    if base is None:base=daily
                    difference=daily.pnl.to_numpy()-base.pnl.to_numpy()
                    dlo,dhi=block_interval(difference,cfg['bootstrap_repetitions'],cfg['bootstrap_block_days'],cfg['seed'])
                    summaries.append({**common,**summary,**control,'selection_excess_low':lo,'selection_excess_high':hi,
                         'net_change_from_base':float(difference.sum()),'net_change_low':dlo,'net_change_high':dhi,
                         'lot_fraction':summary['lots']/(len(tr)*base_lots)})
                    tr['risk_prediction']=np.nan if name=='baseline' else fm[name].to_numpy()
                    tr['future_rv']=fm.future_rv.to_numpy();tr['direction_mae_bps']=mae
                    tr['matched_expected_pnl']=expected
                    tr.to_csv(dest/(prefix+'_opportunities.csv'),index=False)
                    daily['matched_expected_pnl']=expected_daily.to_numpy();daily['selection_excess']=excess
                    daily.to_csv(dest/(prefix+'_daily.csv'),index=False)
                    if ticks==1:eq.to_parquet(dest/(prefix+'_equity.parquet'),index=False)
                    daily_lookup[(ticks,application,name)]=daily
                    daily_all.append(daily.assign(**common))
                    monthly=daily.assign(month=daily.trading_date.str[:7]).groupby('month')[['pnl','selection_excess']].sum().reset_index()
                    month_all.append(monthly.assign(**common))
                    if name!='baseline':
                        for used,mask in [('kept',tr.lots>0),('skipped',tr.lots==0)]:
                            diagnostics.append({**common,'group':used,'n':int(mask.sum()),
                                'realized_rv_mean':tr.loc[mask,'future_rv'].mean(),
                                'mae_mean':tr.loc[mask,'direction_mae_bps'].mean(),
                                'opportunity_unit_net_mean':opportunities.loc[mask,'net_pnl'].mean()})
                print(f'evaluated {direction}/{application}/{ticks:g} tick; fixed opportunity clock',flush=True)
        for ticks in cfg['slippage_scenarios']:
            for application in ['filter','sizing']:
                for target in cfg['targets']:
                    left=daily_lookup[(ticks,application,'stdp__'+target)]
                    for comparator in ['random','raw_history','raw_plus_stdp','persistence']:
                        right=daily_lookup[(ticks,application,comparator+'__'+target)]
                        diff=left.pnl.to_numpy()-right.pnl.to_numpy()
                        lo,hi=block_interval(diff,cfg['bootstrap_repetitions'],cfg['bootstrap_block_days'],cfg['seed'])
                        pairrows.append({'direction':direction,'application':application,'target':target,'ticks':ticks,
                            'left':'stdp','right':comparator,'net_difference':diff.sum(),'low':lo,'high':hi})
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(pred_metrics).to_csv(out/'opportunity_prediction_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'kept_skipped_diagnostics.csv',index=False)
    pd.concat(daily_all,ignore_index=True).to_csv(out/'daily.csv',index=False)
    pd.concat(month_all,ignore_index=True).to_csv(out/'monthly.csv',index=False)
    pd.DataFrame(pairrows).to_csv(out/'paired_comparisons.csv',index=False)
    for p,h in lock['old_artifacts'].items():assert sha256(p)==h,p
    for p,h in lock['inputs'].items():assert sha256(p)==h,p
    save_json(out/'verification.json',{'baseline_reproductions':audits,'old_artifacts_unchanged':len(lock['old_artifacts']),
        'old_inputs_unchanged':True,'cases':len(summaries),'integer_lots_only':True,
        'max_margin_rejections':max(r['margin_rejections'] for r in summaries),
        'future_labels_not_used_in_rules':True,'new_training':False})
    save_json(out/'run_sources.json',{str(p):sha256(p) for p in sorted(ROOT.glob('*.py'))})


if __name__=='__main__':run()

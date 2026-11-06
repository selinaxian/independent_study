import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from event_stdp_v7.core import (encode,price_events,samples,score,lagged,classification,ranking)
from event_stdp_v7.run import event_rows
from finance_v8.core import delayed_history,hidden_threshold,mid_percentile
from finance_v8.train import ROOT,HERE,OUT


def verify_and_leads(cfg,forecasts):
    bars=pd.read_parquet(ROOT/'independent_study/results/v1/bars_features.parquet')
    xraw=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy();close=bars.close.to_numpy()
    finite=np.isfinite(xraw).all(axis=1);errors=[];lead_rows=[]
    for fid,frame in forecasts.groupby('fold',sort=True):
        fd=OUT/'folds'/f'fold_{fid:02d}';lock=json.loads((fd/'lock.json').read_text());fold=lock['fold']
        for name,expected in lock['artifacts'].items():assert sha256(fd/name)==expected,name
        assert max(fold['train_dates']) < min(fold['validation_dates'])
        assert max(fold['validation_dates']) < min(fold['forecast_dates'])
        train=bars.trading_date.isin(fold['train_dates']).to_numpy()
        future=bars.trading_date.isin(fold['forecast_dates']).to_numpy()
        p=np.load(fd/'preprocess.npz');val=pd.read_parquet(fd/'validation.parquet')
        np.testing.assert_array_equal(p['bounds'],np.quantile(xraw[train & finite],cfg['feature_quantiles'],axis=0))
        ret,raw,event=price_events(close,seg,float(p['event_threshold']),cfg['quiet_minutes'])
        assert float(p['event_threshold'])==np.quantile(abs(ret[train & np.isfinite(ret)]),cfg['event_quantile'])
        eligible,target=samples(xraw,seg,ret,raw,event,cfg['horizon'])
        ids=frame.bar_id.to_numpy()
        np.testing.assert_array_equal(frame.covered,eligible[ids]);np.testing.assert_array_equal(frame.target,target[ids])
        alarms={}
        for model in cfg['models']:
            np.testing.assert_array_equal(p[f'ref_{model}'],np.sort(val.loc[val.covered,model]))
            alarms[model]=np.zeros(len(bars),bool)
            alarms[model][ids]=frame.covered.to_numpy() & (frame[model].to_numpy()>lock['rules'][model]['q80'])
        lead_rows.extend(event_rows(bars,event,eligible,future,alarms,int(fid),cfg['horizon']))
        if fid not in [0,11,22]:continue
        s=encode(xraw,seg,p['bounds']);arrivals,past_valid=delayed_history(s,xraw,seg,cfg['delays'])
        pieces={k:[] for k in ['teacher','random','shuffled','fixed13','autonomous','equal_hidden']}
        for seed in cfg['seeds']:
            m=np.load(fd/f'seed_{seed}.npz')
            np.testing.assert_array_equal(m['theta'],hidden_threshold(m['hidden'],arrivals[train & past_valid],cfg['hidden_quantile']))
            np.testing.assert_array_equal(m['random_theta'],hidden_threshold(m['initial'],arrivals[train & past_valid],cfg['hidden_quantile']))
            h=(arrivals@m['hidden'].T>m['theta']).astype(float);h[~past_valid]=0
            hr=(arrivals@m['initial'].T>m['random_theta']).astype(float);hr[~past_valid]=0
            hf=(arrivals@m['hidden'].T>cfg['threshold']).astype(float);hf[~past_valid]=0
            for model in ['teacher','shuffled','autonomous']:pieces[model].append(h@m[model])
            pieces['random'].append(hr@m['random']);pieces['fixed13'].append(hf@m['fixed13'])
            pieces['equal_hidden'].append(h@np.full(cfg['hidden'],.5))
        scores={k:np.mean(v,axis=0) for k,v in pieces.items()}
        scores['delay_ridge']=((arrivals-p['ridge_mean'])/p['ridge_scale'])@p['ridge_coef']+p['ridge_target_mean']
        scores['current_rv']=np.sqrt(sum(lagged(np.nan_to_num(ret)**2,seg,d) for d in range(5)))
        old=np.load(ROOT/f'independent_study/results/v7/folds/fold_{fid:02d}/model.npz')
        scores['v7'],_=score(s,seg,old['weight'],cfg['output'])
        for data in [frame,val]:
            for model,values in scores.items():
                error=float(np.max(abs(values[data.bar_id.to_numpy()]-data[model])))
                assert error<1e-12,(fid,model,error)
                errors.append(error)
    old=pd.read_parquet(ROOT/'independent_study/results/v7/predictions.parquet')
    old=old[old.model=='teacher_stdp'].sort_values('bar_id')
    quiet=forecasts[forecasts.covered].sort_values('bar_id')
    np.testing.assert_array_equal(old.bar_id,quiet.bar_id);np.testing.assert_array_equal(old.target,quiet.target)
    np.testing.assert_allclose(old.score_raw,quiet.v7,atol=1e-12,rtol=0)
    source_lock=json.loads((OUT/'source_lock.json').read_text())
    for key in ['sources','old_artifacts']:
        for path,expected in source_lock[key].items():assert sha256(ROOT/path)==expected,path
    return pd.DataFrame(lead_rows),dict(reloaded_folds=[0,11,22],reconstructed_split_model_outputs=len(errors),
                maximum_prediction_error=max(errors),all_fold_thresholds_checked=True,
                v7_same_quiet_samples=True,old_artifacts_unchanged=len(source_lock['old_artifacts']))


def main():
    cfg=json.loads((HERE/'config.json').read_text());frame=pd.read_parquet(OUT/'forecasts.parquet')
    leads,verification=verify_and_leads(cfg,frame)
    leads.to_csv(OUT/'event_leads.csv',index=False)
    summaries=[]
    for model,g in leads.groupby('model'):
        eligible=g[g.evaluable];detected=eligible[eligible.detected]
        summaries.append(dict(model=model,total_events=len(g),evaluable_events=len(eligible),detected_events=len(detected),
           event_recall=len(detected)/len(eligible),median_earliest_lead=detected.earliest_lead_minutes.median()))
    pd.DataFrame(summaries).to_csv(OUT/'event_summary.csv',index=False)
    print('Model, source, v7 alignment and event lead checks passed',flush=True)
    parts=[];fold_metrics=[]
    for fid,g in frame.groupby('fold',sort=True):
        fd=OUT/'folds'/f'fold_{fid:02d}';p=np.load(fd/'preprocess.npz');lock=json.loads((fd/'lock.json').read_text())
        quiet=g[g.covered]
        for model in cfg['models']:
            values=quiet[model].to_numpy();percentiles=mid_percentile(values,p[f'ref_{model}'])
            alarm=values>lock['rules'][model]['q80']
            fold_metrics.append(dict(fold=fid,model=model,**classification(quiet.target,percentiles,alarm)))
            part=quiet[['bar_id','trading_date','fold','target']].copy()
            part['model']=model;part['score']=percentiles;part['alarm']=alarm;parts.append(part)
    long=pd.concat(parts,ignore_index=True)
    metrics=pd.DataFrame([dict(model=m,**classification(g.target,g.score,g.alarm)) for m,g in long.groupby('model')])
    metrics.to_csv(OUT/'event_metrics.csv',index=False);pd.DataFrame(fold_metrics).to_csv(OUT/'event_fold_metrics.csv',index=False)
    long['month']=long.trading_date.str[:7]
    pd.DataFrame([dict(model=m,month=month,**classification(g.target,g.score,g.alarm))
                  for (m,month),g in long.groupby(['model','month'])]).to_csv(OUT/'event_monthly_metrics.csv',index=False)
    # Keep sample order identical for paired day-block resampling.
    grouped={m:long[long.model==m].sort_values('bar_id').reset_index(drop=True) for m in cfg['models']}
    ref=grouped['teacher'];y=ref.target.to_numpy();days=ref.trading_date.unique()
    day_indices=[np.flatnonzero(ref.trading_date.to_numpy()==d) for d in days]
    scores={m:g.score.to_numpy() for m,g in grouped.items()}
    rng=np.random.default_rng(cfg['seed']);block=cfg['bootstrap_block_days']
    differences={(m,key):[] for m in cfg['models'] if m!='teacher' for key in ['auc','ap']}
    for _ in range(cfg['bootstrap_repeats']):
        starts=rng.integers(0,len(days)-block+1,int(np.ceil(len(days)/block)))
        chosen=np.concatenate([np.arange(s,s+block) for s in starts])[:len(days)]
        idx=np.concatenate([day_indices[d] for d in chosen])
        values={m:ranking(y[idx],s[idx]) for m,s in scores.items()}
        for (other,key),arr in differences.items():arr.append(values['teacher'][key]-values[other][key])
    actual={m:ranking(y,s) for m,s in scores.items()};intervals=[]
    for (other,key),values in differences.items():
        lo,hi=np.nanquantile(values,[.025,.975])
        intervals.append(dict(other=other,metric=key,difference=actual['teacher'][key]-actual[other][key],lower=lo,upper=hi))
    pd.DataFrame(intervals).to_csv(OUT/'event_paired_intervals.csv',index=False)
    seeds=pd.read_parquet(OUT/'seed_forecasts.parquet');seed_metrics=[]
    for (fold,seed),g in seeds.groupby(['fold','seed']):
        for model in ['teacher','autonomous','random','shuffled','fixed13','equal_hidden']:
            seed_metrics.append(dict(fold=fold,seed=seed,model=model,**ranking(g.target,g[model])))
    pd.DataFrame(seed_metrics).to_csv(OUT/'seed_fold_metrics.csv',index=False)
    dump_json(OUT/'prediction_verification.json',verification)
    print(metrics[['model','n','base_rate','auc','ap','precision','coverage']].to_string(index=False),flush=True)


if __name__=='__main__':main()

import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from event_stdp_v7.core import encode,price_events,samples,classification,ranking
from event_stdp_v7.run import event_rows
from finance_v8.core import delayed_history,hidden_threshold,mid_percentile
from eligibility_v9.core import matured_feedback,mismatch_feedback,EventModulatedPatterns
from eligibility_v9.train import ROOT,HERE,OUT,OLD,ARMS


def verify_and_leads(cfg,forecasts):
    bars=pd.read_parquet(ROOT/'independent_study/results/v1/bars_features.parquet')
    features=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy()
    finite=np.isfinite(features).all(axis=1);errors=[];rows=[];retrained=0
    for fid,frame in forecasts.groupby('fold',sort=True):
        fd=OUT/'folds'/f'fold_{fid:02d}';lock=json.loads((fd/'lock.json').read_text());fold=lock['fold']
        for name,expected in lock['artifacts'].items():assert sha256(fd/name)==expected,name
        assert max(fold['train_dates'])<min(fold['validation_dates'])
        assert max(fold['validation_dates'])<min(fold['forecast_dates'])
        train=bars.trading_date.isin(fold['train_dates']).to_numpy();future=bars.trading_date.isin(fold['forecast_dates']).to_numpy()
        p=np.load(fd/'preprocess.npz');val=pd.read_parquet(fd/'validation.parquet')
        np.testing.assert_array_equal(p['bounds'],np.quantile(features[train&finite],cfg['feature_quantiles'],axis=0))
        ret,raw,event=price_events(bars.close.to_numpy(),seg,float(p['event_threshold']),cfg['quiet_minutes'])
        assert float(p['event_threshold'])==np.quantile(abs(ret[train&np.isfinite(ret)]),cfg['event_quantile'])
        eligible,y=samples(features,seg,ret,raw,event,cfg['horizon'])
        mask,feedback,rate=matured_feedback(event,seg,train,eligible,cfg['horizon'])
        np.testing.assert_array_equal(np.flatnonzero(mask),p['feedback_bar_ids'])
        np.testing.assert_array_equal(feedback[mask],p['feedback_target']);assert rate==float(p['p_train'])
        ids=frame.bar_id.to_numpy();np.testing.assert_array_equal(frame.covered,eligible[ids]);np.testing.assert_array_equal(frame.target,y[ids])
        alarms={}
        for model in cfg['models']:
            np.testing.assert_array_equal(p[f'ref_{model}'],np.sort(val.loc[val.covered,model]))
            assert lock['rules'][model]['q60']==np.quantile(val.loc[val.covered,model],.6)
            assert lock['rules'][model]['q80']==np.quantile(val.loc[val.covered,model],.8)
            alarms[model]=np.zeros(len(bars),bool)
            alarms[model][ids]=frame.covered.to_numpy()&(frame[model].to_numpy()>lock['rules'][model]['q80'])
        rows.extend(event_rows(bars,event,eligible,future,alarms,int(fid),cfg['horizon']))
        oldframe=pd.read_parquet(OLD/'folds'/fd.name/'forecast.parquet')
        np.testing.assert_array_equal(frame.unsupervised,oldframe.teacher)
        for k in ['current_rv','delay_ridge']:np.testing.assert_array_equal(frame[k],oldframe[k])
        if fid not in [0,11,22]:continue
        s=encode(features,seg,p['bounds']);x,past_valid=delayed_history(s,features,seg,cfg['delays'])
        th=train&past_valid;scores={a:[] for a in ARMS}
        for seed in cfg['seeds']:
            wrong=mismatch_feedback(feedback,seg,mask,seed)
            for arm in ARMS:
                m=np.load(fd/f'{arm}_{seed}.npz')
                np.testing.assert_array_equal(m['theta'],hidden_threshold(m['hidden'],x[th],cfg['hidden_quantile']))
                if seed==cfg['seeds'][0]:
                    fb=None if arm=='unsupervised' else (feedback if arm=='true_feedback' else wrong)[th]
                    learned=EventModulatedPatterns(x.shape[1],cfg,seed).fit(x[th],seed+1,fb,rate)
                    np.testing.assert_array_equal(learned.weight,m['hidden']);retrained+=1
                h=(x@m['hidden'].T>m['theta']).astype(float);h[~past_valid]=0
                scores[arm].append(h@m['output'])
        for arm,values in scores.items():
            pred=np.mean(values,axis=0)
            for data in [frame,val]:
                error=float(abs(pred[data.bar_id.to_numpy()]-data[arm]).max());assert error==0
                errors.append(error)
    source=json.loads((OUT/'source_lock.json').read_text())
    for kind in ['sources','old_artifacts']:
        for path,expected in source[kind].items():assert sha256(ROOT/path)==expected,path
    # Training-only single-neuron choice: no forecast-period neuron selection.
    probe=pd.read_parquet(OUT/'hidden_selected_probe.parquet')
    pooled=probe.groupby(['model','bar_id','trading_date'],as_index=False).agg(target=('target','first'),response=('response','mean'))
    records=[]
    for model,g in pooled.groupby('model'):
        records.append(dict(model=model,**classification(g.target,g.response,g.response>0)))
    pd.DataFrame(records).to_csv(OUT/'hidden_probe_metrics.csv',index=False)
    return pd.DataFrame(rows),dict(reloaded_folds=[0,11,22],reconstructed_split_model_outputs=len(errors),
        maximum_prediction_error=max(errors),hidden_retrained_exact=retrained,
        v8_baseline_exact_models=69,all_fold_bounds_feedback_thresholds_checked=True,
        old_artifacts_unchanged=len(source['old_artifacts']))


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
    print('Model, source, v8 alignment and event lead checks passed',flush=True)
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
    ref=grouped['true_feedback'];y=ref.target.to_numpy();days=ref.trading_date.unique()
    day_indices=[np.flatnonzero(ref.trading_date.to_numpy()==d) for d in days]
    scores={m:g.score.to_numpy() for m,g in grouped.items()}
    rng=np.random.default_rng(cfg['seed']);block=cfg['bootstrap_block_days']
    differences={(m,key):[] for m in cfg['models'] if m!='true_feedback' for key in ['auc','ap']}
    for _ in range(cfg['bootstrap_repeats']):
        starts=rng.integers(0,len(days)-block+1,int(np.ceil(len(days)/block)))
        chosen=np.concatenate([np.arange(s,s+block) for s in starts])[:len(days)]
        idx=np.concatenate([day_indices[d] for d in chosen])
        values={m:ranking(y[idx],s[idx]) for m,s in scores.items()}
        for (other,key),arr in differences.items():arr.append(values['true_feedback'][key]-values[other][key])
    actual={m:ranking(y,s) for m,s in scores.items()};intervals=[]
    for (other,key),values in differences.items():
        lo,hi=np.nanquantile(values,[.025,.975])
        intervals.append(dict(other=other,metric=key,difference=actual['true_feedback'][key]-actual[other][key],lower=lo,upper=hi))
    pd.DataFrame(intervals).to_csv(OUT/'event_paired_intervals.csv',index=False)
    seeds=pd.read_parquet(OUT/'seed_forecasts.parquet');seed_metrics=[]
    for (fold,seed),g in seeds.groupby(['fold','seed']):
        for model in ARMS:
            seed_metrics.append(dict(fold=fold,seed=seed,model=model,**ranking(g.target,g[model])))
    pd.DataFrame(seed_metrics).to_csv(OUT/'seed_fold_metrics.csv',index=False)
    dump_json(OUT/'prediction_verification.json',verification)
    print(metrics[['model','n','base_rate','auc','ap','precision','coverage']].to_string(index=False),flush=True)


if __name__=='__main__':main()

"""Stage A: identical supervised probes of frozen v9 representations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from event_stdp_v7.core import encode,price_events,samples,fit_ridge,ranking
from finance_v8.core import delayed_history,mid_percentile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OUT=ROOT/'independent_study/results/v10_probe'
V9=ROOT/'independent_study/results/v9_eligibility'


def main():
    if OUT.exists():raise RuntimeError('Refuse to overwrite probe')
    OUT.mkdir(parents=True)
    cfg=json.loads((ROOT/'independent_study/eligibility_v9/config.json').read_text())
    paths=[HERE/'probe.py',HERE/'PROTOCOL.md',ROOT/'independent_study/event_stdp_v7/core.py',
           ROOT/'independent_study/results/v1/bars_features.parquet',ROOT/'independent_study/results/v2/folds.json']
    source={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    dump_json(OUT/'source_lock.json',source)
    bars=pd.read_parquet(paths[-2]);folds=json.loads(paths[-1].read_text())
    f=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy();frames=[];metrics=[];alphas=[]
    for fold in folds:
        fid=fold['fold'];fd=OUT/f'fold_{fid:02d}';fd.mkdir()
        old=V9/'folds'/fd.name;p=np.load(old/'preprocess.npz')
        ret,raw,event=price_events(bars.close.to_numpy(),seg,float(p['event_threshold']),5)
        eligible,y=samples(f,seg,ret,raw,event,5)
        s=encode(f,seg,p['bounds']);x,valid=delayed_history(s,f,seg,cfg['delays'])
        train=bars.trading_date.isin(fold['train_dates']).to_numpy()&eligible
        val=bars.trading_date.isin(fold['validation_dates']).to_numpy()&eligible
        future=bars.trading_date.isin(fold['forecast_dates']).to_numpy()&eligible
        predictions={};rawp,model=fit_ridge(x,y,train,val,cfg)
        predictions['input_ridge']=rawp
        np.savez_compressed(fd/'input_ridge.npz',**model)
        for arm in ['random','unsupervised','true_feedback','mismatched']:
            values={kind:[] for kind in ['spike','drive']}
            for seed in cfg['seeds']:
                if arm=='random':
                    m=np.load(ROOT/f'independent_study/results/v8_finance/folds/{fd.name}/seed_{seed}.npz')
                    w,theta=m['initial'],m['random_theta']
                else:
                    m=np.load(old/f'{arm}_{seed}.npz');w,theta=m['hidden'],m['theta']
                drive=x@w.T;spikes=(drive>theta).astype(float)
                for kind,z in [('spike',spikes),('drive',drive)]:
                    pred,model=fit_ridge(z,y,train,val,cfg);values[kind].append(pred)
                    np.savez_compressed(fd/f'{arm}_{kind}_{seed}.npz',**model)
                    alphas.append(dict(fold=fid,arm=arm,kind=kind,seed=seed,alpha=model['alpha'],validation_brier=model['validation_brier']))
            for kind,v in values.items():predictions[f'{arm}_{kind}']=np.mean(v,axis=0)
        oldval=pd.read_parquet(old/'validation.parquet').set_index('bar_id')
        oldfuture=pd.read_parquet(old/'forecast.parquet').set_index('bar_id')
        for arm in ['unsupervised','true_feedback','mismatched','current_rv']:
            a=np.zeros(len(bars));a[oldval.index]=oldval[arm];a[oldfuture.index]=oldfuture[arm]
            predictions[f'original_{arm}']=a
        for name,score in predictions.items():
            ref=score[val];calibrated=mid_percentile(score[future],ref)
            g=bars.loc[future,['bar_id','trading_date']].copy();g['fold']=fid;g['model']=name
            g['target']=y[future];g['score']=calibrated;g['raw_score']=score[future]
            g['brier']=(np.clip(score[future],0,1)-y[future])**2 if not name.startswith('original_') else np.nan
            frames.append(g)
            metrics.append(dict(fold=fid,model=name,**ranking(y[future],calibrated),brier=float(g.brier.mean())))
        print('probe fold',fid,flush=True)
    result=pd.concat(frames,ignore_index=True);result.to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.DataFrame(alphas).to_csv(OUT/'readout_selection.csv',index=False)
    summary=pd.DataFrame([dict(model=m,n=len(g),**ranking(g.target,g.score),brier=g.brier.mean()) for m,g in result.groupby('model')])
    summary.to_csv(OUT/'metrics.csv',index=False)
    # Paired date blocks compare the same representation under different readouts.
    groups={m:g.sort_values('bar_id') for m,g in result.groupby('model')}
    reference=next(iter(groups.values()));days=reference.trading_date.unique();y=reference.target.to_numpy()
    ix=[np.flatnonzero(reference.trading_date.to_numpy()==d) for d in days]
    pairs=[('true_feedback_spike','original_true_feedback'),('true_feedback_spike','random_spike'),
           ('unsupervised_spike','random_spike'),('true_feedback_drive','input_ridge'),
           ('true_feedback_spike','input_ridge')]
    diff={(a,b,key):[] for a,b in pairs for key in ['auc','ap']};rng=np.random.default_rng(20261029)
    for _ in range(1000):
        starts=rng.integers(0,len(days)-4,int(np.ceil(len(days)/5)))
        ds=np.concatenate([np.arange(a,a+5) for a in starts])[:len(days)];index=np.concatenate([ix[i] for i in ds])
        needed=set(a for pair in pairs for a in pair)
        values={m:ranking(y[index],groups[m].score.to_numpy()[index]) for m in needed}
        for (a,b,key),v in diff.items():v.append(values[a][key]-values[b][key])
    st=summary.set_index('model');out=[]
    for (a,b,key),v in diff.items():
        lo,hi=np.quantile(v,[.025,.975]);out.append(dict(left=a,right=b,metric=key,difference=st.loc[a,key]-st.loc[b,key],lower=lo,upper=hi))
    pd.DataFrame(out).to_csv(OUT/'paired_intervals.csv',index=False)
    for path,h in source.items():assert sha256(ROOT/path)==h,path
    dump_json(OUT/'status.json',dict(status='complete',folds=23,readouts=23*(1+4*2*3),quiet_samples=len(reference),
        sources_unchanged=True,old_models_frozen=True))
    print(summary.to_string(index=False))


if __name__=='__main__':main()

import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from research_v10.state_core import ProbabilitySTDP,categories,fit_variance,qlike
from research_v10.state_run import ROOT,HERE,OUT as STATES,load,metric

OUT=ROOT/'independent_study/results/v10_increment'


def main():
    if OUT.exists():raise RuntimeError('Refuse to overwrite incremental experiment')
    assert json.loads((STATES/'stage_b_status.json').read_text())['status']=='complete'
    OUT.mkdir(parents=True);cfg=json.loads((HERE/'config.json').read_text())
    k=json.loads((STATES/'selection_lock.json').read_text())['selected_hidden']
    sources=[HERE/'increment.py',HERE/'STAGE_C.md',HERE/'state_core.py',HERE/'config.json',STATES/'selection_lock.json']
    dump_json(OUT/'source_lock.json',{str(p.relative_to(ROOT)):sha256(p) for p in sources})
    bars,meta,folds,y,base,current=load();ids=meta.bar_id.to_numpy();features=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy()
    frames=[];choices=[]
    for fold in folds:
        fid=fold['fold'];parent=STATES/f'develop_k{k}' if fid in cfg['development_folds'] else STATES/'audit'
        old=parent/f'fold_{fid:02d}';fd=OUT/f'fold_{fid:02d}';fd.mkdir()
        bounds=np.load(old/'preprocess.npz')['bounds'];cat,raw=categories(features,bounds,seg,ids,cfg['delays'])
        xbase=np.c_[base,raw];train=meta.trading_date.isin(fold['train_dates']).to_numpy()
        val=meta.trading_date.isin(fold['validation_dates']).to_numpy();future=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
        rawpred=np.zeros(len(meta))
        for split in ['validation','forecast']:
            frame=pd.read_parquet(old/f'{split}.parquet');rawpred[frame.sample_id.to_numpy()]=frame.raw
        predictions={'raw':rawpred}
        for kind in ['stdp','random','shuffled']:
            pieces=[]
            for seed in cfg['seeds']:
                saved=np.load(old/f'{kind}_{seed}.npz');net=ProbabilitySTDP(cat.shape[1],cfg['bins'],k,seed=seed);net.q=saved['q']
                state,_=net.posterior(cat)
                pred,model=fit_variance(np.c_[xbase,state],y,train,val,cfg['ridge_alphas'],cfg['target_variance_floor'])
                pieces.append(pred);np.savez_compressed(fd/f'{kind}_{seed}_readout.npz',**model)
            pred=np.mean(pieces,axis=0);predictions[f'raw_plus_{kind}']=pred
            trials=[(float(qlike(y[val],(1-w)*rawpred[val]+w*pred[val]).mean()),w) for w in [0.,.25,.5,.75,1.]]
            loss,w=min(trials);predictions[f'gated_{kind}']=(1-w)*rawpred+w*pred
            choices.append(dict(fold=fid,model=kind,weight=w,validation_qlike=loss))
        for split,mask in [('validation',val),('forecast',future)]:
            frame=meta.loc[mask,['sample_id','bar_id','trading_date','datetime','contract_id']].copy();frame['fold']=fid;frame['realized']=y[mask]
            for name,values in predictions.items():frame[name]=values[mask]
            frame.to_parquet(fd/f'{split}.parquet',index=False)
            if split=='forecast':frames.append(frame)
        dump_json(fd/'lock.json',dict(fold=fold,source_hidden_directory=str(old.relative_to(ROOT)),files={p.name:sha256(p) for p in fd.iterdir() if p.is_file()}))
        print('increment fold',fid,flush=True)
    allframe=pd.concat(frames,ignore_index=True);allframe.to_parquet(OUT/'forecasts.parquet',index=False)
    pd.DataFrame(choices).to_csv(OUT/'gates.csv',index=False)
    names=list(predictions);records=[];intervals=[]
    for scope,frame in [('development',allframe[allframe.fold.isin(cfg['development_folds'])]),
                         ('audit',allframe[~allframe.fold.isin(cfg['development_folds'])]),('all_exploratory',allframe)]:
        records.extend(dict(scope=scope,model=m,**metric(frame.realized.to_numpy(),frame[m].to_numpy())) for m in names)
        days=frame.trading_date.unique();counts=np.array([(frame.trading_date==d).sum() for d in days])
        losses={m:qlike(frame.realized.to_numpy(),frame[m].to_numpy()) for m in names}
        sums={m:np.array([losses[m][frame.trading_date.to_numpy()==d].sum() for d in days]) for m in names}
        rng=np.random.default_rng(cfg['seed']);weights=[]
        for _ in range(cfg['bootstrap_repeats']):
            starts=rng.integers(0,len(days)-4,int(np.ceil(len(days)/5)));ix=np.concatenate([np.arange(s,s+5) for s in starts])[:len(days)]
            weights.append(np.bincount(ix,minlength=len(days)))
        weights=np.array(weights);denom=weights@counts
        for left in ['raw_plus_stdp','gated_stdp']:
            others=['raw',left.replace('stdp','random'),left.replace('stdp','shuffled')]
            for other in others:
                diff=(weights@(sums[left]-sums[other]))/denom;lo,hi=np.quantile(diff,[.025,.975])
                intervals.append(dict(scope=scope,left=left,right=other,difference=float((losses[left]-losses[other]).mean()),lower=lo,upper=hi))
    pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    dump_json(OUT/'status.json',dict(status='complete',folds=23,hidden_retrained=0,readouts=207,forecasts=len(allframe)))
    print(pd.DataFrame(records).to_string(index=False));print(pd.DataFrame(intervals).to_string(index=False))


if __name__=='__main__':main()

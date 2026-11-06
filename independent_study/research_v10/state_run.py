from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from research_v10.state_core import (ProbabilitySTDP,fit_bins,categories,one_hot,shuffle_delays,
                                     fit_variance,predict_variance,risk_features,qlike)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OUT=ROOT/'independent_study/results/v10_state_risk'


def load():
    bars=pd.read_parquet(ROOT/'independent_study/results/v1/bars_features.parquet')
    meta=pd.read_parquet(ROOT/'independent_study/results/v2/samples.parquet')
    folds=json.loads((ROOT/'independent_study/results/v2/folds.json').read_text())
    ids=meta.bar_id.to_numpy();seg=bars.segment_id.to_numpy()
    assert np.array_equal(meta.segment_id,seg[ids])
    assert np.array_equal(meta.contract_id,bars.contract_id.iloc[ids])
    e=meta.entry_bar_id.to_numpy();path=bars.open.to_numpy()[e[:,None]+np.arange(6)]
    assert (seg[e[:,None]+np.arange(6)]==seg[ids,None]).all()
    y=np.sum(np.diff(np.log(path),axis=1)**2,axis=1)*1e8
    np.testing.assert_allclose(np.sqrt(y),meta.future_rv,atol=1e-11,rtol=0)
    base,current=risk_features(bars,ids)
    np.testing.assert_allclose(np.sqrt(current),meta.current_rv,atol=1e-11,rtol=0)
    return bars,meta,folds,y,base,current


def metric(y,p):
    q=qlike(y,p)-np.log(y)-1
    rv=np.sqrt(y);ph=np.sqrt(p)
    return dict(n=len(y),qlike=float(q.mean()),rv_ic=float(np.corrcoef(rv,ph)[0,1]),
                rv_rank_ic=float(pd.Series(rv).rank().corr(pd.Series(ph).rank())),
                mean_prediction=float(p.mean()),mean_realized=float(y.mean()))


def execute(fold,k,cfg,bars,meta,y,base,current,dest):
    fid=fold['fold'];fd=dest/f'fold_{fid:02d}';fd.mkdir(parents=True,exist_ok=False)
    ids=meta.bar_id.to_numpy();features=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy()
    tb=bars.trading_date.isin(fold['train_dates']).to_numpy()&np.isfinite(features).all(axis=1)
    train=meta.trading_date.isin(fold['train_dates']).to_numpy();val=meta.trading_date.isin(fold['validation_dates']).to_numpy()
    forecast=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
    bounds=fit_bins(features,tb,cfg['bins']);cat,raw=categories(features,bounds,seg,ids,cfg['delays'])
    hot=one_hot(cat,cfg['bins'])
    pred={};models={};validation={}
    for name,x in [('har',base),('raw',np.c_[base,raw]),('encoded',np.c_[base,hot])]:
        p,m=fit_variance(x,y,train,val,cfg['ridge_alphas'],cfg['target_variance_floor'])
        pred[name]=p;models[name]=m;np.savez_compressed(fd/f'{name}_readout.npz',**m)
        np.testing.assert_allclose(p,predict_variance(m,x),atol=1e-9,rtol=1e-12)
    pred['current']=current
    diags=[];states=[];seedparts=[];pieces={name:[] for name in ['stdp','random','shuffled','state_only']}
    for seed in cfg['seeds']:
        wrong=shuffle_delays(cat[train],len(FEATURES),len(cfg['delays']),seed+2)
        for kind in ['stdp','random','shuffled']:
            network=ProbabilitySTDP(cat.shape[1],cfg['bins'],k,cfg['eta'],cfg['epochs'],seed,cfg['probability_floor'])
            if kind!='random':network.fit(cat[train] if kind=='stdp' else wrong,seed+1)
            post,surprise=network.posterior(cat)
            p,m=fit_variance(np.c_[base,post],y,train,val,cfg['ridge_alphas'],cfg['target_variance_floor'])
            pieces[kind].append(p)
            np.savez_compressed(fd/f'{kind}_{seed}.npz',q=network.q,initial=network.initial,counts=network.counts,
                                **{f'readout_{key}':v for key,v in m.items()})
            if kind=='stdp':
                pp,mm=fit_variance(post,y,train,val,cfg['ridge_alphas'],cfg['target_variance_floor'])
                pieces['state_only'].append(pp);np.savez_compressed(fd/f'state_only_{seed}.npz',**mm)
            p_for=post[forecast];p_train=post[train]
            diags.append(dict(fold=fid,hidden=k,seed=seed,model=kind,train_units=int((p_train.argmax(axis=1)[:,None]==np.arange(k)).any(axis=0).sum()),
                forecast_units=int((p_for.argmax(axis=1)[:,None]==np.arange(k)).any(axis=0).sum()),
                train_entropy=float((-p_train*np.log(np.maximum(p_train,1e-15))).sum(axis=1).mean()),
                forecast_entropy=float((-p_for*np.log(np.maximum(p_for,1e-15))).sum(axis=1).mean()),
                weight_change=float(abs(network.q-network.initial).mean()),alpha=m['alpha'],
                validation_qlike=m['validation_qlike']))
            sf=meta.loc[forecast,['bar_id','trading_date']].copy();sf['model']=kind;sf['seed']=seed;sf['fold']=fid
            sf['prediction']=p[forecast];sf['realized']=y[forecast];seedparts.append(sf)
            if kind=='stdp':
                for j in range(k):
                    a=p_train[:,j];b=p_for[:,j]
                    states.append(dict(fold=fid,hidden=k,seed=seed,unit=j,train_mass=float(a.sum()),forecast_mass=float(b.sum()),
                        train_rv=float(a@np.sqrt(y[train])/a.sum()),forecast_rv=float(b@np.sqrt(y[forecast])/max(b.sum(),1e-12)),
                        train_efficiency=float(a@meta.current_efficiency.to_numpy()[train]/a.sum()),
                        forecast_efficiency=float(b@meta.current_efficiency.to_numpy()[forecast]/max(b.sum(),1e-12)),
                        train_volume_surprise=float(a@features[ids[train],FEATURES.index('volume_surprise')]/a.sum())))
    for kind,values in pieces.items():pred[kind]=np.mean(values,axis=0)
    for split,mask in [('validation',val),('forecast',forecast)]:
        frame=meta.loc[mask,['sample_id','bar_id','datetime','trading_date','contract_id','segment_id','entry_bar_id','exit_bar_id']].copy()
        frame['fold']=fid;frame['realized']=y[mask]
        for name,values in pred.items():frame[name]=values[mask]
        frame.to_parquet(fd/f'{split}.parquet',index=False)
    np.savez_compressed(fd/'preprocess.npz',bounds=bounds)
    dump_json(fd/'lock.json',dict(fold=fold,hidden=k,files={p.name:sha256(p) for p in fd.iterdir() if p.is_file()}))
    pd.DataFrame(diags).to_csv(fd/'diagnostics.csv',index=False);pd.DataFrame(states).to_csv(fd/'state_profiles.csv',index=False)
    pd.concat(seedparts,ignore_index=True).to_parquet(fd/'seed_forecasts.parquet',index=False)
    return frame


def summarize(dest,ids):
    frames=[pd.read_parquet(dest/f'fold_{i:02d}/forecast.parquet') for i in ids]
    data=pd.concat(frames,ignore_index=True);data.to_parquet(dest/'forecasts.parquet',index=False)
    names=['current','har','raw','encoded','stdp','random','shuffled','state_only']
    summary=pd.DataFrame([dict(model=m,**metric(data.realized.to_numpy(),data[m].to_numpy())) for m in names])
    summary.to_csv(dest/'metrics.csv',index=False)
    per=pd.DataFrame([dict(fold=f,model=m,**metric(g.realized.to_numpy(),g[m].to_numpy())) for f,g in data.groupby('fold') for m in names])
    per.to_csv(dest/'fold_metrics.csv',index=False)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','audit']);args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());bars,meta,folds,y,base,current=load();start=time.time()
    if args.stage=='develop':
        if OUT.exists():raise RuntimeError('Refuse to overwrite development')
        OUT.mkdir(parents=True)
        sources=[HERE/'state_core.py',HERE/'state_run.py',HERE/'STAGE_B.md',HERE/'config.json',
                 ROOT/'independent_study/results/v1/bars_features.parquet',ROOT/'independent_study/results/v2/samples.parquet',
                 ROOT/'independent_study/results/v2/folds.json']
        dump_json(OUT/'source_lock.json',{str(p.relative_to(ROOT)):sha256(p) for p in sources})
        results=[]
        for k in cfg['hidden_candidates']:
            dest=OUT/f'develop_k{k}';dest.mkdir()
            for f in folds:
                if f['fold'] not in cfg['development_folds']:continue
                execute(f,k,cfg,bars,meta,y,base,current,dest)
                print(f'develop k={k} fold={f["fold"]}; {time.time()-start:.1f}s',flush=True)
            summary=summarize(dest,cfg['development_folds']);loss=float(summary.set_index('model').loc['stdp','qlike'])
            results.append(dict(hidden=k,development_qlike=loss));print(summary.to_string(index=False),flush=True)
        best=min(results,key=lambda a:a['development_qlike']);small=min(results,key=lambda a:a['hidden'])
        if small['development_qlike']<=best['development_qlike']*1.005:best=small
        dump_json(OUT/'selection_lock.json',dict(selected_hidden=best['hidden'],candidates=results,config=cfg,
            chosen_before_audit=True,criterion='Development forecast normalized QLIKE; within 0.5% choose smaller',
            development_files={str(p.relative_to(OUT)):sha256(p) for p in OUT.glob('develop_k*/metrics.csv')}))
    else:
        lock=json.loads((OUT/'selection_lock.json').read_text());k=lock['selected_hidden'];dest=OUT/'audit';dest.mkdir(exist_ok=False)
        for path,expected in json.loads((OUT/'source_lock.json').read_text()).items():assert sha256(ROOT/path)==expected,path
        chosen=[]
        for f in folds:
            if f['fold'] in cfg['development_folds']:continue
            execute(f,k,cfg,bars,meta,y,base,current,dest);chosen.append(f['fold'])
            print(f'audit k={k} fold={f["fold"]}; {time.time()-start:.1f}s',flush=True)
        summary=summarize(dest,chosen);print(summary.to_string(index=False),flush=True)
        dump_json(OUT/'stage_b_status.json',dict(status='complete',selected_hidden=k,development_folds=cfg['development_folds'],audit_folds=chosen))


if __name__=='__main__':main()

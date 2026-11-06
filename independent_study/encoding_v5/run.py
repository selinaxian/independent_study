"""Train only the new encoding arm; reuse locked v2 comparisons."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from iclab.data import sha256
from iclab.neural import encode
from diagnostics_v2.common import config as oldconfig,save_json
from diagnostics_v2.data import load,apply_scale
from diagnostics_v2.engine import FastSNN
from diagnostics_v2.probes import fit_probe,predict_probe
from .encoding import latency,time_codes,fidelity

ROOT=Path(__file__).parent


def settings():return json.loads((ROOT/'config.json').read_text())


def sources():
    files=list(ROOT.glob('*.py'))+[ROOT/'config.json',ROOT/'PROTOCOL.md']
    files+=list(Path('independent_study/diagnostics_v2').glob('*.py'))+list(Path('independent_study/iclab').glob('*.py'))
    return {str(p):sha256(p) for p in sorted(files)}


def init():
    cfg=settings();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():
        assert json.loads((out/'run_manifest.json').read_text())['sources']==sources()
        return
    old={}
    for version in ['v1','v2','v3','v4']:
        root=out.parent/version
        for rel,h in json.loads((root/'artifact_manifest.json').read_text()).items():
            p=root/rel;assert sha256(p)==h,str(p);old[str(p)]=h
    inputs=[Path(cfg['source'])/name for name in ['dataset.npz','samples.parquet','folds.json','ensemble_predictions.npz','forecast_samples.parquet']]
    save_json(out/'run_manifest.json',{'sources':sources(),'inputs':{str(p):sha256(p) for p in inputs},
         'old_artifacts':old,'locked_at_utc':datetime.now(timezone.utc).isoformat()})


def run(worker=0,workers=1):
    cfg=settings();old,v1=oldconfig();out=Path(cfg['output']);torch.set_num_threads(cfg['torch_threads'])
    assert json.loads((out/'run_manifest.json').read_text())['sources']==sources()
    raw,y,meta,bars,folds=load(old)
    for fold in folds:
        f=fold['fold']
        if f%workers!=worker:continue
        dest=out/'folds'/f'fold_{f:02d}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        start=time.monotonic()
        train=meta.trading_date.isin(fold['train_dates']).to_numpy();val=meta.trading_date.isin(fold['validation_dates']).to_numpy();test=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
        indices=np.flatnonzero(train|val|test);local=meta.iloc[indices].copy()
        tm,vm,em=train[indices],val[indices],test[indices];yy=y[indices]
        assert max(local.loc[tm,'trading_date'])<min(local.loc[vm,'trading_date'])<min(local.loc[em,'trading_date'])
        original=Path(cfg['source'])/'folds'/f'fold_{f:02d}'/'rolling_all'/'lock.json'
        oldlock=json.loads(original.read_text());enc=Path(oldlock['encoder_manifest']).parent
        scale=np.load(enc/'scale.npy');np.save(dest/'scale.npy',scale)
        x=apply_scale(raw[indices],scale);s=latency(x)
        reps={'time_code':time_codes(s).reshape(len(x),-1),
              'counts_minute':s.reshape(len(x),12,8,24).sum(axis=2).reshape(len(x),-1).astype(np.float32)}
        diag={};hashes={};fidelity_rows=[]
        for split,mask in [('train',tm),('validation',vm),('forecast',em)]:
            fidelity_rows.append({'fold':f,'encoding':'latency','seed':-1,'split':split,**fidelity(x[mask],s[mask],'latency')})
        for seed in cfg['seeds']:
            rate=encode(x,local.sample_id.to_numpy(),'rate',seed)
            for split,mask in [('train',tm),('validation',vm),('forecast',em)]:
                fidelity_rows.append({'fold':f,'encoding':'rate','seed':seed,'split':split,**fidelity(x[mask],rate[mask],'rate')})
            for mode in cfg['modes']:
                name=f'{mode}_s{seed}';model=FastSNN(24,v1['network'],seed)
                model.fit(s[tm],plastic=mode=='stdp')
                model.save(dest/(name+'_snn.npz'));hashes[name]=sha256(dest/(name+'_snn.npz'))
                save_json(dest/(name+'_encoder_lock.json'),{'train_dates':fold['train_dates'],'sha256':hashes[name],
                    'locked_at_utc':datetime.now(timezone.utc).isoformat()})
                w,t=model.weight.clone(),model.theta.clone();r=model.transform(s)
                torch.testing.assert_close(w,model.weight,rtol=0,atol=0);torch.testing.assert_close(t,model.theta,rtol=0,atol=0)
                if mode=='random':torch.testing.assert_close(model.weight,model.initial_weight,rtol=0,atol=0)
                restored=FastSNN(24,v1['network'],seed).load(dest/(name+'_snn.npz'))
                np.testing.assert_array_equal(restored.batch(s[:17]),model.batch(s[:17]))
                reps[name]=r;diag[name]=model.diagnostics(r[tm])
                if mode=='stdp':reps[f'raw_plus_stdp_s{seed}']=np.c_[x.reshape(len(x),-1),r]
        models={};probehash={}
        for name,r in reps.items():
            model=fit_probe(r[tm],r[vm],yy[tm],yy[vm],'linear',old['alphas'])
            np.savez_compressed(dest/(name+'_probe.npz'),**model)
            models[name]=model;probehash[name]=sha256(dest/(name+'_probe.npz'))
        save_json(dest/'lock.json',{'fold':fold,'encoding':'latency','old_lock_sha256':sha256(original),
             'scale_sha256':sha256(dest/'scale.npy'),'neural_hashes':hashes,'probe_hashes':probehash,
             'locked_at_utc':datetime.now(timezone.utc).isoformat()})
        forecasts={};validation={}
        for name,model in models.items():
            forecasts[name]=predict_probe(model,reps[name][em]);validation[name]=predict_probe(model,reps[name][vm])
        for mode in ['stdp','random','raw_plus_stdp']:
            forecasts[mode]=np.mean([forecasts[f'{mode}_s{s}'] for s in cfg['seeds']],axis=0)
            validation[mode]=np.mean([validation[f'{mode}_s{s}'] for s in cfg['seeds']],axis=0)
        fm=local[em].copy();fm['fold']=f;fm.to_parquet(dest/'samples.parquet',index=False)
        local[vm].to_parquet(dest/'validation_samples.parquet',index=False)
        np.savez_compressed(dest/'predictions.npz',**forecasts)
        np.savez_compressed(dest/'validation_predictions.npz',**validation)
        np.savez_compressed(dest/'forecast_representations.npz',**{n:r[em] for n,r in reps.items()})
        save_json(dest/'diagnostics.json',diag);pd.DataFrame(fidelity_rows).to_csv(dest/'fidelity.csv',index=False)
        save_json(dest/'complete.json',{'fold':f,'rows':int(em.sum()),'seconds':time.monotonic()-start,
            'files':{str(p.name):sha256(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='complete.json'}})
        print(f'LATENCY FOLD {f+1}/23 COMPLETE ({time.monotonic()-start:.1f}s)',flush=True)
    save_json(out/f'worker_{worker}_complete.json',{'worker':worker,'workers':workers,'sources':sources()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','run']);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();init() if a.stage=='init' else run(a.worker,a.workers)

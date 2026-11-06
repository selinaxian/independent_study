"""Reconstruct past validation inference; never refit existing models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from iclab.data import sha256
from iclab.neural import encode
from diagnostics_v2.common import config,save_json
from diagnostics_v2.data import load,apply_scale
from diagnostics_v2.engine import FastSNN
from diagnostics_v2.probes import predict_probe
from .core import fit_risk_rule


ROOT=Path(__file__).parent


def settings():return json.loads((ROOT/'config.json').read_text())


def read_npz(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def prepare():
    cfg=settings();old,v1=config();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    original=Path(cfg['source']);torch.set_num_threads(1)
    if (out/'prepared.json').exists():return
    raw,y,meta,bars,folds=load(old)
    before={}
    for version in ['v1','v2','v3']:
        root=original.parent/version
        manifest=json.loads((root/'artifact_manifest.json').read_text())
        for relative,h in manifest.items():
            p=root/relative
            assert sha256(p)==h,str(p)
            before[str(p)]=h
    inputs=[original/'dataset.npz',original/'samples.parquet',original/'ensemble_predictions.npz',
            original/'forecast_samples.parquet',original/'folds.json',Path(v1['output_dir'])/'bars_features.parquet']
    save_json(out/'source_lock.json',{'old_artifacts':before,'inputs':{str(p):sha256(p) for p in inputs},
               'protocol_sha256':sha256(ROOT/'PROTOCOL.md'),'config_sha256':sha256(ROOT/'config.json')})
    audits=[];allforecast=[]
    for fold in folds:
        f=fold['fold'];dest=out/'calibration'/f'fold_{f:02d}';dest.mkdir(parents=True,exist_ok=True)
        d=original/'folds'/f'fold_{f:02d}'
        lock=json.loads((d/'rolling_all'/'lock.json').read_text())
        for p,h in lock['probe_hashes'].items():assert sha256(p)==h,p
        for p,h in json.loads(Path(lock['encoder_manifest']).read_text())['model_hashes'].items():assert sha256(p)==h,p
        enc=Path(lock['encoder_manifest']).parent
        val=meta.trading_date.isin(fold['validation_dates']).to_numpy()
        forecast=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
        indices=np.flatnonzero(val|forecast);local=meta.iloc[indices]
        isval=val[indices];istest=forecast[indices]
        x=apply_scale(raw[indices],np.load(enc/'scale.npy'))
        rawrep=x.reshape(len(x),-1)
        modeldir=Path(next(iter(lock['probe_hashes']))).parent
        preds={'raw_history':predict_probe(read_npz(modeldir/'raw_history__linear.npz'),rawrep)}
        pieces={'stdp':[],'random':[],'raw_plus_stdp':[]}
        for seed in old['seeds']:
            spikes=encode(x,local.sample_id.to_numpy(),old['encoding'],seed)
            for mode in ['stdp','random']:
                model=FastSNN(24,v1['network'],seed).load(enc/f'{mode}_s{seed}_snn.npz')
                w,t=model.weight.clone(),model.theta.clone()
                rep=model.transform(spikes)
                torch.testing.assert_close(w,model.weight,rtol=0,atol=0)
                torch.testing.assert_close(t,model.theta,rtol=0,atol=0)
                pieces[mode].append(predict_probe(read_npz(modeldir/f'{mode}_s{seed}__linear.npz'),rep))
                if mode=='stdp':
                    pieces['raw_plus_stdp'].append(predict_probe(read_npz(modeldir/f'raw_plus_stdp_s{seed}__linear.npz'),np.c_[rawrep,rep]))
        preds.update({k:np.mean(v,axis=0) for k,v in pieces.items()})
        saved=read_npz(d/'predictions.npz');maxdiff=0.
        for name,p in preds.items():
            expected=saved[f'rolling_all__{name}__linear']
            np.testing.assert_allclose(p[istest],expected,rtol=0,atol=1e-8)
            maxdiff=max(maxdiff,float(np.max(abs(p[istest]-expected))))
        ix=local.bar_id.to_numpy();past=(bars.close.to_numpy()[ix]/bars.close.to_numpy()[ix-5]-1)*10000
        preds['persistence']=np.c_[past,abs(past),local.current_rv.to_numpy()]
        rules={};vp={};fp={}
        for model,p in preds.items():
            for target,col in cfg['targets'].items():
                key=model+'__'+target
                rules[key]=fit_risk_rule(p[isval,col],local.loc[isval,'trading_date'].to_numpy(),fold['forecast_dates'][0],cfg['filter_quantile'],cfg['risk_floor_bps'])
                vp[key]=p[isval,col];fp[key]=p[istest,col]
        np.savez_compressed(dest/'validation_predictions.npz',sample_id=local.loc[isval,'sample_id'].to_numpy(),**vp)
        np.savez_compressed(dest/'forecast_predictions.npz',sample_id=local.loc[istest,'sample_id'].to_numpy(),**fp)
        save_json(dest/'rules.json',{'fold':fold,'rules':rules,'source_lock_sha256':sha256(d/'rolling_all'/'lock.json'),
                  'validation_predictions_sha256':sha256(dest/'validation_predictions.npz')})
        audits.append({'fold':f,'validation_rows':int(isval.sum()),'forecast_rows':int(istest.sum()),'reproduction_max_abs_diff':maxdiff})
        fm=local[istest].copy();fm['fold']=f
        for key,values in fp.items():fm[key]=values
        allforecast.append(fm)
        print(f'calibration {f+1}/{len(folds)}; reproduction difference={maxdiff:.3g}',flush=True)
    pd.concat(allforecast,ignore_index=True).to_parquet(out/'risk_forecasts.parquet',index=False)
    save_json(out/'prepared.json',{'folds':audits,'source_lock_sha256':sha256(out/'source_lock.json'),
                'retrained':False,'validation_only_calibration':True})


if __name__=='__main__':prepare()

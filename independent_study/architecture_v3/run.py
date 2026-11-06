"""Rolling structural ablation; disjoint folds may run in separate processes."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from iclab.data import sha256
from iclab.neural import encode
from diagnostics_v2.common import save_json,source_hashes
from diagnostics_v2.data import load,apply_scale
from diagnostics_v2.probes import fit_probe,predict_probe
from diagnostics_v2.rolling import ensemble
from .neural import SoftLayer


def settings():
    cfg=json.loads(Path('independent_study/architecture_v3/config.json').read_text())
    c2=json.loads(Path(cfg['v2_config']).read_text());v1=json.loads(Path(cfg['v1_config']).read_text())
    return cfg,c2,v1


def sources():
    h=source_hashes()
    for p in sorted(Path('independent_study/architecture_v3').glob('*')):
        if p.suffix in ['.py','.json','.md']:h[str(p)]=sha256(p)
    return h


def manifest(cfg):
    v2=Path(cfg['v2_dir'])
    inputs=[v2/n for n in ['artifact_manifest.json','dataset.npz','samples.parquet','folds.json','state_definition.json','verification.json']]
    for f in sorted((v2/'folds').glob('fold_*')):
        inputs += [f/'complete.json',f/'predictions.npz',f/'samples.parquet']
        inputs += [f/'rolling_all'/'encoder'/'scale.npy'] if f.name!='fold_00' else [v2/'initial_frozen'/'scale.npy']
    return {'sources':sources(),'inputs':{str(p):sha256(p) for p in inputs}}


def initialize(cfg):
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    assert all(json.loads((Path(cfg['v2_dir'])/'verification.json').read_text()).values())
    f=out/'run_manifest.json';m=manifest(cfg)
    if f.exists():assert json.loads(f.read_text())==m,'Existing run differs; preserve it'
    else:save_json(f,m)
    print('v3 sources and inherited v2 inputs locked',flush=True)


def fit_layer(spikes,train,network,seed,adaptive,plastic,path,fold,parent=None):
    start=time.monotonic();layer=SoftLayer(spikes.shape[2],network,seed,adaptive)
    layer.fit(spikes[train],plastic=plastic)
    layer.save(path)
    save_json(path.with_suffix('.lock.json'),{'train_dates':fold['train_dates'],'adaptive':adaptive,'plastic':plastic,'seed':seed,'network':network,'parent':parent,'model_sha256':sha256(path),'locked_at_utc':datetime.now(timezone.utc).isoformat()})
    w,theta=layer.weight.clone(),layer.theta.clone()
    r,seq=layer.transform_with_spikes(spikes)
    torch.testing.assert_close(w,layer.weight,rtol=0,atol=0);torch.testing.assert_close(theta,layer.theta,rtol=0,atol=0)
    restored=SoftLayer(spikes.shape[2],network,seed,adaptive).load(path)
    np.testing.assert_array_equal(layer.batch(spikes[:32]),restored.batch(spikes[:32]))
    print(f'{path.stem}: {time.monotonic()-start:.1f}s',flush=True)
    return layer,r,seq


def run_fold(cfg,c2,v1,raw,y,meta,bars,fold):
    out=Path(cfg['output_dir']);dest=out/'folds'/f"fold_{fold['fold']:02d}"
    if (dest/'complete.json').exists():
        done=json.loads((dest/'complete.json').read_text());assert sha256(dest/'predictions.npz')==done['predictions_sha256']
        print('reuse '+dest.name,flush=True);return
    dest.mkdir(parents=True,exist_ok=True);t0=time.monotonic()
    idx=np.flatnonzero(meta.trading_date.isin(fold['train_dates']+fold['validation_dates']+fold['forecast_dates']).to_numpy())
    local=meta.iloc[idx];train=local.trading_date.isin(fold['train_dates']).to_numpy();val=local.trading_date.isin(fold['validation_dates']).to_numpy();future=local.trading_date.isin(fold['forecast_dates']).to_numpy()
    assert max(fold['train_dates'])<min(fold['validation_dates']) and max(fold['validation_dates'])<min(fold['forecast_dates'])
    v2=Path(cfg['v2_dir']);old=v2/'folds'/dest.name
    scale_path=v2/'initial_frozen'/'scale.npy' if fold['fold']==0 else old/'rolling_all'/'encoder'/'scale.npy'
    x=apply_scale(raw[idx],np.load(scale_path));ym=y[idx]
    testmeta=local[future].copy();testmeta['fold']=fold['fold'];testmeta.to_parquet(dest/'samples.parquet',index=False)
    predictions={};diagnostics={};model_files={};probe_files={};saved_reps={}
    def probe(name,rep):
        m=fit_probe(rep[train],rep[val],ym[train],ym[val],'linear',cfg['alphas'])
        file=dest/(name+'__linear_probe.npz');np.savez_compressed(file,**m);probe_files[str(file)]=sha256(file)
        save_json(dest/(name+'__probe_lock.json'),{'fold':fold,'model_sha256':probe_files[str(file)],'locked_at_utc':datetime.now(timezone.utc).isoformat()})
        predictions[name+'__linear']=predict_probe(m,rep[future]);saved_reps[name]=rep[future]
    for seed in cfg['seeds']:
        spikes=encode(x,local.sample_id.to_numpy(),c2['encoding'],seed)
        for mode in cfg['modes']:
            plastic=mode=='stdp'
            name=f'soft_fixed_{mode}_s{seed}';file=dest/(name+'_snn.npz')
            layer,rep,seq=fit_layer(spikes,train,v1['network'],seed,False,plastic,file,fold)
            diagnostics[name]=layer.diagnostics(rep[train]);model_files[str(file)]=sha256(file);probe(name,rep)
            del layer,rep,seq
            name=f'soft_adapt_{mode}_s{seed}';file=dest/(name+'_snn.npz')
            layer,rep,seq=fit_layer(spikes,train,v1['network'],seed,True,plastic,file,fold)
            diagnostics[name]=layer.diagnostics(rep[train]);model_files[str(file)]=sha256(file);probe(name,rep)
            # Greedy depth: layer one frozen. Layer two sees its spikes, not analog raw input.
            first_w,first_theta=layer.weight.clone(),layer.theta.clone()
            second_cfg={**v1['network'],'hidden':cfg['second_hidden']}
            dname=f'deep_layer2_{mode}_s{seed}';dfile=dest/(dname+'_snn.npz')
            second,r2,s2=fit_layer(seq,train,second_cfg,seed+cfg['second_seed_offset'],True,plastic,dfile,fold,parent=str(file))
            torch.testing.assert_close(layer.weight,first_w,rtol=0,atol=0);torch.testing.assert_close(layer.theta,first_theta,rtol=0,atol=0)
            diagnostics[dname]=second.diagnostics(r2[train]);model_files[str(dfile)]=sha256(dfile)
            probe(f'deep_last_{mode}_s{seed}',r2);probe(f'deep_concat_{mode}_s{seed}',np.c_[rep,r2])
            del layer,second,rep,seq,r2,s2
    ensemble(predictions)
    # Forecasts already locked in v2; common samples and causal scaling are identical.
    oldmeta=pd.read_parquet(old/'samples.parquet')
    np.testing.assert_array_equal(oldmeta.sample_id,testmeta.sample_id)
    with np.load(old/'predictions.npz') as a:
        for rep in ['raw_history','risk_control','counts_minute','counts_total','random','stdp','shuffle','raw_plus_stdp']:
            name={'random':'hard_adapt_random','stdp':'hard_adapt_stdp','shuffle':'hard_adapt_shuffle'}.get(rep,rep)
            predictions[name+'__linear']=a[f'rolling_all__{rep}__linear']
        predictions['persistence__rule']=a['baseline__persistence__rule']
    assert all(np.isfinite(v).all() for v in predictions.values())
    np.savez_compressed(dest/'forecast_representations.npz',**saved_reps)
    save_json(dest/'diagnostics.json',diagnostics)
    save_json(dest/'lock.json',{'fold':fold,'model_files':model_files,'probe_files':probe_files,'scale_path':str(scale_path),'scale_sha256':sha256(scale_path),'all_encoder_freeze_reload':True,'first_layer_frozen':True,'run_manifest_sha256':sha256(out/'run_manifest.json')})
    np.savez_compressed(dest/'predictions.npz',**predictions)
    save_json(dest/'complete.json',{'rows':int(future.sum()),'seconds':time.monotonic()-t0,'predictions_sha256':sha256(dest/'predictions.npz'),'lock_sha256':sha256(dest/'lock.json')})
    print(f"FOLD {fold['fold']+1}/23 DONE: {time.monotonic()-t0:.1f}s",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','train']);parser.add_argument('--partition',type=int,default=0);args=parser.parse_args()
    torch.set_num_threads(1);cfg,c2,v1=settings()
    if args.stage=='init':initialize(cfg);return
    assert 0<=args.partition<cfg['workers']
    assert json.loads((Path(cfg['output_dir'])/'run_manifest.json').read_text())==manifest(cfg)
    raw,y,meta,bars,folds=load(c2)
    for fold in folds:
        if fold['fold']%cfg['workers']==args.partition:run_fold(cfg,c2,v1,raw,y,meta,bars,fold)
    save_json(Path(cfg['output_dir'])/f'worker_{args.partition}_complete.json',{'folds':[f['fold'] for f in folds if f['fold']%cfg['workers']==args.partition]})


if __name__=='__main__':main()

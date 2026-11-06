from datetime import datetime,timezone
from pathlib import Path
import argparse,json,re,time
import numpy as np
import pandas as pd
import torch
from iclab.data import sha256
from iclab.neural import encode
from iclab.backtest import simulate
from .common import config,save_json,source_hashes
from .data import prepare,load,fit_scale,apply_scale
from .engine import FastSNN
from .probes import fit_probe,predict_probe


def ensemble(predictions):
    groups={}
    for key,p in list(predictions.items()):
        rep,kind=key.split("__")
        base=re.sub(r"_s\d+$","",rep)
        if base!=rep: groups.setdefault(base+"__"+kind,[]).append(p)
    for key,values in groups.items(): predictions[key]=np.mean(values,axis=0)
    return predictions


def representation_set(cfg,v1,raw,meta,indices,scale,train_dates,path,save_all=False):
    path.mkdir(parents=True,exist_ok=True)
    x=apply_scale(raw[indices],scale); local=meta.iloc[indices]
    tr=local.trading_date.isin(train_dates).to_numpy()
    r={"raw_history":x.reshape(len(x),-1),"risk_control":x[:,-1,[10,11]]}
    diagnostics={}; model_hashes={}
    np.save(path/"scale.npy",scale)
    for seed in cfg["seeds"]:
        base=encode(x,local.sample_id.to_numpy(),cfg["encoding"],seed)
        r[f"counts_total_s{seed}"]=base.sum(axis=1).astype(np.float32)
        r[f"counts_minute_s{seed}"]=base.reshape(-1,12,8,24).sum(axis=2).reshape(len(x),-1).astype(np.float32)
        for mode in cfg["modes"]:
            key=f"{mode}_s{seed}"; start=time.monotonic()
            spikes=encode(x,local.sample_id.to_numpy(),cfg["encoding"],seed,shuffled=True) if mode=="shuffle" else base
            model=FastSNN(24,v1["network"],seed)
            model.fit(spikes[tr],plastic=mode!="random")
            file=path/(key+"_snn.npz"); model.save(file)
            model_hashes[str(file)]=sha256(file)
            # Freeze encoder before extracting any future-window representation.
            save_json(path/(key+"_encoder_lock.json"),{"train_dates":train_dates,"scale":scale,"checkpoint_sha256":model_hashes[str(file)],"locked_at_utc":datetime.now(timezone.utc).isoformat()})
            w,t=model.weight.clone(),model.theta.clone()
            r[key]=model.transform(spikes)
            torch.testing.assert_close(w,model.weight,rtol=0,atol=0); torch.testing.assert_close(t,model.theta,rtol=0,atol=0)
            restored=FastSNN(24,v1["network"],seed).load(file)
            np.testing.assert_array_equal(model.batch(spikes[:32]),restored.batch(spikes[:32]))
            diagnostics[key]=model.diagnostics(r[key][tr])
            if mode=="stdp": r[f"raw_plus_stdp_s{seed}"]=np.c_[r["raw_history"],r[key]]
            print(f"{path.name} {key} trained ({time.monotonic()-start:.1f}s)",flush=True)
    save_json(path/"diagnostics.json",diagnostics)
    save_json(path/"encoder_manifest.json",{"model_hashes":model_hashes,"scale_sha256":sha256(path/"scale.npy"),"train_dates":train_dates,"rows":len(indices),"frozen_inference_and_reload":True})
    if save_all:
        cache=path/"representations";cache.mkdir(exist_ok=True)
        for key,value in r.items(): np.save(cache/(key+".npy"),value)
        save_json(path/"cache_complete.json",{"files":{str(p):sha256(p) for p in cache.glob("*.npy")}})
    return r


def fit_probes(reps,train,val,y,cfg,path):
    path.mkdir(parents=True,exist_ok=True)
    models={}; pt={}; pv={}
    for name,values in reps.items():
        for kind in cfg["probe_types"]:
            key=name+"__"+kind
            m=fit_probe(values[train],values[val],y[train],y[val],kind,cfg["alphas"],components=cfg["rff_components"])
            np.savez_compressed(path/(key+".npz"),**m)
            models[key]=m
            pt[key]=predict_probe(m,values[train]); pv[key]=predict_probe(m,values[val])
    return models,ensemble(pt),ensemble(pv)


def predict_models(models,reps,mask):
    out={}
    for key,m in models.items():
        name,_=key.split("__")
        out[key]=predict_probe(m,reps[name][mask])
    return ensemble(out)


def read_models(path):
    models={}
    for p in Path(path).glob("*.npz"):
        with np.load(p) as a: models[p.stem]={k:a[k] for k in a.files}
    return models


def choose_policies(pt,pv,bars,meta,dates,cfg,v1):
    rows=[]; policies={}
    for name in ["raw_history","stdp","raw_plus_stdp"]:
        key=name+"__linear"; candidates=[]
        for q in cfg["policy_quantiles"]:
            threshold=float(np.quantile(np.abs(pt[key][:,0]),q))
            result,*_=simulate(bars,meta,pv[key][:,0],threshold,v1["trading"],dates)
            row={"model":name,"quantile":q,"threshold":threshold,**result};rows.append(row)
            if result["trades"]>=cfg["min_validation_trades"]: candidates.append(row)
        if candidates: policies[name]=max(candidates,key=lambda a:a["net_pnl"])
        else: policies[name]={"model":name,"threshold":float(np.max(np.abs(pt[key][:,0]))+1),"net_pnl":0.,"trades":0,"disabled":True}
    best=max(policies.values(),key=lambda a:a["net_pnl"])
    selected=best["model"] if best["net_pnl"]>0 else "flat"
    return {"policies":policies,"selected":selected},rows


def run(cfg,v1):
    started=time.monotonic();out=Path(cfg["output_dir"])
    raw,y,meta,bars,folds=load(cfg)
    manifest={"sources":source_hashes(),"prepared_sha256":sha256(out/"prepared.json"),"state_definition_sha256":sha256(out/"state_definition.json"),"dataset_sha256":sha256(out/"dataset.npz")}
    mf=out/"run_manifest.json"
    if mf.exists():
        assert json.loads(mf.read_text())==manifest,"Inputs or core source changed; do not resume silently"
    else: save_json(mf,manifest)
    if (out/"rolling_complete.json").exists(): return
    cache=out/"initial_frozen";initial=folds[0]
    if (cache/"cache_complete.json").exists():
        frozen_reps={p.stem:np.load(p,mmap_mode="r") for p in (cache/"representations").glob("*.npy")}
    else:
        scale=fit_scale(bars,initial["train_dates"])
        frozen_reps=representation_set(cfg,v1,raw,meta,np.arange(len(meta)),scale,initial["train_dates"],cache,True)
    initial_models=None;initial_policy=None
    for fold in folds:
        f=fold["fold"];dest=out/"folds"/f"fold_{f:02d}";dest.mkdir(parents=True,exist_ok=True)
        if (dest/"complete.json").exists():
            if f==0:
                initial_models=read_models(dest/"frozen"/"probes")
                initial_policy=json.loads((dest/"frozen"/"lock.json").read_text())["policy"]
            print(f"reuse completed fold {f}",flush=True);continue
        t0=time.monotonic()
        train=meta.trading_date.isin(fold["train_dates"]).to_numpy();val=meta.trading_date.isin(fold["validation_dates"]).to_numpy();test=meta.trading_date.isin(fold["forecast_dates"]).to_numpy()
        assert not np.any(train&val) and not np.any((train|val)&test)
        testmeta=meta[test].copy();testmeta["fold"]=f
        testmeta.to_parquet(dest/"samples.parquet",index=False)
        valbars=bars[bars.trading_date.isin(fold["validation_dates"])]
        predictions={};locks={};allpolicyrows=[]
        if f==0:
            initial_models,pt,pv=fit_probes(frozen_reps,train,val,y,cfg,dest/"frozen"/"probes")
            initial_policy,grid=choose_policies(pt,pv,valbars,meta[val],fold["validation_dates"],cfg,v1)
            firstfit=(initial_models,pt,pv,initial_policy,grid)
        for arm in cfg["arms"]:
            ad=dest/arm;ad.mkdir(exist_ok=True)
            if arm=="frozen" or f==0:
                r=frozen_reps;ym=y;tm=train;vm=val;em=test
                models=initial_models;policy=initial_policy
                modeldir=out/"folds"/"fold_00"/"frozen"/"probes"
                enc_manifest=cache/"encoder_manifest.json"
                if f==0:
                    grid=firstfit[4]
                    allpolicyrows.extend({"arm":arm,**row} for row in grid)
            elif arm=="readout_only":
                r=frozen_reps;ym=y;tm=train;vm=val;em=test
                modeldir=ad/"probes"
                models,pt,pv=fit_probes(r,tm,vm,ym,cfg,modeldir)
                policy,grid=choose_policies(pt,pv,valbars,meta[val],fold["validation_dates"],cfg,v1)
                allpolicyrows.extend({"arm":arm,**row} for row in grid)
                enc_manifest=cache/"encoder_manifest.json"
            else:
                indices=np.flatnonzero(train|val|test)
                tm,vm,em=train[indices],val[indices],test[indices];ym=y[indices]
                scale=fit_scale(bars,fold["train_dates"])
                r=representation_set(cfg,v1,raw,meta,indices,scale,fold["train_dates"],ad/"encoder")
                modeldir=ad/"probes"
                models,pt,pv=fit_probes(r,tm,vm,ym,cfg,modeldir)
                policy,grid=choose_policies(pt,pv,valbars,meta[val],fold["validation_dates"],cfg,v1)
                allpolicyrows.extend({"arm":arm,**row} for row in grid)
                enc_manifest=ad/"encoder"/"encoder_manifest.json"
                np.savez_compressed(ad/"forecast_representations.npz",**{k:a[em] for k,a in r.items() if k.startswith(("stdp_","random_","shuffle_"))})
            lock={"arm":arm,"fold":fold,"effective_fit_dates":initial if arm=="frozen" else fold,"policy":policy,"probe_hashes":{str(p):sha256(p) for p in modeldir.glob("*.npz")},"encoder_manifest":str(enc_manifest),"encoder_manifest_sha256":sha256(enc_manifest),"locked_at_utc":datetime.now(timezone.utc).isoformat(),"run_manifest_sha256":sha256(mf)}
            save_json(ad/"lock.json",lock);locks[arm]=lock
            p=predict_models(models,r,em)
            assert all(np.isfinite(a).all() for a in p.values())
            for key,value in p.items(): predictions[arm+"__"+key]=value
            fitmask=meta.trading_date.isin(initial["train_dates"]).to_numpy() if arm=="frozen" else train
            predictions[arm+"__mean__linear"]=np.broadcast_to(y[fitmask].mean(axis=0),(int(test.sum()),11)).copy()
            for name in ["raw_history","stdp","raw_plus_stdp"]:
                policyrow=policy["policies"][name]
                score=p[name+"__linear"][:,0]
                predictions[arm+"__trade_"+name]=np.where((abs(score)>=policyrow["threshold"]) & (not policyrow.get("disabled",False)),score,0.)
            selected=policy["selected"]
            predictions[arm+"__trade_selected"]=np.zeros(test.sum()) if selected=="flat" else predictions[arm+"__trade_"+selected].copy()
            print(f"fold {f:02d} {arm} locked: selected={selected}",flush=True)
        current=testmeta.current_state.to_numpy()
        past_return=(bars.close.to_numpy()[testmeta.bar_id.to_numpy()]/bars.close.to_numpy()[testmeta.bar_id.to_numpy()-5]-1)*10000
        predictions["baseline__persistence__rule"]=np.c_[past_return,abs(past_return),testmeta.current_rv,np.eye(4)[current],np.eye(4)[current]]
        if f==0:
            for key in ["raw_history__linear","stdp__linear","stdp__rff"]:
                np.testing.assert_array_equal(predictions["frozen__"+key],predictions["readout_only__"+key])
                np.testing.assert_array_equal(predictions["frozen__"+key],predictions["rolling_all__"+key])
        np.savez_compressed(dest/"predictions.npz",**predictions)
        pd.DataFrame(allpolicyrows).to_csv(dest/"validation_policy_grid.csv",index=False)
        save_json(dest/"complete.json",{"seconds":time.monotonic()-t0,"rows":int(test.sum()),"forecast_dates":fold["forecast_dates"],"predictions_sha256":sha256(dest/"predictions.npz"),"locks":{a:sha256(dest/a/"lock.json") for a in cfg["arms"]}})
        print(f"FOLD {f+1}/{len(folds)} COMPLETE ({time.monotonic()-t0:.1f}s), forecast rows={test.sum()}",flush=True)
    save_json(out/"rolling_complete.json",{"folds":len(folds),"seconds_this_invocation":time.monotonic()-started,"run_manifest_sha256":sha256(mf)})


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","synthetic","rolling"]);args=p.parse_args()
    cfg,v1=config();torch.set_num_threads(cfg["torch_threads"])
    if args.stage=="prepare":prepare(cfg)
    elif args.stage=="synthetic":
        from .synthetic import run as synth
        synth(cfg,v1)
    else:run(cfg,v1)


if __name__=="__main__":main()

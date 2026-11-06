"""Run prepare -> train/validate/lock -> evaluate; final test is never a selector."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from .data import prepare, dump_json, sha256, TARGETS
from .neural import CompetitiveSNN, encode
from .analysis import fit_readout, predict_readout, evaluate, block_bootstrap
from .backtest import simulate


def raw_designs(x):
    return {"mean": np.zeros((len(x),1)), "price_history": x[:,:,[0,1,2,5]].reshape(len(x),-1), "raw_last": x[:,-1,:], "raw_history": x.reshape(len(x),-1), "without_levels": x[:,:,[0,1,2,5,6,7,8,9,10,11]].reshape(len(x),-1), "risk_control": x[:,-1,[10,11]]}


def loaded(cfg):
    out=Path(cfg["output_dir"]); a=np.load(out/"dataset.npz")
    meta=pd.read_parquet(out/"samples.parquet"); bars=pd.read_parquet(out/"bars_features.parquet")
    selection=pd.read_csv(out/"contract_selection.csv")
    return out,a["X"],a["y"],meta,bars,selection


def calendar_for(selection, split, cfg):
    d=selection.trading_date
    mask=d.le(cfg["train_end"]) if split=="train" else (d.gt(cfg["train_end"]) & d.le(cfg["validation_end"])) if split=="validation" else d.gt(cfg["validation_end"])
    return d[mask].tolist()


def combine_predictions(preds, cfg, encoding=None):
    for enc in cfg["encodings"]:
        for mode in cfg["modes"]+["raw_plus_stdp"]:
            preds[f"{enc}_{mode}"]=np.mean([preds[f"{enc}_{mode}_s{s}"] for s in cfg["seeds"]],axis=0)
    if encoding:
        for mode in cfg["modes"]+["raw_plus_stdp"]: preds[mode]=preds[f"{encoding}_{mode}"]


def save_predictions(path, meta, y, predictions):
    table=meta.copy()
    for i,target in enumerate(TARGETS): table["actual_"+target]=y[:,i]
    extra={f"{name}__{target}": values[:,i] for name,values in predictions.items() for i,target in enumerate(TARGETS)}
    pd.concat([table.reset_index(drop=True),pd.DataFrame(extra)],axis=1).to_parquet(path,index=False)


def metric_table(preds,y,meta):
    rows=[]
    for name,p in preds.items():
        for j,target in enumerate(TARGETS): rows.append({"model":name,"target":target,**evaluate(p[:,j],y[:,j],meta.trading_date.to_numpy(),risk=j==4)})
    return pd.DataFrame(rows)


def train(cfg):
    start=time.monotonic(); out,x,y,meta,bars,selection=loaded(cfg)
    if (out/"selection_lock.json").exists(): raise RuntimeError("Selection already locked; use a new version for a new experiment")
    models=out/"models"; models.mkdir(exist_ok=True)
    reps=out/"representations"; reps.mkdir(exist_ok=True)
    tr=meta.split.eq("train").to_numpy(); va=meta.split.eq("validation").to_numpy()
    xt,xv,yt,yv=x[tr],x[va],y[tr],y[va]
    rawt,rawv=raw_designs(xt),raw_designs(xv)
    pt,pv,diagnostics={},{},{}
    def fit(name,a,b):
        readout=fit_readout(a,b,yt,yv,cfg["ridge_alphas"])
        np.savez_compressed(models/(name+"_readout.npz"),**readout)
        pt[name],pv[name]=predict_readout(readout,a),predict_readout(readout,b)
    for name in rawt: fit(name,rawt[name],rawv[name])
    for enc in cfg["encodings"]:
        for seed in cfg["seeds"]:
            for mode in cfg["modes"]:
                name=f"{enc}_{mode}_s{seed}"
                t0=time.monotonic()
                st=encode(xt,meta.sample_id.to_numpy()[tr],enc,seed,cfg["network"]["steps_per_minute"],cfg["network"]["p_max"],mode=="shuffle")
                sv=encode(xv,meta.sample_id.to_numpy()[va],enc,seed,cfg["network"]["steps_per_minute"],cfg["network"]["p_max"],mode=="shuffle")
                model=CompetitiveSNN(xt.shape[-1]*2,cfg["network"],seed)
                model.fit(st,plastic=mode!="random")
                rt,rv=model.transform(st),model.transform(sv)
                model.save(models/(name+"_snn.npz"))
                reloaded=CompetitiveSNN(24,cfg["network"],seed).load(models/(name+"_snn.npz"))
                np.testing.assert_array_equal(model.batch(sv[:128]),reloaded.batch(sv[:128]))
                np.savez_compressed(reps/(name+".npz"),train=rt,validation=rv)
                fit(name,rt,rv)
                if mode=="stdp": fit(f"{enc}_raw_plus_stdp_s{seed}",np.c_[rawt["raw_history"],rt],np.c_[rawv["raw_history"],rv])
                diagnostics[name]={"train":model.diagnostics(rt),"validation":model.diagnostics(rv),"seconds":time.monotonic()-t0,"checkpoint_reload_exact":True}
                dump_json(out/"training_diagnostics.json",diagnostics)
                print(f"trained {name}: {time.monotonic()-t0:.1f}s, active={diagnostics[name]['train']['active_neurons']}, weight_delta={diagnostics[name]['train']['weight_change_l2']:.3f}",flush=True)
    combine_predictions(pt,cfg); combine_predictions(pv,cfg)
    encoding=min(cfg["encodings"],key=lambda e: np.mean((pv[e+"_stdp"][:,2]-yv[:,2])**2))
    combine_predictions(pt,cfg,encoding); combine_predictions(pv,cfg,encoding)
    save_predictions(out/"train_predictions.parquet",meta[tr],yt,pt)
    save_predictions(out/"validation_predictions.parquet",meta[va],yv,pv)
    metric_table(pv,yv,meta[va]).to_csv(out/"validation_metrics.csv",index=False)
    policy_models=["price_history","raw_last","raw_history","without_levels","random","stdp","shuffle","raw_plus_stdp"]
    policies,grid={},[]
    vb=bars[bars.split.eq("validation")]; vm=meta[va]
    days=calendar_for(selection,"validation",cfg)
    for name in policy_models:
        candidates=[]
        for q in cfg["policy_quantiles"]:
            threshold=float(np.quantile(np.abs(pt[name][:,2]),q))
            summary,*_=simulate(vb,vm,pv[name][:,2],threshold,cfg["trading"],days)
            r={"model":name,"quantile":q,"threshold":threshold,**summary}
            grid.append(r)
            if summary["trades"]>=cfg["min_validation_trades"]: candidates.append(r)
        if candidates: policies[name]=max(candidates,key=lambda r:r["net_pnl"])
        else: policies[name]={"model":name,"quantile":None,"threshold":float(np.max(np.abs(pt[name][:,2]))+1),"net_pnl":0.,"trades":0,"disabled":True}
    eligible=[policies[n] for n in ["raw_history","stdp","raw_plus_stdp"] if not policies[n].get("disabled",False)]
    best=max(eligible,key=lambda r:r["net_pnl"]) if eligible else None
    chosen=best["model"] if best is not None and best["net_pnl"]>0 else "flat"
    pd.DataFrame(grid).to_csv(out/"validation_policy_grid.csv",index=False)
    sourcefiles=sorted(Path("independent_study/iclab").glob("*.py"))+[Path("independent_study/configs/experiment.json"),Path("independent_study/PROTOCOL.md")]
    lock={"locked_at_utc":datetime.now(timezone.utc).isoformat(),"encoding":encoding,"selected_strategy":chosen,"policies":policies,"models_sha256":{str(p):sha256(p) for p in sorted(models.glob("*.npz"))},"source_sha256":{str(p):sha256(p) for p in sourcefiles},"seconds":time.monotonic()-start,"test_predictions_seen":False,"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__}
    dump_json(out/"selection_lock.json",lock)
    print(f"LOCKED: encoding={encoding}, selected_strategy={chosen}, seconds={lock['seconds']:.1f}",flush=True)


def final_evaluation(cfg):
    out,x,y,meta,bars,selection=loaded(cfg)
    lock=json.loads((out/"selection_lock.json").read_text())
    if (out/"evaluation_complete.json").exists(): raise RuntimeError("Evaluation already completed; do not silently overwrite")
    for p,h in lock["models_sha256"].items(): assert sha256(p)==h,p
    for p,h in lock["source_sha256"].items(): assert sha256(p)==h,p
    mask=meta.split.eq("test").to_numpy(); xx,yy,mm=x[mask],y[mask],meta[mask]
    raw=raw_designs(xx); preds={}; diagnostics={}
    models=out/"models"
    def read(name,values):
        with np.load(models/(name+"_readout.npz")) as r: preds[name]=predict_readout(r,values)
    for name in raw: read(name,raw[name])
    for enc in cfg["encodings"]:
        for seed in cfg["seeds"]:
            for mode in cfg["modes"]:
                name=f"{enc}_{mode}_s{seed}"
                model=CompetitiveSNN(24,cfg["network"],seed).load(models/(name+"_snn.npz"))
                spikes=encode(xx,mm.sample_id.to_numpy(),enc,seed,cfg["network"]["steps_per_minute"],cfg["network"]["p_max"],mode=="shuffle")
                w,t=model.weight.clone(),model.theta.clone()
                rep=model.transform(spikes)
                torch.testing.assert_close(w,model.weight,rtol=0,atol=0); torch.testing.assert_close(t,model.theta,rtol=0,atol=0)
                read(name,rep)
                if mode=="stdp": read(f"{enc}_raw_plus_stdp_s{seed}",np.c_[raw["raw_history"],rep])
                np.save(out/"representations"/(name+"_test.npy"),rep)
                diagnostics[name]=model.diagnostics(rep)
                print("evaluated "+name,flush=True)
    combine_predictions(preds,cfg,lock["encoding"])
    save_predictions(out/"test_predictions.parquet",mm,yy,preds)
    metric_table(preds,yy,mm).to_csv(out/"test_metrics.csv",index=False)
    dump_json(out/"test_neural_diagnostics.json",diagnostics)
    names=["mean","price_history","raw_last","raw_history","without_levels","risk_control","random","stdp","shuffle","raw_plus_stdp"]
    days=calendar_for(selection,"test",cfg)
    boot={target:block_bootstrap({n:preds[n][:,j] for n in names},yy[:,j],mm.trading_date.to_numpy(),days,cfg["bootstrap_repetitions"],cfg["bootstrap_block_days"]) for j,target in [(2,"return_5"),(4,"absolute_5")]}
    dump_json(out/"bootstrap.json",boot)
    stability=[]
    for group,values in [("month",mm.trading_date.str[:7]),("contract",mm.contract_id),("day",mm.trading_date)]:
        for value in sorted(values.unique()):
            sub=values.eq(value).to_numpy()
            for name in names:
                for j in [2,4]: stability.append({"group":group,"value":value,"model":name,"target":TARGETS[j],**evaluate(preds[name][sub,j],yy[sub,j],mm.trading_date.to_numpy()[sub],risk=j==4)})
    pd.DataFrame(stability).to_csv(out/"stability.csv",index=False)
    tp=pd.read_parquet(out/"train_predictions.parquet")
    buckets=[]
    for name in names:
        for j in [2,4]:
            thresholds=np.quantile(tp[f"{name}__{TARGETS[j]}"],[.2,.4,.6,.8])
            ids=np.searchsorted(thresholds,preds[name][:,j],side="right")
            for b in range(5):
                sub=ids==b
                buckets.append({"model":name,"target":TARGETS[j],"bucket":b+1,"n":int(sub.sum()),"mean_prediction":float(preds[name][sub,j].mean()) if sub.any() else None,"mean_actual":float(yy[sub,j].mean()) if sub.any() else None})
    pd.DataFrame(buckets).to_csv(out/"score_buckets.csv",index=False)
    back=out/"backtests"; back.mkdir(exist_ok=True)
    btrows=[]; testbars=bars[bars.split.eq("test")]
    for name,policy in lock["policies"].items():
        scenarios={"base":{},"zero_slippage":{"slippage_ticks":0.},"two_ticks":{"slippage_ticks":2.},"high_fees":{"open_fee_rate":.000092,"close_today_fee_rate":.000299,"broker_extra_rate":0.},"extra_minute":{}}
        for scenario,changes in scenarios.items():
            c={**cfg["trading"],**changes}
            summary,tr,eq,daily=simulate(testbars,mm,preds[name][:,2],policy["threshold"],c,days,extra_delay=int(scenario=="extra_minute"),enabled=not policy.get("disabled",False))
            btrows.append({"model":name,"scenario":scenario,**summary})
            if scenario=="base":
                tr.to_csv(back/(name+"_trades.csv"),index=False)
                eq.to_parquet(back/(name+"_equity.parquet"),index=False)
                daily.to_csv(back/(name+"_daily.csv"),index=False)
    summary,tr,eq,daily=simulate(testbars,mm,preds["stdp"][:,2],0,cfg["trading"],days,enabled=False)
    btrows.append({"model":"flat","scenario":"base",**summary})
    daily.to_csv(back/"flat_daily.csv",index=False)
    bt=pd.DataFrame(btrows); bt.to_csv(out/"backtest_summary.csv",index=False)
    selected=bt[(bt.model==lock["selected_strategy"]) & (bt.scenario=="base")].iloc[0].to_dict()
    dump_json(out/"evaluation_complete.json",{"completed_at_utc":datetime.now(timezone.utc).isoformat(),"selected_strategy":selected,"model_hashes_unchanged":all(sha256(p)==h for p,h in lock["models_sha256"].items()),"selection_lock_sha256":sha256(out/"selection_lock.json")})
    print("FINAL EVALUATION COMPLETE: "+json.dumps(selected,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["prepare","train","evaluate"]); p.add_argument("--config",default="independent_study/configs/experiment.json")
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text()); torch.set_num_threads(2)
    if args.stage=="prepare": prepare(cfg,Path(cfg["output_dir"]))
    elif args.stage=="train": train(cfg)
    else: final_evaluation(cfg)


if __name__=="__main__": main()

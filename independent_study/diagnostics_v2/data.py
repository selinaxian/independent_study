from pathlib import Path
import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES, sha256
from .common import save_json


def make_folds(calendar_dates, available_days, cfg):
    forecast=[d for d in calendar_dates if cfg["first_forecast"]<=d<=cfg["last_forecast"]]
    folds=[]
    for start in range(0,len(forecast),cfg["forecast_package_days"]):
        test=forecast[start:start+cfg["forecast_package_days"]]
        previous=[d for d in available_days if d<test[0]]
        nt,nv=cfg["train_days"],cfg["validation_days"]
        if len(previous)<nt+nv: raise ValueError("Insufficient prior dates")
        train=previous[-nt-nv:-nv]; val=previous[-nv:]
        assert max(train)<min(val) and max(val)<min(test)
        folds.append({"fold":len(folds),"train_dates":train,"validation_dates":val,"forecast_dates":test})
    return folds


def path_stats(path):
    diff=np.diff(path,axis=1)
    rv=np.sqrt(np.sum(np.diff(np.log(path),axis=1)**2,axis=1))*10000
    efficiency=np.abs(path[:,-1]-path[:,0])/np.maximum(np.abs(diff).sum(axis=1),1e-8)
    return rv,efficiency


def states(rv,eff,thresholds):
    return (2*(rv>thresholds[0])+(eff>thresholds[1])).astype(np.int64)


def prepare(cfg):
    root=Path(cfg["v1_dir"]); out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    if (out/"prepared.json").exists(): raise RuntimeError("Preparation exists; use run stage to reuse it")
    bars=pd.read_parquet(root/"bars_features.parquet").sort_values("bar_id").reset_index(drop=True)
    meta=pd.read_parquet(root/"samples.parquet").sort_values("sample_id").reset_index(drop=True)
    selection=pd.read_csv(root/"contract_selection.csv")
    assert np.array_equal(bars.bar_id,np.arange(len(bars)))
    assert np.array_equal(meta.sample_id,np.arange(len(meta)))
    ix=meta.bar_id.to_numpy()[:,None]+np.arange(-11,1)[None,:]
    raw=bars[FEATURES].to_numpy()[ix]
    assert np.isfinite(raw).all()
    nowpath=bars.close.to_numpy()[meta.bar_id.to_numpy()[:,None]+np.arange(-5,1)]
    futpath=bars.open.to_numpy()[meta.entry_bar_id.to_numpy()[:,None]+np.arange(6)]
    nowrv,noweff=path_stats(nowpath); futurerv,futureeff=path_stats(futpath)
    returns=(futpath[:,-1]/futpath[:,0]-1)*10000
    days=sorted(meta.trading_date.unique()); calendar=selection.trading_date.tolist()
    folds=make_folds(calendar,days,cfg)
    first_train=meta.trading_date.isin(folds[0]["train_dates"]).to_numpy()
    thresholds=np.array([np.median(nowrv[first_train]),np.quantile(noweff[first_train],2/3)])
    current=states(nowrv,noweff,thresholds); future=states(futurerv,futureeff,thresholds)
    y=np.c_[returns,abs(returns),futurerv,np.eye(4)[current],np.eye(4)[future]]
    meta["current_state"]=current; meta["future_state"]=future
    meta["current_rv"]=nowrv; meta["current_efficiency"]=noweff
    meta["future_rv"]=futurerv; meta["future_efficiency"]=futureeff
    assert np.isfinite(y).all()
    np.savez_compressed(out/"dataset.npz",raw=raw,y=y)
    meta.to_parquet(out/"samples.parquet",index=False)
    save_json(out/"folds.json",folds)
    save_json(out/"state_definition.json",{"rv_threshold_bps":thresholds[0],"efficiency_threshold":thresholds[1],"fit_dates":folds[0]["train_dates"],"names":["低波震荡","低波趋势","高波震荡","高波趋势"],"future_label_ends":"entry_bar_id+5 open"})
    inputs=[root/"bars_features.parquet",root/"samples.parquet",root/"contract_selection.csv",root/"artifact_manifest.json"]
    save_json(out/"prepared.json",{"samples":len(meta),"raw_shape":raw.shape,"folds":len(folds),"forecast_days":sum(len(f["forecast_dates"]) for f in folds),"forecast_samples":int(meta.trading_date.ge(cfg["first_forecast"]).sum()),"inputs":{str(p):sha256(p) for p in inputs}})
    return raw,y,meta,bars,folds


def load(cfg):
    out=Path(cfg["output_dir"]); a=np.load(out/"dataset.npz")
    bars=pd.read_parquet(Path(cfg["v1_dir"])/"bars_features.parquet")
    return a["raw"],a["y"],pd.read_parquet(out/"samples.parquet"),bars,json.loads((out/"folds.json").read_text())


def fit_scale(bars,train_dates):
    a=bars.loc[bars.trading_date.isin(train_dates),FEATURES].dropna().to_numpy()
    return np.maximum(np.quantile(abs(a),.99,axis=0),1e-8)


def apply_scale(raw,scale):
    return np.clip(raw/scale,-1,1).astype(np.float32)

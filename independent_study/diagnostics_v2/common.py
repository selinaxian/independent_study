from pathlib import Path
import json
import numpy as np
import pandas as pd
from iclab.data import sha256


def save_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, Path): return str(x)
        return x
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)


def config():
    c=json.loads(Path("independent_study/diagnostics_v2/config.json").read_text())
    v1=json.loads(Path(c["v1_config"]).read_text())
    return c,v1


def source_hashes():
    files=list(Path("independent_study/diagnostics_v2").glob("*.py"))
    files+=list(Path("independent_study/iclab").glob("*.py"))
    files+=[Path("independent_study/diagnostics_v2/config.json"),Path("independent_study/diagnostics_v2/PROTOCOL.md")]
    return {str(p):sha256(p) for p in sorted(files)}


def normalized_class_scores(scores):
    p=np.maximum(scores,1e-6)
    return p/p.sum(axis=1,keepdims=True)


def classification(y, scores, classes=4):
    y=np.asarray(y,dtype=int); predicted=np.asarray(scores).argmax(axis=1)
    cm=np.bincount(y*classes+predicted,minlength=classes*classes).reshape(classes,classes)
    support=cm.sum(axis=1); guess=cm.sum(axis=0); tp=np.diag(cm)
    recall=np.divide(tp,support,out=np.full(classes,np.nan),where=support>0)
    f1=np.divide(2*tp,support+guess,out=np.zeros(classes,dtype=float),where=(support+guess)>0)
    p=normalized_class_scores(np.asarray(scores))
    return {"accuracy":float((y==predicted).mean()),"balanced_accuracy":float(np.nanmean(recall)),"macro_f1":float(f1.mean()),"brier":float(np.mean(np.sum((p-np.eye(classes)[y])**2,axis=1))),"n":len(y),"confusion":cm,"support":support}


def probed_representations(x, spikes, neural):
    n,length,_=x.shape
    d={"raw_history":x.reshape(n,-1),"risk_control":x[:,-1,[10,11]],
       "counts_minute":spikes.reshape(n,length,-1,24).sum(axis=2).reshape(n,-1).astype(float),
       "counts_total":spikes.sum(axis=1).astype(float)}
    d.update(neural)
    if "stdp" in neural: d["raw_plus_stdp"]=np.c_[d["raw_history"],neural["stdp"]]
    return d

"""Comparable linear and RBF random-feature probes, trained only on past data."""
import numpy as np


def fit_probe(x, val, y, yval, kind, alphas, seed=7331, components=256, groups=None, nonnegative=(1,2)):
    x,val=np.asarray(x,dtype=np.float64),np.asarray(val,dtype=np.float64)
    model={"input_mean":x.mean(axis=0),"input_scale":np.maximum(x.std(axis=0),1e-6),"kind":np.array(kind),"nonnegative":np.asarray(nonnegative,dtype=int)}
    a=(x-model["input_mean"])/model["input_scale"]; b=(val-model["input_mean"])/model["input_scale"]
    if kind=="rff":
        rng=np.random.default_rng(seed)
        i=rng.integers(len(a),size=512); j=rng.integers(len(a),size=512)
        dist=np.linalg.norm(a[i]-a[j],axis=1)
        bw=max(float(np.median(dist[dist>1e-9])) if np.any(dist>1e-9) else 1.,1e-6)
        model.update(omega=rng.normal(size=(x.shape[1],components))/bw,phase=rng.uniform(0,2*np.pi,components),bandwidth=np.array(bw))
        a=np.sqrt(2/components)*np.cos(a@model["omega"]+model["phase"])
        b=np.sqrt(2/components)*np.cos(b@model["omega"]+model["phase"])
    elif kind!="linear": raise ValueError(kind)
    model.update(basis_mean=a.mean(axis=0),basis_scale=np.maximum(a.std(axis=0),1e-6),target_mean=y.mean(axis=0))
    a=(a-model["basis_mean"])/model["basis_scale"]
    b=(b-model["basis_mean"])/model["basis_scale"]
    eig,vec=np.linalg.eigh(a.T@a); rhs=vec.T@(a.T@(y-model["target_mean"]))
    coefs=np.array([vec@(rhs/(np.maximum(eig,0)[:,None]+alpha)) for alpha in alphas])
    candidates=np.einsum("nd,adj->anj",b,coefs)+model["target_mean"]
    if len(nonnegative): candidates[:,:,list(nonnegative)]=np.maximum(candidates[:,:,list(nonnegative)],0)
    mse=np.mean((candidates-yval)**2,axis=1)
    if groups is None: groups=[[0],[1],[2],list(range(3,7)),list(range(7,11))]
    best=np.zeros(y.shape[1],dtype=int)
    for group in groups: best[group]=np.argmin(mse[:,group].mean(axis=1))
    model["coef"]=np.column_stack([coefs[best[j],:,j] for j in range(y.shape[1])])
    model["alphas"]=np.asarray(alphas)[best]; model["validation_mse_grid"]=mse
    return model


def predict_probe(model,x):
    a=(np.asarray(x,dtype=np.float64)-model["input_mean"])/model["input_scale"]
    if str(model["kind"])=="rff": a=np.sqrt(2/len(model["phase"]))*np.cos(a@model["omega"]+model["phase"])
    a=(a-model["basis_mean"])/model["basis_scale"]
    p=a@model["coef"]+model["target_mean"]
    nn=model["nonnegative"].astype(int)
    if len(nn): p[:,nn]=np.maximum(p[:,nn],0)
    return p

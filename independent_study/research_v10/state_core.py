"""Probability-coordinate post-triggered STDP and positive variance readout.

Research variant inspired by probabilistic WTA learning, not an exact
continuous-time reproduction of Nessler et al. or an HMM implementation.
"""
import numpy as np
from event_stdp_v7.core import lagged


def fit_bins(features,train,bins=5):
    return np.quantile(features[train],np.arange(1,bins)/bins,axis=0)


def categories(features,bounds,segments,anchors,delays):
    anchors=np.asarray(anchors,int);delays=np.asarray(delays,int)
    if (delays<0).any():raise ValueError('Future delay')
    idx=anchors[:,None]-delays[None,:]
    if (idx<0).any() or not (segments[idx]==segments[anchors,None]).all():
        raise ValueError('Delay crosses a segment')
    raw=features[idx].transpose(0,2,1).reshape(len(anchors),-1)
    if not np.isfinite(raw).all():raise ValueError('Missing history')
    # Group ordering: feature, delay; categorical amplitude, not crossing count.
    expanded=np.repeat(bounds,len(delays),axis=1)
    return (raw[:,None,:]>expanded[None,:,:]).sum(axis=1).astype(np.int64),raw


def one_hot(cat,bins):
    return np.eye(bins)[cat].reshape(len(cat),-1)


def shuffle_delays(cat,features,delays,seed):
    rng=np.random.default_rng(seed);x=cat.reshape(len(cat),features,delays)
    order=np.argsort(rng.random((len(cat),delays)),axis=1)
    return np.take_along_axis(x,order[:,None,:],axis=2).reshape(cat.shape)


class ProbabilitySTDP:
    def __init__(self,groups,bins,hidden,eta=.02,epochs=3,seed=104,floor=1e-5):
        self.groups,self.bins,self.hidden=groups,bins,hidden
        self.eta,self.epochs,self.floor=eta,epochs,floor
        r=np.random.default_rng(seed).gamma(5.,1.,(hidden,groups,bins))
        self.q=r/r.sum(axis=2,keepdims=True);self.initial=self.q.copy()
        self.counts=np.zeros(hidden,int)

    def posterior(self,cat):
        logq=np.log(self.q);score=np.zeros((len(cat),self.hidden))
        for group in range(self.groups):score+=logq[:,group,cat[:,group]].T
        maximum=score.max(axis=1,keepdims=True);z=np.exp(score-maximum)
        posterior=z/z.sum(axis=1,keepdims=True)
        surprise=-(maximum[:,0]+np.log(z.sum(axis=1))-np.log(self.hidden))
        return posterior,surprise

    def local_update(self,pre,winner):
        # The winner's post follows local delayed-input arrival. Recent pre
        # potentiates; absent-pre categories depress. No financial label enters.
        self.q[winner]*=1-self.eta
        self.q[winner,np.arange(self.groups),pre]+=self.eta
        self.q[winner]=np.maximum(self.q[winner],self.floor)
        self.q[winner]/=self.q[winner].sum(axis=1,keepdims=True)
        self.counts[winner]+=1

    def fit(self,cat,seed):
        rng=np.random.default_rng(seed);group=np.arange(self.groups)
        for _ in range(self.epochs):
            for pre in cat:  # chronological windows; reset local arrivals
                score=np.log(self.q[:,group,pre]).sum(axis=1)
                p=np.exp(score-score.max());p/=p.sum()
                winner=min(int(np.searchsorted(np.cumsum(p),rng.random())),self.hidden-1)
                self.local_update(pre,winner)
        return self


def qlike(y,pred):
    if (np.asarray(pred)<=0).any():raise ValueError('Variance must be positive')
    return np.log(pred)+np.asarray(y)/pred


def predict_variance(model,x):
    z=np.clip((x-model['mean'])/model['scale'],-8,8)
    linear=z@model['coef']+model['intercept']
    return np.exp(np.clip(linear,-12,20))*model['smearing']


def fit_variance(x,y,train,val,alphas=(1,10,100,1000),floor=.01):
    """Same log-variance Ridge and train-only smearing for every representation."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    mean=x[train].mean(axis=0);scale=np.maximum(x[train].std(axis=0),1e-8)
    z=np.clip((x-mean)/scale,-8,8)
    target=np.log(y[train]+floor);intercept=float(target.mean())
    xx=z[train].T@z[train];xy=z[train].T@(target-intercept)
    eig,vectors=np.linalg.eigh(xx);projection=vectors.T@xy
    candidates=[]
    for alpha in alphas:
        coef=vectors@(projection/np.maximum(eig+alpha,1e-10))
        raw=np.exp(np.clip(z@coef+intercept,-12,20))
        smear=float(np.mean(y[train]/raw[train]))
        pred=raw*smear;loss=float(qlike(y[val],pred[val]).mean())
        model=dict(mean=mean,scale=scale,coef=coef,intercept=intercept,smearing=smear,
                   alpha=float(alpha),validation_qlike=loss)
        candidates.append((loss,float(alpha),model,pred))
    _,_,model,pred=min(candidates,key=lambda a:(a[0],a[1]))
    return pred,model


def risk_features(bars,anchors):
    seg=bars.segment_id.to_numpy();close=bars.close.to_numpy();r=np.full(len(close),np.nan)
    good=seg[1:]==seg[:-1];r[1:]=np.where(good,np.diff(np.log(close))*10000,np.nan)
    columns=[]
    for window in [1,5,20]:
        values=np.stack([lagged(r,seg,d) for d in range(window)],axis=1)
        # lagged zeros at boundaries are excluded by finite/full-segment check.
        assert (anchors-window>=0).all() and (seg[anchors-window]==seg[anchors]).all()
        v=(values[anchors]**2).mean(axis=1)*5
        columns.append(np.log(v+.01))
    clock=bars.session_progress.to_numpy()[anchors]
    afternoon=(bars.datetime.dt.hour.to_numpy()[anchors]>=12).astype(float)
    base=np.column_stack(columns+[clock,clock**2,afternoon])
    assert np.isfinite(base).all()
    return base,np.exp(base[:,1])-.01

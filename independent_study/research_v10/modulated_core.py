import numpy as np
from research_v10.state_core import qlike


def softmax(x,axis=-1):
    e=np.exp(x-x.max(axis=axis,keepdims=True));return e/e.sum(axis=axis,keepdims=True)


def forward(theta,a,b,pre,base):
    q=softmax(theta,axis=2)
    p=softmax(pre@np.log(q).reshape(len(q),-1).T,axis=1)
    correction=b+p@a
    return base*np.exp(np.clip(correction,-8,8)),p,q


def gradients(theta,a,b,pre,base,y,hidden_target=None):
    """Expected local pre/post eligibility times neuron-specific loss signal."""
    h,p,q=forward(theta,a,b,pre,base)
    active=(np.abs(b+p@a)<8).astype(float)
    error=(1-y/h)*active
    hidden_error=error if hidden_target is None else (1-hidden_target/h)*active
    ga=p.T@error/len(pre);gb=float(error.mean())
    signal=hidden_error[:,None]*(a[None,:]-(p@a)[:,None])
    weighted=signal*p
    shape=q.shape
    gtheta=(weighted.T@pre).reshape(shape)/len(pre)-weighted.mean(axis=0)[:,None,None]*q
    return gtheta,ga,gb


def clip_norm(g,limit=5.):
    return g*min(1.,limit/max(float(np.linalg.norm(g)),1e-15))


def fit_modulated(pre,base,y,train,val,initial_q,mode,seed,wrong=None):
    theta=np.log(initial_q).copy();anchor=theta.copy();a=np.zeros(len(theta));b=np.array(0.)
    parameters=[theta,a,b];m=[np.zeros_like(v) for v in parameters];v=[np.zeros_like(a) for a in parameters]
    rng=np.random.default_rng(seed);indices=np.flatnonzero(train);best=None;history=[];step=0
    for epoch in range(1,11):
        shuffled=rng.permutation(indices)
        for start in range(0,len(shuffled),128):
            ix=shuffled[start:start+128];target=None if mode!='mismatched' else wrong[ix]
            gt,ga,gb=gradients(theta,a,b,pre[ix],base[ix],y[ix],target)
            grads=[gt+.001*(theta-anchor),ga+.01*a,np.array(gb)]
            step+=1
            for j,(parameter,gradient) in enumerate(zip(parameters,grads)):
                if j==0 and mode=='frozen':continue
                gradient=clip_norm(gradient)
                m[j]=.9*m[j]+.1*gradient;v[j]=.999*v[j]+.001*gradient**2
                parameter-=.005*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
        prediction,_,_=forward(theta,a,b,pre[val],base[val])
        loss=float(qlike(y[val],prediction).mean());history.append(loss)
        if best is None or loss<best[0]:best=(loss,epoch,theta.copy(),a.copy(),b.copy())
    loss,epoch,theta,a,b=best
    pred,p,q=forward(theta,a,b,pre,base)
    return pred,dict(theta=theta,a=a,b=b,selected_epoch=epoch,validation_qlike=loss,
                      validation_history=np.array(history),q_change=float(abs(q-initial_q).mean()))

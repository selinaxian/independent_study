import numpy as np
from research_v10.modulated_core import softmax,clip_norm
from research_v10.state_core import qlike


def posterior(q,pre,temperature):
    if temperature<=0:raise ValueError('Positive temperature required')
    return softmax((pre@np.log(q).reshape(len(q),-1).T)/temperature,axis=1)


def predict(model,p,base):
    return base*np.exp(np.clip(model['b']+p@model['a'],-8,8))


def fit_output(p,base,y,train,val,seed):
    a=np.zeros(p.shape[1]);b=np.array(0.);params=[a,b]
    first=[np.zeros_like(z) for z in params];second=[np.zeros_like(z) for z in params]
    rng=np.random.default_rng(seed);indices=np.flatnonzero(train);step=0;best=None;history=[]
    for epoch in range(1,11):
        order=rng.permutation(indices)
        for start in range(0,len(order),128):
            ix=order[start:start+128];pa=p[ix];correction=b+pa@a
            h=base[ix]*np.exp(np.clip(correction,-8,8))
            error=(1-y[ix]/h)*(abs(correction)<8)
            gradients=[pa.T@error/len(ix)+.01*a,np.array(error.mean())];step+=1
            for j,(parameter,g) in enumerate(zip(params,gradients)):
                g=clip_norm(g);first[j]=.9*first[j]+.1*g;second[j]=.999*second[j]+.001*g*g
                parameter-=.005*(first[j]/(1-.9**step))/(np.sqrt(second[j]/(1-.999**step))+1e-8)
        loss=float(qlike(y[val],predict(dict(a=a,b=b),p[val],base[val])).mean());history.append(loss)
        if best is None or loss<best[0]:best=(loss,epoch,a.copy(),b.copy())
    loss,epoch,a,b=best;model=dict(a=a,b=b,selected_epoch=epoch,validation_qlike=loss,history=np.array(history))
    return predict(model,p,base),model

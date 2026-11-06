import numpy as np
from research_v10.modulated_core import clip_norm
from research_v10.temperature_core import predict
from research_v10.state_core import fit_variance


def fixed_variance(x,y,past,alpha):
    pred,model=fit_variance(x,y,past,past,[alpha])
    model['training_qlike']=model.pop('validation_qlike')
    return pred,model


def fixed_output(p,base,y,past,epochs,seed):
    a=np.zeros(p.shape[1]);b=np.array(0.);params=[a,b]
    first=[np.zeros_like(z) for z in params];second=[np.zeros_like(z) for z in params]
    rng=np.random.default_rng(seed);indices=np.flatnonzero(past);step=0
    for _ in range(int(epochs)):
        order=rng.permutation(indices)
        for start in range(0,len(order),128):
            ix=order[start:start+128];pa=p[ix];correction=b+pa@a
            h=base[ix]*np.exp(np.clip(correction,-8,8));error=(1-y[ix]/h)*(abs(correction)<8)
            gradients=[pa.T@error/len(ix)+.01*a,np.array(error.mean())];step+=1
            for j,(parameter,g) in enumerate(zip(params,gradients)):
                g=clip_norm(g);first[j]=.9*first[j]+.1*g;second[j]=.999*second[j]+.001*g*g
                parameter-=.005*(first[j]/(1-.9**step))/(np.sqrt(second[j]/(1-.999**step))+1e-8)
    model=dict(a=a,b=b,epochs=int(epochs));return predict(model,p,base),model

import numpy as np
from research_v10.modulated_core import fit_modulated,softmax
from research_v10.temperature_core import posterior,fit_output


def test_temperature_one_reproduces_frozen_three_factor_readout():
    rng=np.random.default_rng(8);cat=rng.integers(0,3,(200,2));x=np.eye(3)[cat].reshape(200,-1)
    y=np.exp(cat[:,0]*.2);base=np.ones(200);q=softmax(rng.normal(size=(3,2,3)),axis=2)
    train=np.arange(200)<120;val=(np.arange(200)>=120)&(np.arange(200)<160)
    q=softmax(np.log(q),axis=2)
    p=posterior(q,x,1)
    first,m=fit_output(p,base,y,train,val,9);second,m2=fit_modulated(x,base,y,train,val,q,'frozen',9)
    np.testing.assert_allclose(first,second,rtol=1e-12,atol=1e-12)
    assert m['selected_epoch']==m2['selected_epoch']
    softer=posterior(q,x,6)
    np.testing.assert_allclose(softer.sum(axis=1),1)
    assert (-softer*np.log(softer)).sum(axis=1).mean()>(-p*np.log(p)).sum(axis=1).mean()

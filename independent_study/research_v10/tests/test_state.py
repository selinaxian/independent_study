import numpy as np
import pytest
from research_v10.state_core import (ProbabilitySTDP,categories,fit_bins,shuffle_delays,
                                     fit_variance,predict_variance,qlike)


def test_local_stdp_is_normalized_selective_and_signed():
    m=ProbabilitySTDP(3,5,4,seed=10);before=m.q.copy();pre=np.array([0,1,2])
    m.local_update(pre,2)
    np.testing.assert_allclose(m.q.sum(axis=2),1)
    assert (m.q>0).all();np.testing.assert_array_equal(m.q[[0,1,3]],before[[0,1,3]])
    assert (m.q[2,np.arange(3),pre]>before[2,np.arange(3),pre]).all()
    assert m.q[2,0,1]<before[2,0,1]


def test_population_encoding_preserves_order_and_future_invariance():
    x=np.array([[0,0],[1,0],[0,1],[0,0],[1,0],[0,1]],float);seg=np.zeros(6)
    b=np.array([[.5,.5]]);a,raw=categories(x,b,seg,np.array([2,5]),[0,2])
    changed=x.copy();changed[3:]=3
    newer,_=categories(changed,b,seg,np.array([2]),[0,2])
    np.testing.assert_array_equal(a[:1],newer)
    with pytest.raises(ValueError):categories(x,b,np.repeat([0,1],3),np.array([3]),[0,2])
    reverse,_=categories(x,b,seg,np.array([2]),[2,0]);assert not np.array_equal(a[:1],reverse)


def test_temporal_shuffle_retains_per_feature_multiset():
    x=np.arange(4*3*3).reshape(4,9)%5;y=shuffle_delays(x,3,3,4)
    np.testing.assert_array_equal(np.sort(x.reshape(4,3,3),axis=2),np.sort(y.reshape(4,3,3),axis=2))


def test_simple_states_and_equal_count_sequences_are_learnable():
    # Different categorical temporal patterns with exactly two input spikes.
    a=np.tile([[0,1],[1,0]],(300,1));m=ProbabilitySTDP(2,2,2,eta=.04,epochs=3,seed=104).fit(a,105)
    p,_=m.posterior(np.array([[0,1],[1,0]]))
    assert p[0].argmax()!=p[1].argmax() and p.max(axis=1).min()>.9
    again=ProbabilitySTDP(2,2,2,eta=.04,epochs=3,seed=104).fit(a,105)
    np.testing.assert_array_equal(m.q,again.q)


def test_variance_readout_no_forecast_target_leak_and_reload():
    rng=np.random.default_rng(22);x=rng.normal(size=(180,3));y=np.exp(1+.5*x[:,0])
    train=np.arange(180)<100;val=(np.arange(180)>=100)&(np.arange(180)<140)
    p,m=fit_variance(x,y,train,val)
    y2=y.copy();y2[140:]*=100
    p2,m2=fit_variance(x,y2,train,val)
    np.testing.assert_array_equal(p,p2)
    np.testing.assert_allclose(p,predict_variance(m,x),rtol=1e-13)
    assert (p>0).all() and qlike(y[140:],p[140:]).mean()<qlike(y[140:],np.full(40,y[train].mean())).mean()

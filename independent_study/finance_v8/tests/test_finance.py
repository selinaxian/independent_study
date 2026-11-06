import numpy as np
import pandas as pd
import pytest
from finance_v8.core import (delayed_history, hidden_threshold, fit_autonomous,
                             mid_percentile, quantities, shifted_opportunities)


def test_delay_windows_reset_and_are_causal():
    s=np.eye(12,2);f=np.ones((12,2));seg=np.repeat([1,2],6)
    x,valid=delayed_history(s,f,seg,range(5))
    assert valid.tolist()==[False]*5+[True]+[False]*5+[True]
    assert not x[6:].any()
    s[8:]=1
    y,_=delayed_history(s,f,seg,range(5))
    np.testing.assert_array_equal(x[:8],y[:8])


def test_hidden_threshold_is_above_single_weight_and_train_only():
    w=np.array([[.3,.7],[.6,.2]])
    train=np.array([[0,0],[1,0],[1,1]],dtype=float)
    t=hidden_threshold(w,train)
    assert (t>w.max(axis=1)).all()
    assert not (np.eye(2)@w.T>t).any()
    np.testing.assert_array_equal(t,hidden_threshold(w,train.copy()))


def test_autonomous_learning_future_invariance_and_boundary():
    cfg=dict(initial_weight=.5,horizon=5,tau_minutes=3.,epochs=2,
             eta_plus=.01,eta_minus=.01,homeostasis_rate=.01,target_firing=.2)
    rng=np.random.default_rng(23);h=(rng.random((60,4))<.3).astype(float)
    seg=np.repeat(np.arange(3),20);train=np.arange(60)<40
    w,stats=fit_autonomous(h,seg,train,cfg)
    h[40:]=1-h[40:]
    w2,stats2=fit_autonomous(h,seg,train,cfg)
    np.testing.assert_array_equal(w,w2);assert stats==stats2
    assert ((w>=0)&(w<=1)).all()


def test_ordinal_rules_constants_and_complement():
    s=np.array([0.,.6,.7,.8,.9,np.nan]);covered=np.array([1,1,1,1,1,0],bool)
    lo=quantities(s,covered,.6,.8,'filter');hi=quantities(s,covered,.6,.8,'timing')
    np.testing.assert_array_equal(lo+hi,np.ones(6))
    np.testing.assert_array_equal(quantities(s,covered,.6,.8,'sizing'),[2,2,1,1,0,2])
    np.testing.assert_array_equal(quantities(np.zeros(4),np.ones(4,bool),0,0,'filter'),np.ones(4))
    np.testing.assert_array_equal(mid_percentile(np.zeros(4),np.zeros(10)),np.full(4,.5))


def test_delayed_execution_same_hold_and_boundary_rejection():
    times=pd.date_range('2026-04-01 10:00',periods=12,freq='min',tz='Asia/Shanghai')
    bars=pd.DataFrame(dict(bar_id=np.arange(12),datetime=times,segment_id=[1]*10+[2]*2,
                           open=100+np.arange(12),volume=np.ones(12)))
    opportunities=pd.DataFrame(dict(signal_bar_id=[0],signal_time=[times[0]],entry_bar_id=[2],
                                     exit_bar_id=[7],exit_bar_id_actual=[7]))
    result=shifted_opportunities(opportunities,bars,1)
    assert result.entry_bar_id.iloc[0]==3 and result.exit_bar_id_actual.iloc[0]==8
    assert result.entry_open.iloc[0]==103 and result.exit_open.iloc[0]==108
    with pytest.raises(ValueError,match='segment boundary'):
        shifted_opportunities(opportunities,bars,3)

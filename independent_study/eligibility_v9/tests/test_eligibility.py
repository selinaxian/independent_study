import json
from pathlib import Path
import numpy as np
from sequence_v8.core import CompetitivePatterns
from eligibility_v9.core import (EventModulatedPatterns, matured_feedback,
                                  mismatch_feedback, eligibility, auxiliary_delta)

CFG=json.loads((Path(__file__).parents[1]/'config.json').read_text())


def test_original_and_zero_auxiliary_exact_equivalence():
    x=(np.random.default_rng(18).random((80,120))<.12).astype(float)
    original=CompetitivePatterns(120,CFG,104).fit(x,105)
    no=EventModulatedPatterns(120,CFG,104).fit(x,105)
    zero=EventModulatedPatterns(120,{**CFG,'feedback_eta':0.},104).fit(x,105,np.arange(80)%2,.5)
    np.testing.assert_array_equal(original.weight,no.weight)
    np.testing.assert_array_equal(original.weight,zero.weight)
    np.testing.assert_array_equal(original.usage,no.usage)


def test_local_credit_sign_decay_and_inactive_preservation():
    w=np.array([.2,.3,.4]);pre=np.array([1.,0.,1.])
    e=eligibility(pre,5,5)
    np.testing.assert_allclose(e,pre*np.exp(-1))
    up=auxiliary_delta(w,e,.9,.15);down=auxiliary_delta(w,e,-.1,.15)
    assert (up[pre>0]>0).all() and (down[pre>0]<0).all()
    assert up[1]==down[1]==0
    np.testing.assert_allclose(eligibility(pre,10,5),e*np.exp(-1))


def test_feedback_cannot_cross_training_holes_segments_or_future():
    n=40;seg=np.repeat([0,1],20);event=np.zeros(n,bool);event[[12,18,24,31]]=True
    train=np.arange(n)<27;train[8]=False;eligible=np.ones(n,bool)
    mask,y,p=matured_feedback(event,seg,train,eligible,5)
    assert not mask[3:9].any() and not mask[15:20].any() and not mask[22:].any()
    altered=event.copy();altered[~train]=~altered[~train]
    m2,y2,p2=matured_feedback(altered,seg,train,eligible,5)
    np.testing.assert_array_equal(mask,m2);np.testing.assert_array_equal(y,y2);assert p==p2
    # Event at t itself never supplies positive future feedback.
    e=np.zeros(20,bool);e[10]=True
    m,z,_=matured_feedback(e,np.zeros(20),np.ones(20,bool),np.ones(20,bool),5)
    assert z[10]==0 and z[9]==1 and z[5]==1 and z[4]==0


def test_shuffle_preserves_scope_and_segment_counts():
    seg=np.repeat(np.arange(3),30);mask=np.arange(90)%5!=0
    y=np.where(mask,(np.arange(90)%7==0).astype(float),np.nan)
    fake=mismatch_feedback(y,seg,mask,11)
    assert np.isnan(fake[~mask]).all()
    for sid in np.unique(seg):
        np.testing.assert_array_equal(np.sort(fake[mask&(seg==sid)]),np.sort(y[mask&(seg==sid)]))
    assert np.any(fake[mask]!=y[mask])


def test_outcome_changes_hidden_only_when_eligible():
    cfg={**CFG,'hidden':2,'epochs':1}
    x=np.array([[1.,1.,0.,0.]])
    no=EventModulatedPatterns(4,cfg,104).fit(x,1)
    pos=EventModulatedPatterns(4,cfg,104).fit(x,1,np.array([1.]),.2)
    neg=EventModulatedPatterns(4,cfg,104).fit(x,1,np.array([0.]),.2)
    missing=EventModulatedPatterns(4,cfg,104).fit(x,1,np.array([np.nan]),.2)
    winner=np.argmax(no.usage)
    assert pos.weight[winner,:2].sum()>no.weight[winner,:2].sum()>neg.weight[winner,:2].sum()
    np.testing.assert_array_equal(missing.weight,no.weight)
    np.testing.assert_array_equal(pos.weight[1-winner],pos.initial_weight[1-winner])
    assert pos.audit['positive_updates']==1 and neg.audit['negative_updates']==1

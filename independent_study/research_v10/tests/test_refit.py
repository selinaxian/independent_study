import numpy as np
from research_v10.refit_core import fixed_output,fixed_variance
from research_v10.temperature_core import fit_output


def test_refit_fixed_epoch_matches_selected_checkpoint_and_ignores_future():
    rng=np.random.default_rng(14);x=rng.normal(size=(200,4));p=np.exp(x);p/=p.sum(axis=1,keepdims=True)
    y=np.exp(x[:,0]*.2);base=np.ones(200)
    tr=np.arange(200)<100;val=(np.arange(200)>=100)&(np.arange(200)<150)
    first,m=fit_output(p,base,y,tr,val,11)
    second,n=fixed_output(p,base,y,tr,int(m['selected_epoch']),11)
    np.testing.assert_array_equal(first,second)
    past=np.arange(200)<150;changed=y.copy();changed[150:]*=100
    a,_=fixed_output(p,base,y,past,4,11);b,_=fixed_output(p,base,changed,past,4,11)
    np.testing.assert_array_equal(a,b)
    va,ma=fixed_variance(x,y,past,10);vb,mb=fixed_variance(x,changed,past,10)
    np.testing.assert_array_equal(va,vb);assert 'validation_qlike' not in ma

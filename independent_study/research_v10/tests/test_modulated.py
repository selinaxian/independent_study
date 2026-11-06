import numpy as np
from research_v10.modulated_core import forward,gradients,fit_modulated,softmax
from research_v10.state_core import qlike


def test_local_three_factor_matches_numerical_loss_gradient():
    rng=np.random.default_rng(7);theta=rng.normal(0,.2,(3,2,3));a=np.array([.2,-.3,.1]);b=.03
    cat=rng.integers(0,3,(12,2));pre=np.eye(3)[cat].reshape(12,-1);base=np.ones(12)*2;y=np.exp(rng.normal(0,.3,12))
    gt,ga,gb=gradients(theta,a,b,pre,base,y)
    def loss(t,aa,bb):return qlike(y,forward(t,aa,bb,pre,base)[0]).mean()
    eps=1e-5
    for index in np.ndindex(theta.shape):
        plus=theta.copy();minus=theta.copy();plus[index]+=eps;minus[index]-=eps
        numeric=(loss(plus,a,b)-loss(minus,a,b))/(2*eps)
        np.testing.assert_allclose(gt[index],numeric,atol=1e-9,rtol=1e-5)
    np.testing.assert_allclose(gt.sum(axis=2),0,atol=1e-12)
    for j in range(3):
        plus=a.copy();minus=a.copy();plus[j]+=eps;minus[j]-=eps
        np.testing.assert_allclose(ga[j],(loss(theta,plus,b)-loss(theta,minus,b))/(2*eps),atol=1e-9)
    np.testing.assert_allclose(gb,(loss(theta,a,b+eps)-loss(theta,a,b-eps))/(2*eps),atol=1e-9)


def test_training_target_scope_and_frozen_state():
    rng=np.random.default_rng(8);cat=rng.integers(0,3,(200,2));pre=np.eye(3)[cat].reshape(200,-1)
    y=np.exp(cat[:,0]*.2);base=np.ones(200);q=softmax(rng.normal(size=(3,2,3)),axis=2)
    train=np.arange(200)<120;val=(np.arange(200)>=120)&(np.arange(200)<160)
    p,m=fit_modulated(pre,base,y,train,val,q,'frozen',9)
    np.testing.assert_allclose(softmax(m['theta'],axis=2),q,atol=1e-15)
    altered=y.copy();altered[160:]*=20
    p2,m2=fit_modulated(pre,base,altered,train,val,q,'frozen',9)
    np.testing.assert_array_equal(p,p2);assert (p>0).all()

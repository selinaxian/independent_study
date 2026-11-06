import json
from pathlib import Path
import numpy as np
import torch
from architecture_v3.neural import SoftLayer
from iclab.neural import CompetitiveSNN


def config():
    return {**json.loads(Path('independent_study/configs/experiment.json').read_text())['network'],'epochs':1,'batch_size':8}


def test_soft_allows_multiple_firings_and_lateral_inhibition():
    c={**config(),'hidden':3,'inhibition':.3}
    soft=SoftLayer(3,c,1,False);hard=CompetitiveSNN(3,c,1)
    weights=torch.tensor([[1.2,0.,0.],[1.1,0.,0.],[.8,0.,0.]])
    soft.weight=weights.clone();hard.weight=weights.clone()
    x=np.array([[[1,0,0]]],dtype=bool)
    r,sp=soft.forward(x,record=True)
    assert sp.sum()==2 and hard.batch(x)[:,:3].sum()==1
    np.testing.assert_allclose(r[0,-1],.2,atol=1e-6)


def test_threshold_switch_and_random_weights():
    x=np.ones((16,256,24),dtype=bool)
    fixed=SoftLayer(24,config(),104,False);adapt=SoftLayer(24,config(),104,True)
    for layer in [fixed,adapt]:
        before=layer.weight.clone();layer.fit(x,plastic=False)
        torch.testing.assert_close(before,layer.weight,rtol=0,atol=0)
    assert fixed.theta.count_nonzero()==0 and adapt.theta.min()>0
    assert adapt.batch(x)[:,:64].mean()<fixed.batch(x)[:,:64].mean()


def test_stdp_matches_explicit_soft_reference():
    c={**config(),'hidden':5,'weight_sum':2.}
    layer=SoftLayer(4,c,211);reference=SoftLayer(4,c,211)
    x=np.random.default_rng(42).random((3,12,4))<.3
    w=reference.weight;th=reference.theta;v=torch.zeros(3,5);pt=torch.zeros(3,4);qt=torch.zeros(3,5)
    for t in range(x.shape[1]):
        pre=torch.tensor(x[:,t],dtype=torch.float32);v=np.exp(-1/c['tau_mem'])*v+pre@w.T
        post=(v>=c['threshold']+th).float()
        v=(v*(1-post)-c['inhibition']*(post.sum(1,keepdim=True)-post)).clamp(min=0)
        pt*=np.exp(-1/c['tau_trace']);qt*=np.exp(-1/c['tau_trace'])
        w+=(c['eta_plus']*(1-w)*(post.T@pt)-c['eta_minus']*w*(qt.T@pre))/len(x)
        w.clamp_(1e-6,1);w*=c['weight_sum']/w.sum(1,keepdim=True);w.clamp_(max=1)
        pt+=pre;qt+=post;th*=c['theta_decay'];th+=c['theta_increment']*post.mean(0)
    layer.batch(x,adapt=True,plastic=True)
    torch.testing.assert_close(layer.weight,w,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(layer.theta,th,atol=1e-6,rtol=1e-6)


def test_deep_uses_spikes_freezes_first_and_reload(tmp_path):
    x=np.random.default_rng(9).random((24,32,24))<.2;c=config()
    first=SoftLayer(24,c,104);first.fit(x);rep,seq=first.transform_with_spikes(x)
    assert seq.dtype==bool and seq.shape==(24,32,64)
    before=first.weight.clone();theta=first.theta.clone()
    c2={**c,'hidden':32};second=SoftLayer(64,c2,10104);second.fit(seq)
    assert torch.linalg.norm(second.weight-second.initial_weight)>0
    torch.testing.assert_close(first.weight,before,rtol=0,atol=0);torch.testing.assert_close(first.theta,theta,rtol=0,atol=0)
    path=tmp_path/'layer.npz';second.save(path);other=SoftLayer(64,c2,10104).load(path)
    np.testing.assert_array_equal(second.transform(seq),other.transform(seq))
    np.testing.assert_allclose(other.batch(seq),np.concatenate([other.batch(seq[:7]),other.batch(seq[7:])]),atol=1e-6,rtol=1e-6)
    assert np.c_[rep,second.transform(seq)].shape==(24,288)

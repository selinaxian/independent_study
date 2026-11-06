import numpy as np
import pytest
from iclab.neural import encode
from diagnostics_v2.synthetic import make_pairs
from encoding_v5.encoding import latency,time_codes,decode_latency,decode_rate,fidelity


def test_sign_zero_and_user_examples():
    x=np.array([[[1.,.1,0.,-1.,-.1]]],dtype=np.float32);s=latency(x)
    assert s.shape==(1,8,10) and s.sum()==4
    assert s[0,0,0] and s[0,6,1] and s[0,0,8] and s[0,6,9]
    assert not s[:,:,2].any() and not s[:,:,7].any()
    assert not s[:,:,3:5].any()


def test_nearest_bin_precision_and_zero_presence_distinction():
    x=np.linspace(-1,1,10001).reshape(-1,1,1)
    s=latency(x);restored=decode_latency(s)
    assert abs(restored-x).max()<=1/14+1e-12
    tiny=latency(np.array([[[0.,.0001,-.0001]]]))
    c=time_codes(tiny)
    assert c[0,0,0]==0 and c[0,0,1]==.125 and c[0,0,5]==.125
    np.testing.assert_array_equal(decode_latency(tiny),np.zeros((1,1,3)))


def test_partition_independence_and_future_minute_invariance():
    x=np.random.default_rng(24).uniform(-1,1,(8,12,12)).astype(np.float32)
    s=latency(x)
    np.testing.assert_array_equal(s,np.concatenate([latency(x[:3]),latency(x[3:])]))
    altered=x.copy();altered[:,6:]=-altered[:,6:]
    np.testing.assert_array_equal(s[:,:48],latency(altered)[:,:48])
    early=np.zeros((1,2,2));early[0,0,0]=.01;early[0,1,1]=1.
    out=latency(early);assert out[0,7,0] and out[0,8,1]


def test_time_code_preserves_slot_and_presence():
    x=np.linspace(-1,1,41).reshape(-1,1,1);s=latency(x);c=time_codes(s)
    b=s.reshape(-1,1,8,2)
    for sample in range(len(x)):
        for channel in range(2):
            if c[sample,0,channel]==0:assert not b[sample,0,:,channel].any()
            else:assert b[sample,0,int(8-8*c[sample,0,channel]),channel]
    bad=s.copy();bad[0,:,0]=True
    with pytest.raises(ValueError):time_codes(bad)
    with pytest.raises(ValueError):latency(np.array([[[1.1]]]))
    with pytest.raises(ValueError):latency(np.array([[[np.nan]]]))


def test_ab_ba_same_counts_and_tail_but_different_order():
    for gap in [1,5]:
        x,labels,ids,pos=make_pairs(20,gap);s=latency(x)
        np.testing.assert_array_equal(s[::2].sum(axis=1),s[1::2].sum(axis=1))
        np.testing.assert_array_equal(s[::2,-24:],s[1::2,-24:])
        assert not np.array_equal(s[::2],s[1::2])


def test_rate_noise_and_latency_activity_are_distinct():
    x=np.full((12000,1,1),.1,dtype=np.float32);ids=np.arange(len(x))
    rate=encode(x,ids,'rate',104);lat=latency(x)
    f=fidelity(x,rate,'rate');g=fidelity(x,lat,'latency')
    assert abs(f['small_input_no_spike_fraction']-.97**8)<.02
    assert g['small_input_no_spike_fraction']==0
    assert f['input_spikes_per_minute']<.27 and g['input_spikes_per_minute']==1
    assert decode_rate(rate).shape==x.shape

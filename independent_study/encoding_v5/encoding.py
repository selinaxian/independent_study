"""Deterministic within-minute amplitude-to-time encoding; no market causality claim."""
import numpy as np


def latency(x,steps=8):
    x=np.asarray(x)
    if x.ndim!=3 or steps<2 or not np.isfinite(x).all() or np.max(abs(x),initial=0)>1:
        raise ValueError('Expected finite [samples, minutes, features] in [-1,1], steps>=2')
    channels=np.concatenate([np.maximum(x,0),np.maximum(-x,0)],axis=-1)
    slots=np.floor((steps-1)*(1-channels.astype(np.float64))+.5).astype(np.int64)
    out=(np.arange(steps)[None,None,:,None]==slots[:,:,None,:])&(channels[:,:,None,:]>0)
    return out.reshape(len(x),x.shape[1]*steps,2*x.shape[2])


def blocks(spikes,steps=8):
    s=np.asarray(spikes)
    if s.ndim!=3 or s.shape[1]%steps or s.shape[2]%2:raise ValueError('Invalid spike dimensions')
    b=s.reshape(len(s),-1,steps,s.shape[-1])
    if not np.isin(b,[0,1]).all() or (b.sum(axis=2)>1).any():raise ValueError('At most one spike per channel/minute')
    return b


def time_codes(spikes,steps=8):
    b=blocks(spikes,steps)
    return ((steps-b.argmax(axis=2))/steps*(b.sum(axis=2)>0)).astype(np.float32)


def decode_latency(spikes,steps=8):
    b=blocks(spikes,steps)
    a=(1-b.argmax(axis=2)/(steps-1))*(b.sum(axis=2)>0)
    h=a.shape[-1]//2
    return a[:,:,:h]-a[:,:,h:]


def decode_rate(spikes,steps=8,p_max=.3,clip=True):
    s=np.asarray(spikes);a=s.reshape(len(s),-1,steps,s.shape[-1]).sum(axis=2)/(steps*p_max)
    if clip:a=np.clip(a,0,1)
    h=a.shape[-1]//2
    return a[:,:,:h]-a[:,:,h:]


def fidelity(x,spikes,kind):
    x=np.asarray(x)
    estimate=decode_latency(spikes) if kind=='latency' else decode_rate(spikes)
    b=spikes.reshape(len(x),x.shape[1],8,-1)
    signed_counts=b.sum(axis=2);h=x.shape[-1]
    present=(signed_counts[:,:,:h]+signed_counts[:,:,h:])>0
    small=(abs(x)>0)&(abs(x)<=.1)
    return {'mse':float(np.mean((estimate-x)**2)),
            'mae':float(np.mean(abs(estimate-x))),
            'max_abs_error':float(np.max(abs(estimate-x))),
            'input_spikes_per_minute':float(b.sum(axis=(2,3)).mean()),
            'small_nonzero_values':int(small.sum()),
            'small_input_no_spike_fraction':float((~present[small]).mean()) if small.any() else None,
            'unclipped_rate_mse':float(np.mean((decode_rate(spikes,clip=False)-x)**2)) if kind=='rate' else None}

"""Frozen LIF recorder: the legacy branch exactly matches the existing kernel."""
import math
import numpy as np
import torch


@torch.jit.script
def recorded_kernel(inp:torch.Tensor,weight:torch.Tensor,theta:torch.Tensor,
                    decay:float,threshold_base:float,inhibition:float,steps_per_minute:int,no_wta:bool):
    n,steps,dims=inp.shape;h=theta.shape[0]
    v=torch.zeros((n,h),dtype=inp.dtype);total=torch.zeros_like(v);recent=torch.zeros_like(v)
    first=torch.zeros_like(v);last=torch.zeros_like(v)
    counts=torch.zeros((n,steps//steps_per_minute,h),dtype=inp.dtype)
    volts=torch.zeros_like(counts);phase=torch.zeros((n,steps_per_minute),dtype=inp.dtype)
    occupied=torch.zeros(n,dtype=inp.dtype);arange=torch.arange(n);threshold=threshold_base+theta
    for t in range(steps):
        v=decay*v+inp[:,t]@weight.T
        if no_wta:
            post=(v>=threshold).to(inp.dtype)
        else:
            excess,winner=(v-threshold).max(dim=1)
            post=torch.zeros_like(v);post[arange,winner]=(excess>=0).to(inp.dtype)
        fired=post.sum(dim=1,keepdim=True)
        v=(v*(1-post)-inhibition*fired*(1-post)).clamp(min=0)
        total+=post
        if t>=steps-3*steps_per_minute:recent+=post
        code=torch.full_like(v,float(steps-t)/float(steps))
        first=torch.where((first==0)&(post>0),code,first)
        last=torch.where(post>0,code,last)
        counts[:,t//steps_per_minute]+=post
        if (t+1)%steps_per_minute==0:volts[:,t//steps_per_minute]=v/threshold
        phase[:,t%steps_per_minute]+=fired[:,0]
        occupied+=(fired[:,0]>0).to(inp.dtype)
    legacy=torch.cat([total/steps,recent/(3*steps_per_minute),v/threshold],dim=1)
    return legacy,torch.cat([legacy,first,last],dim=1),counts/steps_per_minute,volts,phase/(steps//steps_per_minute),occupied/steps


@torch.no_grad()
def trace(spikes,weight,theta,cfg,no_wta=False,inhibition=None):
    w=torch.as_tensor(np.asarray(weight),dtype=torch.float32).clone()
    t=torch.as_tensor(np.asarray(theta),dtype=torch.float32).clone();before_w=w.clone();before_t=t.clone()
    pieces=[[] for _ in range(6)]
    for start in range(0,len(spikes),cfg['batch_size']):
        x=torch.as_tensor(spikes[start:start+cfg['batch_size']],dtype=torch.float32)
        result=recorded_kernel(x,w,t,math.exp(-1/cfg['tau_mem']),float(cfg['threshold']),
                     float(cfg['inhibition'] if inhibition is None else inhibition),cfg['steps_per_minute'],no_wta)
        for rows,a in zip(pieces,result):rows.append(a.numpy())
    torch.testing.assert_close(w,before_w,rtol=0,atol=0);torch.testing.assert_close(t,before_t,rtol=0,atol=0)
    values=[np.concatenate(rows) for rows in pieces]
    return {'legacy':values[0],'timing':values[1],'minute_counts':values[2].reshape(len(spikes),-1),
            'minute_voltage':values[3].reshape(len(spikes),-1),'phase':values[4],'occupied':values[5]}


def views(record,input_time,weight,projection):
    result={k:record[k] for k in ['legacy','timing','minute_counts','minute_voltage']}
    result['voltage_project192']=(record['minute_voltage']@projection).astype(np.float32)
    result['synaptic_time']=(input_time@np.asarray(weight).T).reshape(len(input_time),-1)
    return result


def diagnostics(record,weight,mask):
    w=np.asarray(weight,dtype=float);h=len(w);r=record['legacy'][mask]
    cos=w@w.T/np.maximum(np.linalg.norm(w,axis=1)[:,None]*np.linalg.norm(w,axis=1)[None,:],1e-30)
    sv=np.linalg.svd(w,compute_uv=False)
    return {'active_neurons':int((r[:,:h].sum(axis=0)>0).sum()),'mean_rate':float(r[:,:h].mean()),
            'population_occupied_steps':float(record['occupied'][mask].mean()),
            'spikes_per_phase':record['phase'][mask].mean(axis=0),
            'weight_cosine':float(cos[~np.eye(h,dtype=bool)].mean()),'weight_singular_values':sv,
            'weight_rank':int(np.linalg.matrix_rank(w)),'weight_condition':float(sv[0]/max(sv[-1],1e-30))}

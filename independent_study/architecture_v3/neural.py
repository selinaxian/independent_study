import math
import numpy as np
import torch
from iclab.neural import CompetitiveSNN


@torch.jit.script
def soft_batch(inp:torch.Tensor,w:torch.Tensor,theta:torch.Tensor,adapt:bool,plastic:bool,
               record:bool,decay:float,trace_decay:float,threshold:float,inhibition:float,
               eta_plus:float,eta_minus:float,weight_sum:float,theta_decay:float,
               theta_increment:float,recent_steps:int):
    n,steps,inputs=inp.shape;h=w.shape[0]
    v=torch.zeros((n,h));pt=torch.zeros((n,inputs));qt=torch.zeros((n,h))
    total=torch.zeros((n,h));recent=torch.zeros((n,h))
    history=torch.zeros((n,steps if record else 0,h),dtype=torch.bool)
    for t in range(steps):
        pre=inp[:,t];v=decay*v+pre@w.T
        post=(v>=threshold+theta).float()
        v=torch.clamp(v*(1-post)-inhibition*(post.sum(1,keepdim=True)-post),min=0.)
        total+=post
        if t>=steps-recent_steps:recent+=post
        if record:history[:,t]=post>0
        pt*=trace_decay;qt*=trace_decay
        if plastic:
            w+=(eta_plus*(1-w)*(post.T@pt)-eta_minus*w*(qt.T@pre))/n
            w.clamp_(min=1e-6,max=1.)
            w*=weight_sum/w.sum(1,keepdim=True)
            w.clamp_(max=1.)
        pt+=pre;qt+=post
        if adapt:
            theta*=theta_decay
            theta+=theta_increment*post.mean(0)
    return torch.cat((total/steps,recent/recent_steps,v/(threshold+theta)),1),history


class SoftLayer(CompetitiveSNN):
    def __init__(self,inputs,config,seed,adaptive=True):
        super().__init__(inputs,config,seed)
        self.adaptive=adaptive

    @torch.no_grad()
    def forward(self,spikes,adapt=False,plastic=False,record=False):
        c=self.cfg
        rep,seq=soft_batch(torch.as_tensor(spikes,dtype=torch.float32),self.weight,self.theta,
            adapt and self.adaptive,plastic,record,math.exp(-1/c['tau_mem']),math.exp(-1/c['tau_trace']),
            float(c['threshold']),float(c['inhibition']),float(c['eta_plus']),float(c['eta_minus']),
            float(c['weight_sum']),float(c['theta_decay']),float(c['theta_increment']),3*c['steps_per_minute'])
        return rep.numpy(),seq.numpy()

    def batch(self,spikes,adapt=False,plastic=False):
        return self.forward(spikes,adapt,plastic)[0]

    def transform_with_spikes(self,spikes):
        reps=[];trains=[]
        for start in range(0,len(spikes),self.cfg['batch_size']):
            r,s=self.forward(spikes[start:start+self.cfg['batch_size']],record=True)
            reps.append(r);trains.append(s)
        return np.concatenate(reps),np.concatenate(trains)

"""TorchScript execution of the unchanged v1 dynamics; checked against v1."""
import math
import numpy as np
import torch
from iclab.neural import CompetitiveSNN


@torch.jit.script
def sequence_kernel(inp: torch.Tensor, weight: torch.Tensor, theta: torch.Tensor,
                    adapt: bool, plastic: bool, decay: float, trace_decay: float,
                    threshold_base: float, inhibition: float, plus: float, minus: float,
                    row_sum: float, theta_decay: float, theta_inc: float, recent_steps: int):
    n, steps, dims = inp.shape
    h = theta.shape[0]
    v = torch.zeros((n, h), dtype=inp.dtype)
    pre_trace = torch.zeros((n, dims), dtype=inp.dtype)
    post_trace = torch.zeros((n, h), dtype=inp.dtype)
    total = torch.zeros((n, h), dtype=inp.dtype)
    recent = torch.zeros((n, h), dtype=inp.dtype)
    arange = torch.arange(n)
    for t in range(steps):
        pre = inp[:, t]
        v = decay*v + pre@weight.T
        threshold = threshold_base+theta
        excess, winner = (v-threshold).max(dim=1)
        post = torch.zeros_like(v)
        post[arange, winner] = (excess >= 0).to(inp.dtype)
        fired = post.sum(dim=1, keepdim=True)
        v = (v*(1-post)-inhibition*fired*(1-post)).clamp(min=0)
        total += post
        if t >= steps-recent_steps:
            recent += post
        pre_trace *= trace_decay
        post_trace *= trace_decay
        if plastic:
            delta = (plus*(1-weight)*(post.T@pre_trace)-minus*weight*(post_trace.T@pre))/n
            weight += delta
            weight.clamp_(min=1e-6, max=1.)
            weight *= row_sum/weight.sum(dim=1, keepdim=True)
            weight.clamp_(max=1.)
        pre_trace += pre
        post_trace += post
        if adapt:
            theta *= theta_decay
            theta += theta_inc*post.mean(dim=0)
    state = v/(threshold_base+theta)
    return torch.cat([total/steps, recent/recent_steps, state], dim=1)


class FastSNN(CompetitiveSNN):
    @torch.no_grad()
    def batch(self, spikes, adapt=False, plastic=False):
        c = self.cfg
        inp = torch.as_tensor(spikes, dtype=torch.float32)
        return sequence_kernel(inp, self.weight, self.theta, adapt, plastic,
             math.exp(-1/c["tau_mem"]), math.exp(-1/c["tau_trace"]), float(c["threshold"]),
             float(c["inhibition"]), float(c["eta_plus"]), float(c["eta_minus"]),
             float(c["weight_sum"]), float(c["theta_decay"]), float(c["theta_increment"]),
             3*c["steps_per_minute"]).numpy()

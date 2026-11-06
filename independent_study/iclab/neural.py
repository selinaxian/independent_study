import numpy as np
import torch


def uniforms(ids, shape, seed):
    """Counter-based uniforms: identical for an item regardless of batch partition."""
    with np.errstate(over="ignore"):
        x = np.asarray(ids, dtype=np.uint32).reshape((-1,)+(1,)*len(shape))*np.uint32(2654435761)
        x = x + np.arange(np.prod(shape), dtype=np.uint32).reshape((1,)+shape)*np.uint32(2246822519) + np.uint32(seed)
        x = (x ^ (x >> 16))*np.uint32(0x7feb352d)
        x = (x ^ (x >> 15))*np.uint32(0x846ca68b)
        x = x ^ (x >> 16)
    return (x.astype(np.float64)+0.5)/4294967296.0


def encode(x, ids, encoding, seed, steps=8, p_max=.3, shuffled=False):
    channels = np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], axis=-1)
    n, minutes, dims = channels.shape
    p = np.repeat(channels, steps, axis=1)*p_max
    if encoding == "rate":
        spikes = uniforms(ids, (minutes*steps, dims), seed) < p
    elif encoding == "density":
        phase = (np.arange(dims)*0.6180339887498949) % 1
        cumulative = np.cumsum(p, axis=1, dtype=np.float64)+phase
        previous = np.concatenate([np.broadcast_to(phase, (n, 1, dims)), cumulative[:, :-1]], axis=1)
        spikes = np.floor(cumulative) > np.floor(previous)
    else:
        raise ValueError(encoding)
    if shuffled:
        blocks = spikes.reshape(n, minutes, steps, dims)
        order = np.argsort(uniforms(ids, (minutes-1,), seed+7919), axis=1)
        order = np.concatenate([order, np.full((n, 1), minutes-1)], axis=1)
        spikes = blocks[np.arange(n)[:, None], order].reshape(n, minutes*steps, dims)
    return spikes


def stdp_delta(weight, pre_trace, post_trace, pre, post, plus, minus):
    return (plus*(1-weight)*(post.T@pre_trace) - minus*weight*(post_trace.T@pre))/len(pre)


class CompetitiveSNN:
    def __init__(self, inputs, config, seed):
        self.cfg = config
        self.seed = seed
        rng = np.random.default_rng(seed)
        w = rng.uniform(.02, .98, (config["hidden"], inputs)).astype(np.float32)
        w *= config["weight_sum"]/w.sum(axis=1, keepdims=True)
        self.weight = torch.from_numpy(w)
        self.initial_weight = self.weight.clone()
        self.theta = torch.zeros(config["hidden"])
        self.history = []

    @torch.no_grad()
    def batch(self, spikes, adapt=False, plastic=False):
        c = self.cfg
        inp = torch.as_tensor(spikes, dtype=torch.float32)
        n, steps, _ = inp.shape
        h = len(self.theta)
        v = torch.zeros(n, h)
        pre_trace, post_trace = torch.zeros(n, inp.shape[2]), torch.zeros(n, h)
        total, recent = torch.zeros(n, h), torch.zeros(n, h)
        decay = np.exp(-1/c["tau_mem"])
        trace_decay = np.exp(-1/c["tau_trace"])
        arange = torch.arange(n)
        for t in range(steps):
            pre = inp[:, t]
            v = decay*v + pre@self.weight.T
            threshold = c["threshold"]+self.theta
            excess, winner = (v-threshold).max(dim=1)
            post = torch.zeros_like(v)
            post[arange, winner] = (excess >= 0).float()
            fired = post.sum(dim=1, keepdim=True)
            v = (v*(1-post) - c["inhibition"]*fired*(1-post)).clamp(min=0)
            total += post
            if t >= steps-3*c["steps_per_minute"]: recent += post
            pre_trace *= trace_decay
            post_trace *= trace_decay
            if plastic:
                self.weight += stdp_delta(self.weight, pre_trace, post_trace, pre, post, c["eta_plus"], c["eta_minus"])
                self.weight.clamp_(min=1e-6, max=1)
                self.weight *= c["weight_sum"]/self.weight.sum(dim=1, keepdim=True)
                # Configured sum=4.8 over 24 synapses: clipping still enforces bounds.
                self.weight.clamp_(max=1)
            pre_trace += pre
            post_trace += post
            if adapt:
                self.theta *= c["theta_decay"]
                self.theta += c["theta_increment"]*post.mean(dim=0)
        state = v/(c["threshold"]+self.theta)
        return torch.cat([total/steps, recent/(3*c["steps_per_minute"]), state], dim=1).numpy()

    def fit(self, spikes, plastic=True):
        rng = np.random.default_rng(self.seed+3001)
        bs = self.cfg["batch_size"]
        for epoch in range(self.cfg["epochs"]):
            order = rng.permutation(len(spikes))
            fired = np.zeros(self.cfg["hidden"])
            for start in range(0, len(order), bs):
                rep = self.batch(spikes[order[start:start+bs]], adapt=True, plastic=plastic)
                fired += rep[:, :len(fired)].sum(axis=0)
            self.history.append({"epoch": epoch+1, "active_neurons": int((fired>0).sum()), "mean_rate": float(fired.sum()/len(spikes)/len(fired)), "weight_change_l2": float(torch.linalg.norm(self.weight-self.initial_weight)), "theta_mean": float(self.theta.mean())})

    def transform(self, spikes):
        return np.concatenate([self.batch(spikes[i:i+self.cfg["batch_size"]]) for i in range(0, len(spikes), self.cfg["batch_size"])])

    def save(self, path):
        np.savez_compressed(path, weight=self.weight.numpy(), initial_weight=self.initial_weight.numpy(), theta=self.theta.numpy())

    def load(self, path):
        a = np.load(path)
        self.weight = torch.from_numpy(a["weight"].copy())
        self.initial_weight = torch.from_numpy(a["initial_weight"].copy())
        self.theta = torch.from_numpy(a["theta"].copy())
        return self

    def diagnostics(self, representation):
        h = self.cfg["hidden"]
        w = self.weight.numpy()
        cos = (w@w.T)/(np.linalg.norm(w, axis=1)[:, None]*np.linalg.norm(w, axis=1)[None, :])
        centered = representation-representation.mean(axis=0)
        sv = np.linalg.svd(centered, compute_uv=False)**2
        probs = sv/sv.sum() if sv.sum() else sv
        return {"active_neurons": int((representation[:, :h].sum(axis=0)>0).sum()), "mean_rate": float(representation[:, :h].mean()), "weight_change_l2": float(torch.linalg.norm(self.weight-self.initial_weight)), "mean_weight_cosine": float(cos[~np.eye(h, dtype=bool)].mean()), "effective_rank": float(np.exp(-np.sum(probs*np.log(probs+1e-30)))) if sv.sum() else 0., "row_sum_min": float(w.sum(axis=1).min()), "weight_max": float(w.max()), "history": self.history}

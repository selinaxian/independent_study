"""Competitive LIF neurons with bounded, trace-based pair STDP (no autograd)."""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NetworkConfig:
    hidden: int = 64
    steps_per_minute: int = 10
    max_spike_probability: float = 0.25
    tau_membrane: float = 20.0
    tau_pre: float = 20.0
    tau_post: float = 20.0
    threshold: float = 1.0
    inhibition: float = 0.20
    eta_plus: float = 0.02
    eta_minus: float = 0.021
    weight_max: float = 1.0
    weight_sum: float = 4.0
    theta_increment: float = 0.05
    tau_theta: float = 2000.0
    epochs: int = 3
    batch_size: int = 256

    def __post_init__(self):
        if min(self.tau_pre, self.tau_post) < 1 or self.tau_membrane <= 0:
            raise ValueError("Invalid neural time constants")
        if not 0 < self.max_spike_probability <= 1:
            raise ValueError("Spike probability must be in (0, 1]")
        if min(self.hidden, self.steps_per_minute, self.epochs, self.batch_size) < 1:
            raise ValueError("Counts must be positive")


def signed_channels(x: torch.Tensor) -> torch.Tensor:
    """Alternating [feature0+, feature0-, feature1+, feature1-, ...]."""
    return torch.stack((x.clamp(min=0), (-x).clamp(min=0)), dim=-1).flatten(-2)


def stdp_delta(weight: torch.Tensor, pre: torch.Tensor, post: torch.Tensor,
               x: torch.Tensor, y: torch.Tensor, cfg: NetworkConfig):
    """Exclude zero-lag pairs: decay old traces, update weights, then add spikes.

    pre-before-post potentiates; post-before-pre depresses. Batch reductions
    are means so a partial final batch does not change the learning rate.
    """
    pre_old = pre * (1.0 - 1.0 / cfg.tau_pre)
    post_old = post * (1.0 - 1.0 / cfg.tau_post)
    potentiation = (y.T @ pre_old) / x.shape[0]
    depression = (post_old.T @ x) / x.shape[0]
    delta = (cfg.eta_plus * (cfg.weight_max - weight) * potentiation
             - cfg.eta_minus * weight * depression)
    return delta, pre_old + x, post_old + y


class CompetitiveSTDP:
    def __init__(self, input_features: int, cfg: NetworkConfig, seed: int):
        self.cfg, self.seed = cfg, seed
        self.inputs = input_features * 2
        if not 0 < cfg.weight_sum < self.inputs * cfg.weight_max:
            raise ValueError("Infeasible weight normalization")
        rng = torch.Generator().manual_seed(seed)
        self.weight = torch.rand((cfg.hidden, self.inputs), generator=rng) * 0.3 + 0.05
        self.weight *= cfg.weight_sum / self.weight.sum(dim=1, keepdim=True)
        self.initial_weight = self.weight.clone()
        self.theta = torch.zeros(cfg.hidden)

    def state_dict(self):
        return {"weight": self.weight.clone(), "theta": self.theta.clone(),
                "initial_weight": self.initial_weight.clone()}

    def load_state_dict(self, state):
        for name in ("weight", "theta", "initial_weight"):
            setattr(self, name, state[name].clone().to(dtype=torch.float32, device="cpu"))

    def _normalize_weights(self):
        # Positive weights and ample capacity make this converge immediately in
        # normal operation. Repeating also handles a rare saturated synapse.
        for _ in range(4):
            self.weight *= self.cfg.weight_sum / self.weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
            self.weight.clamp_(0.0, self.cfg.weight_max)

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, rng: torch.Generator, *, learn_weights=False,
                 adapt_thresholds=False, shuffle_rng: torch.Generator | None = None):
        cfg = self.cfg
        b, minutes, _ = x.shape
        probabilities = signed_channels(x) * cfg.max_spike_probability
        # Bernoulli time-bin implementation of rate/Poisson-style encoding.
        noise = torch.rand((b, minutes, cfg.steps_per_minute, self.inputs), generator=rng)
        spikes = (noise < probabilities.unsqueeze(2)).to(torch.float32)
        if shuffle_rng is not None:
            order = torch.rand((b, minutes - 1), generator=shuffle_rng).argsort(dim=1)
            order = torch.cat((order, torch.full((b, 1), minutes - 1, dtype=torch.long)), dim=1)
            spikes = spikes.gather(1, order[:, :, None, None].expand_as(spikes))
        spikes = spikes.flatten(1, 2)
        v = torch.zeros((b, cfg.hidden))
        counts = torch.zeros_like(v)
        pre = torch.zeros((b, self.inputs))
        post = torch.zeros_like(v)
        beta = math.exp(-1.0 / cfg.tau_membrane)
        theta_decay = math.exp(-1.0 / cfg.tau_theta)
        steps = spikes.shape[1]
        for step in range(steps):
            incoming = spikes[:, step]
            v.mul_(beta).add_(incoming @ self.weight.T)
            threshold = cfg.threshold + self.theta
            candidates = v >= threshold
            active = candidates.any(dim=1)
            rank = (v / threshold).masked_fill(~candidates, -float("inf"))
            winner = rank.argmax(dim=1)
            outgoing = torch.zeros_like(v)
            outgoing.scatter_(1, winner[:, None], active[:, None].to(torch.float32))
            counts.add_(outgoing)
            v.mul_(1.0 - outgoing)
            v.sub_(cfg.inhibition * active[:, None]).clamp_(min=0)
            if learn_weights:
                delta, pre, post = stdp_delta(self.weight, pre, post, incoming, outgoing, cfg)
                self.weight.add_(delta).clamp_(0.0, cfg.weight_max)
            if adapt_thresholds:
                self.theta.mul_(theta_decay).add_(cfg.theta_increment * outgoing.mean(dim=0))
        if learn_weights:
            self._normalize_weights()
        representation = torch.cat((counts / steps, v / (cfg.threshold + self.theta)), dim=1)
        return representation, counts

    def fit(self, windows: np.ndarray, *, learn_weights=True, log_prefix=""):
        """Unsupervised API intentionally accepts no future targets."""
        cfg = self.cfg
        x = torch.from_numpy(windows)
        history = []
        for epoch in range(cfg.epochs):
            started = time.monotonic()
            rng = torch.Generator().manual_seed(self.seed + 10000 * (epoch + 1))
            totals = torch.zeros(cfg.hidden)
            # Chronological samples; internal membrane/traces reset per window.
            for start in range(0, len(x), cfg.batch_size):
                _, counts = self.simulate(x[start:start + cfg.batch_size], rng,
                                          learn_weights=learn_weights, adapt_thresholds=True)
                totals += counts.sum(dim=0)
            steps = len(x) * x.shape[1] * cfg.steps_per_minute
            similarity = torch.nn.functional.normalize(self.weight, dim=1)
            similarity = similarity @ similarity.T
            off_diagonal = ~torch.eye(cfg.hidden, dtype=torch.bool)
            record = {
                "epoch": epoch + 1,
                "elapsed_seconds": time.monotonic() - started,
                "mean_spikes_per_sample": float(totals.sum() / len(x)),
                "mean_neuron_rate_per_step": float(totals.sum() / steps / cfg.hidden),
                "active_neurons": int((totals > 0).sum()),
                "weight_relative_change": float(torch.linalg.vector_norm(self.weight - self.initial_weight)
                                                / torch.linalg.vector_norm(self.initial_weight)),
                "mean_pairwise_weight_cosine": float(similarity[off_diagonal].mean()),
                "threshold_min": float(cfg.threshold + self.theta.min()),
                "threshold_max": float(cfg.threshold + self.theta.max()),
                "weight_min": float(self.weight.min()),
                "weight_max": float(self.weight.max()),
                "neuron_spikes": totals.tolist(),
            }
            if not torch.isfinite(self.weight).all() or not torch.isfinite(self.theta).all():
                raise FloatingPointError("Nonfinite network parameters")
            history.append(record)
            print(f"{log_prefix} epoch {epoch + 1}/{cfg.epochs}: "
                  f"spikes/window={record['mean_spikes_per_sample']:.2f}, "
                  f"active={record['active_neurons']}/{cfg.hidden}, "
                  f"weight_change={record['weight_relative_change']:.4f}, "
                  f"seconds={record['elapsed_seconds']:.1f}", flush=True)
        return history

    @torch.no_grad()
    def transform(self, windows: np.ndarray, encoding_seed: int, shuffle_history=False):
        x = torch.from_numpy(windows)
        rng = torch.Generator().manual_seed(encoding_seed)
        order_rng = torch.Generator().manual_seed(encoding_seed + 918273) if shuffle_history else None
        results = []
        for start in range(0, len(x), self.cfg.batch_size):
            response, _ = self.simulate(x[start:start + self.cfg.batch_size], rng,
                                        shuffle_rng=order_rng)
            results.append(response.numpy())
        return np.concatenate(results)

    def save(self, path, feature_names, extra=None):
        torch.save({"state_dict": self.state_dict(), "config": asdict(self.cfg),
                    "seed": self.seed, "feature_names": list(feature_names),
                    "extra": extra or {}}, path)

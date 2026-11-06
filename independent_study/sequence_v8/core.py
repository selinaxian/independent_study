"""Competition-induced STDP prototypes on delayed spike arrivals.

This is a deliberately explicit variant: training induces a winner post
spike after input arrivals, potentiates its active synapses, and normalizes
the input budget. It is not a free-running LIF STDP training simulation.
Inference uses integrate/fire/reset units with the stated fixed threshold.
"""
import numpy as np


def delayed_inputs(sequence, anchors, delays):
    """Past arrivals only; flatten channel first, then delay."""
    sequence = np.asarray(sequence, dtype=float)
    anchors = np.asarray(anchors, dtype=int)
    delays = np.asarray(delays, dtype=int)
    if (delays < 0).any():
        raise ValueError('Only causal, nonnegative delays are allowed')
    indices = anchors[:, None] - delays[None, :]
    if (indices < 0).any() or (indices >= sequence.shape[1]).any():
        raise ValueError('An arrival has no observed source minute')
    values = sequence[np.arange(len(sequence))[:, None], indices]
    return values.transpose(0, 2, 1).reshape(len(sequence), -1)


class CompetitivePatterns:
    def __init__(self, dimensions, cfg, seed):
        self.cfg = dict(cfg)
        rng = np.random.default_rng(seed)
        w = rng.uniform(.4, .6, (cfg['hidden'], dimensions))
        self.weight = w / w.sum(axis=1, keepdims=True) * cfg['weight_sum']
        self.initial_weight = self.weight.copy()
        self.usage = np.zeros(cfg['hidden'], dtype=float)

    def fit(self, arrivals, seed):
        """No teacher events or target labels accepted by hidden fit."""
        rng = np.random.default_rng(seed)
        for _ in range(self.cfg['epochs']):
            for index in rng.permutation(len(arrivals)):
                pre = arrivals[index]
                if not pre.any():
                    continue
                self.usage *= self.cfg['usage_decay']
                drive = self.weight @ pre - self.cfg['usage_penalty'] * self.usage
                winner = int(np.argmax(drive))
                # Input pre at local phase 0; induced winner post at phase 1.
                w = self.weight[winner]
                w += self.cfg['eta'] * (1-w) * pre
                w *= self.cfg['weight_sum'] / w.sum()
                np.clip(w, 0, 1, out=w)
                self.usage[winner] += 1
        return self

    def response(self, arrivals, threshold=None):
        threshold = self.cfg['threshold'] if threshold is None else threshold
        # Each decision window starts at zero potential; one integration,
        # threshold crossing, and reset. History lives in the delay bank.
        potential = arrivals @ self.weight.T
        fired = (potential > threshold).astype(float)
        after_reset = potential * (1-fired)
        return fired, potential, after_reset


KINDS = ['A_B_gap2', 'A_B_gap1', 'A_B_gap3', 'B_A_gap2',
         'A_B_simultaneous', 'A_C_gap2', 'C_B_gap2', 'C_D_gap2']


def generate(per_kind, seed, identities=(0, 1, 2, 3), channels=24):
    rng = np.random.default_rng(seed)
    a, b, c, d = identities
    specifications = [(a,b,2), (a,b,1), (a,b,3), (b,a,2),
                      (a,b,0), (a,c,2), (c,b,2), (c,d,2)]
    labels = np.repeat(np.arange(len(KINDS)), per_kind)
    rng.shuffle(labels)
    anchors = rng.integers(8, 18, len(labels))
    spikes = np.zeros((len(labels), 26, channels), dtype=float)
    events = np.zeros((len(labels), 26), dtype=bool)
    for i, kind in enumerate(labels):
        first, second, gap = specifications[kind]
        anchor = anchors[i]
        spikes[i, anchor-gap, first] = 1
        spikes[i, anchor, second] = 1
        if kind == 0:
            events[i, anchor+1] = True
        elif kind == 3:
            events[i, anchor-2] = True
    assert np.all(spikes.sum(axis=(1,2)) == 2)
    return dict(spikes=spikes, anchors=anchors, events=events, kinds=labels,
                target=(labels == 0).astype(float), identities=np.array(identities))


def output_training_arrays(response, data):
    n, minutes = data['events'].shape
    s = np.zeros((n, minutes, response.shape[1]), dtype=float)
    s[np.arange(n), data['anchors']] = response
    segments = np.repeat(np.arange(n), minutes)
    return s.reshape(n*minutes, -1), data['events'].ravel(), segments

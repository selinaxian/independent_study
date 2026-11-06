"""Local three-factor auxiliary update, replayed only on matured training data.

No chronological online learning claim: retain v8's shuffled offline schedule.
Delayed pre channels arrive before a competition-induced winner post. Its
synaptic eligibility is replayed at the end of a complete market horizon.
"""
import numpy as np
from event_stdp_v7.core import lagged
from sequence_v8.core import CompetitivePatterns


def matured_feedback(events, segments, train, eligible, horizon):
    """Mask every future minute, not just the endpoint of a training window."""
    mask = np.asarray(train, bool) & np.asarray(eligible, bool)
    target = np.zeros(len(events), bool)
    for d in range(1, horizon+1):
        mask &= lagged(np.asarray(train, bool), segments, -d)
        target |= lagged(np.asarray(events, bool), segments, -d)
    y = np.where(mask, target.astype(float), np.nan)
    if not mask.any():
        raise ValueError('No matured training feedback')
    return mask, y, float(y[mask].mean())


def mismatch_feedback(target, segments, mask, seed):
    """Permute only matured training outcomes; preserve each segment's count."""
    result = np.full(len(target), np.nan)
    rng = np.random.default_rng(seed)
    for sid in np.unique(segments[mask]):
        idx = np.flatnonzero(mask & (segments == sid))
        if len(idx) < 2:
            result[idx] = target[idx]
            continue
        shift = rng.integers(6, len(idx)-5) if len(idx) >= 12 else rng.integers(1, len(idx))
        result[idx] = np.roll(target[idx], shift)
    return result


def eligibility(pre, age, tau):
    if age < 0 or tau <= 0:
        raise ValueError('Invalid eligibility age/tau')
    return np.asarray(pre, float) * np.exp(-age/tau)


def auxiliary_delta(weight, trace, modulation, eta):
    bound = 1-weight if modulation >= 0 else weight
    return eta * modulation * trace * bound


class EventModulatedPatterns(CompetitivePatterns):
    def fit(self, arrivals, seed, feedback=None, baseline=0.):
        if feedback is not None and len(feedback) != len(arrivals):
            raise ValueError('Feedback/sample mismatch')
        if not 0 <= baseline <= 1:
            raise ValueError('Invalid training event rate')
        self.audit = dict(positive_updates=0, negative_updates=0, positive_delta=0.,
                          negative_delta=0., unsupervised_delta=0., matured_updates=0)
        rng = np.random.default_rng(seed)
        for _ in range(self.cfg['epochs']):
            for index in rng.permutation(len(arrivals)):
                pre = arrivals[index]
                if not pre.any():
                    continue
                self.usage *= self.cfg['usage_decay']
                drive = self.weight @ pre - self.cfg['usage_penalty']*self.usage
                winner = int(np.argmax(drive))
                w = self.weight[winner]
                du = self.cfg['eta']*(1-w)*pre
                dm = None
                if feedback is not None and np.isfinite(feedback[index]):
                    modulation = float(feedback[index]-baseline)
                    e = eligibility(pre, self.cfg['horizon'], self.cfg['eligibility_tau'])
                    dm = auxiliary_delta(w, e, modulation, self.cfg['feedback_eta'])
                    self.audit['matured_updates'] += 1
                    if modulation > 0:
                        self.audit['positive_updates'] += 1
                        self.audit['positive_delta'] += float(dm.sum())
                    elif modulation < 0:
                        self.audit['negative_updates'] += 1
                        self.audit['negative_delta'] += float(-dm.sum())
                self.audit['unsupervised_delta'] += float(du.sum())
                w += du
                if dm is not None:
                    w += dm
                w *= self.cfg['weight_sum']/w.sum()
                np.clip(w, 0, 1, out=w)
                self.usage[winner] += 1
        return self

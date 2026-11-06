"""Causal minute-domain input/teacher spikes and bounded pair STDP.

No future-event array is accepted by score(). Training teacher spikes are
realized price events, not free-running hidden neuron responses.
"""
import numpy as np


def lagged(x, segments, lag):
    x = np.asarray(x)
    out = np.zeros_like(x)
    n = len(x)
    if lag >= n or -lag >= n:
        return out
    if lag > 0:
        valid = segments[lag:] == segments[:-lag]
        out[lag:][valid] = x[:-lag][valid]
    elif lag < 0:
        d = -lag
        valid = segments[:-d] == segments[d:]
        out[:-d][valid] = x[d:][valid]
    else:
        out[:] = x
    return out


def trace(x, segments, lags, tau):
    return sum(np.exp(-abs(d)/tau)*lagged(x, segments, d) for d in lags)


def encode(features, segments, bounds):
    """Low/high threshold crossings, interleaved per feature; causal."""
    x = np.asarray(features, dtype=float)
    previous = lagged(x, segments, 1)
    continuous = lagged(np.ones(len(x), dtype=bool), segments, 1)
    valid = np.isfinite(x) & np.isfinite(previous) & continuous[:, None]
    low = valid & (x < bounds[0]) & (previous >= bounds[0])
    high = valid & (x > bounds[1]) & (previous <= bounds[1])
    return np.stack([low, high], axis=2).reshape(len(x), -1).astype(float)


def price_events(close, segments, threshold, quiet_minutes=5):
    previous = lagged(np.asarray(close, dtype=float), segments, 1)
    returns = np.full(len(close), np.nan)
    valid = previous > 0
    returns[valid] = np.log(np.asarray(close)[valid]/previous[valid])*10000
    raw = np.abs(returns) > threshold
    prior_count = sum(lagged(raw.astype(int), segments, d) for d in range(1, quiet_minutes+1))
    full_prior = lagged(np.ones(len(raw), dtype=bool), segments, quiet_minutes+1)
    onset = raw & (prior_count == 0) & full_prior
    return returns, raw, onset


def samples(features, segments, returns, raw_events, onset, horizon=5):
    finite = np.isfinite(features).all(axis=1)
    # Include one extra predecessor so every crossing in the input window
    # has an observed predecessor. No features from future minutes checked.
    past_ok = finite.copy()
    for d in range(1, horizon+1):
        past_ok &= lagged(finite, segments, d)
    future_ok = lagged(np.ones(len(finite), dtype=bool), segments, -horizon)
    quiet = np.ones(len(finite), dtype=bool)
    for d in range(horizon):
        quiet &= ~lagged(raw_events, segments, d)
        quiet &= lagged(np.isfinite(returns), segments, d)
    y = np.zeros(len(finite), dtype=bool)
    for d in range(1, horizon+1):
        y |= lagged(onset, segments, -d)
    return past_ok & future_ok & quiet, y.astype(float)


def fit_stdp(spikes, teacher, segments, train_mask, cfg):
    horizon, tau = cfg['horizon'], cfg['tau_minutes']
    pre = trace(spikes, segments, range(1, horizon+1), tau)
    post = trace(teacher.astype(float), segments, range(1, horizon+1), tau)
    w = np.full(spikes.shape[1], cfg['initial_weight'], dtype=float)
    positive = np.zeros_like(w)
    negative = np.zeros_like(w)
    indices = np.flatnonzero(train_mask)
    for _ in range(cfg['epochs']):
        for t in indices:
            plus = cfg['eta_plus']*(1-w)*teacher[t]*pre[t]
            minus = cfg['eta_minus']*w*spikes[t]*post[t]
            w = np.clip(w + plus - minus, 0, 1)
            positive += plus
            negative += minus
    return w, {'ltp': positive, 'ltd': negative,
               'weighted_lead_pairs': (teacher[indices, None]*pre[indices]).sum(axis=0),
               'weighted_lag_pairs': (spikes[indices]*post[indices, None]).sum(axis=0)}


def score(spikes, segments, weights, cfg):
    """Read membrane after a fresh finite integration window at each minute."""
    currents = trace(spikes, segments, range(cfg['horizon']), cfg['tau_minutes'])
    return currents @ weights, currents


def shuffled_teacher(teacher, segments, train_mask, seed):
    rng = np.random.default_rng(seed)
    result = np.zeros(len(teacher), dtype=bool)
    for sid in np.unique(segments[train_mask]):
        idx = np.flatnonzero((segments == sid) & train_mask)
        if len(idx) < 21:
            raise ValueError('Need at least 21 minutes for teacher shift control')
        shift = int(rng.integers(10, len(idx)-9))
        result[idx] = np.roll(teacher[idx], shift)
    return result


def ranking(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if not len(y):
        return {'auc': np.nan, 'ap': np.nan}
    order = np.argsort(-score, kind='stable')
    yy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(y)-1]
    tp = np.cumsum(yy)[ends].astype(float)
    fp = (ends+1)-tp
    positives, negatives = y.sum(), len(y)-y.sum()
    ap = np.sum(np.diff(np.r_[0, tp]) * tp/(ends+1))/positives if positives else np.nan
    # Trapezoidal ROC integration handles score ties without arbitrary order.
    auc = np.trapezoid(np.r_[0, tp/positives], np.r_[0, fp/negatives]) if positives and negatives else np.nan
    return {'auc': float(auc), 'ap': float(ap)}


def classification(y, score, alarm):
    y = np.asarray(y, dtype=bool)
    alarm = np.asarray(alarm, dtype=bool)
    hits = int((y & alarm).sum())
    n, pos, alerts = len(y), int(y.sum()), int(alarm.sum())
    return dict(n=n, positives=pos, base_rate=pos/n if n else np.nan,
                alerts=alerts, coverage=alerts/n if n else np.nan, hits=hits,
                precision=hits/alerts if alerts else np.nan,
                recall=hits/pos if pos else np.nan,
                **ranking(y, score))


def fit_ridge(x, y, train, validation, cfg):
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale = np.maximum(scale, 1e-8)
    z = (x-mean)/scale
    target_mean = y[train].mean()
    xx = z[train].T @ z[train]
    xy = z[train].T @ (y[train]-target_mean)
    candidates = []
    for alpha in cfg['ridge_alphas']:
        coef = np.linalg.solve(xx + alpha*np.eye(x.shape[1]), xy)
        prediction = z @ coef + target_mean
        brier = np.mean((np.clip(prediction[validation], 0, 1)-y[validation])**2)
        candidates.append((float(brier), float(alpha), coef, prediction))
    best = min(candidates, key=lambda a: (a[0], a[1]))
    return best[3], dict(mean=mean, scale=scale, coef=best[2], target_mean=target_mean,
                         alpha=best[1], validation_brier=best[0])

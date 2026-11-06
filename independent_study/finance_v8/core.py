import numpy as np
import pandas as pd
from event_stdp_v7.core import lagged, trace


def delayed_history(spikes, features, segments, delays):
    if min(delays) < 0:
        raise ValueError('No future delays')
    x = np.stack([lagged(spikes, segments, d) for d in delays], axis=2).reshape(len(spikes), -1)
    finite = np.isfinite(features).all(axis=1)
    valid = finite.copy()
    for d in range(1, max(delays)+2):
        valid &= lagged(finite, segments, d)
    return x, valid


def hidden_threshold(weight, train_arrivals, quantile=.9):
    drives = train_arrivals @ weight.T
    return np.maximum(np.quantile(drives, quantile, axis=0), weight.max(axis=1)+1e-6)


def fit_autonomous(hidden, segments, train_mask, cfg):
    """Autonomous IF post spikes; no price event/target argument exists."""
    w = np.full(hidden.shape[1], cfg['initial_weight'], dtype=float)
    theta = max(float(np.quantile(hidden[train_mask]@w, .8)), 1e-6)
    initial_theta = theta
    pre_trace = trace(hidden, segments, range(1, cfg['horizon']+1), cfg['tau_minutes'])
    kernel = np.exp(-np.arange(1,cfg['horizon']+1)/cfg['tau_minutes'])
    counts = []
    indices = np.flatnonzero(train_mask)
    for _ in range(cfg['epochs']):
        history = np.zeros(cfg['horizon']); previous = -2; last_segment = None; fired = 0
        for t in indices:
            if t != previous+1 or segments[t] != last_segment:
                history[:] = 0
            post = float(hidden[t]@w > theta)
            delta = cfg['eta_plus']*(1-w)*post*pre_trace[t] - cfg['eta_minus']*w*hidden[t]*(history@kernel)
            w = np.clip(w+delta, 0, 1)
            theta = max(1e-6, theta+cfg['homeostasis_rate']*(post-cfg['target_firing']))
            history[1:] = history[:-1]; history[0] = post
            previous, last_segment = t, segments[t]
            fired += int(post)
        counts.append(fired/len(indices))
    return w, dict(initial_theta=initial_theta, final_theta=theta, firing_by_epoch=counts)


def mid_percentile(values, reference):
    ref = np.sort(np.asarray(reference))
    if not len(ref) or not np.isfinite(ref).all():
        raise ValueError('Invalid past calibration reference')
    return (np.searchsorted(ref, values, side='left')+np.searchsorted(ref, values, side='right'))/(2*len(ref))


def quantities(values, covered, q60, q80, application):
    values = np.asarray(values, dtype=float)
    covered = np.asarray(covered, dtype=bool)
    if not np.isfinite(values[covered]).all():
        raise ValueError('Invalid covered forecast')
    high = covered & (values > q80)
    if application == 'filter':
        return (~high).astype(int)
    if application == 'timing':
        return high.astype(int)
    if application == 'sizing':
        result = np.full(len(values), 2, dtype=int)
        result[covered & (values > q60)] = 1
        result[high] = 0
        return result
    raise ValueError(application)


def shifted_opportunities(opportunities, bars, delay):
    out = opportunities.copy()
    out['entry_bar_id'] += delay
    out['exit_bar_id_actual'] += delay
    out['exit_bar_id'] += delay
    lookup = bars.set_index('bar_id', drop=False)
    en, ex = out.entry_bar_id.to_numpy(), out.exit_bar_id_actual.to_numpy()
    if not np.isin(en, lookup.index).all() or not np.isin(ex, lookup.index).all():
        raise ValueError('Missing delayed entry/exit')
    signal_segments = lookup.loc[out.signal_bar_id].segment_id.to_numpy()
    if not np.array_equal(signal_segments, lookup.loc[en].segment_id) or not np.array_equal(signal_segments, lookup.loc[ex].segment_id):
        raise ValueError('Delayed execution would cross a segment boundary')
    if (lookup.loc[en].volume.to_numpy() <= 0).any() or (lookup.loc[ex].volume.to_numpy() <= 0).any():
        raise ValueError('Delayed execution has zero-volume bars; explicit fill handling required')
    out['entry_open'] = lookup.loc[en].open.to_numpy()
    out['exit_open'] = lookup.loc[ex].open.to_numpy()
    out['entry_time'] = lookup.loc[en].datetime.to_numpy()-pd.Timedelta(minutes=1)
    out['exit_time'] = lookup.loc[ex].datetime.to_numpy()-pd.Timedelta(minutes=1)
    assert (pd.to_datetime(out.entry_time) > pd.to_datetime(out.signal_time)).all()
    return out

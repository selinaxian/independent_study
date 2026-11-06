import json
from pathlib import Path
import numpy as np
from event_stdp_v7.core import (encode, price_events, samples, fit_stdp, score,
                               shuffled_teacher, ranking)

CFG = json.loads((Path(__file__).parents[1]/'config.json').read_text())


def synthetic():
    # 30-minute episodes: same number of spikes on all four channels.
    segments = np.repeat(np.arange(80), 30)
    spikes = np.zeros((len(segments), 4))
    teacher = np.zeros(len(segments), dtype=bool)
    for k in range(80):
        base = 30*k
        teacher[base+15] = True
        for channel, offset in enumerate([13, 17, 15, 4]):
            spikes[base+offset, channel] = 1
    return spikes, teacher, segments


def test_leading_lagging_simultaneous_and_unpaired():
    s, e, seg = synthetic()
    w, stats = fit_stdp(s, e, seg, seg < 60, CFG)
    assert w[0] > .79 and w[1] < .21
    np.testing.assert_array_equal(w[2:], [.5, .5])
    np.testing.assert_allclose(.5+stats['ltp']-stats['ltd'], w)
    pred, _ = score(s, seg, w, CFG)
    assert pred[30*70+13] > pred[30*70+17]  # equal spike count, different order


def test_exact_minute_decay_and_zero_lag_exclusion():
    s = np.eye(7)
    teacher = np.zeros(7, dtype=bool); teacher[6] = True
    cfg = dict(CFG, epochs=1)
    w, _ = fit_stdp(s, teacher, np.ones(7), np.ones(7, bool), cfg)
    assert w[0] == .5 and w[6] == .5
    np.testing.assert_allclose(w[1:6], .5+.005*np.exp(-np.arange(5, 0, -1)/3))


def test_no_cross_session_pairing():
    s = np.array([[1.], [0.]])
    e = np.array([False, True])
    w, _ = fit_stdp(s, e, np.array([1, 2]), np.ones(2, bool), CFG)
    assert w[0] == .5


def test_encoding_is_crossing_not_persistent_state():
    x = np.array([[0.], [2.], [2.], [-2.], [-2.], [2.]])
    out = encode(x, np.array([1, 1, 1, 1, 1, 2]), np.array([[-1.], [1.]]))
    np.testing.assert_array_equal(out, [[0,0], [0,1], [0,0], [1,0], [0,0], [0,0]])


def test_prefix_invariance_and_teacher_cannot_change_old_predictions():
    s, e, seg = synthetic()
    train = seg < 30
    w, _ = fit_stdp(s, e, seg, train, CFG)
    p, _ = score(s, seg, w, CFG)
    e[1000:] = ~e[1000:]
    w2, _ = fit_stdp(s, e, seg, train, CFG)
    np.testing.assert_array_equal(w, w2)
    s[1100:] = 1-s[1100:]
    p2, _ = score(s, seg, w, CFG)
    np.testing.assert_array_equal(p[:1100], p2[:1100])
    short, _ = score(s[:1000], seg[:1000], w, CFG)
    np.testing.assert_allclose(p[:1000], short)


def test_event_onset_quiet_and_strict_future_labels():
    changes = np.zeros(35); changes[[12, 14, 24]] = .01
    close = 100*np.exp(np.cumsum(changes))
    seg = np.ones(35, dtype=int)
    r, raw, onset = price_events(close, seg, 50)
    np.testing.assert_array_equal(np.flatnonzero(onset), [12, 24])
    valid, y = samples(np.ones((35, 2)), seg, r, raw, onset)
    assert y[11] == 1 and y[12] == 0 and not valid[12]
    assert not valid[14] and valid[11] and not valid[-1]
    changed = close.copy(); changed[20:] *= 2
    r2, raw2, onset2 = price_events(changed, seg, 50)
    np.testing.assert_array_equal(onset[:20], onset2[:20])


def test_shuffle_only_training_counts_preserved():
    s, e, seg = synthetic()
    out = shuffled_teacher(e, seg, seg < 40, 104)
    for sid in range(40):
        assert out[seg == sid].sum() == e[seg == sid].sum()
    assert not out[seg >= 40].any()


def test_ranking_ties_and_perfect_order():
    assert ranking([1, 0, 1, 0], [0, 0, 0, 0]) == {'auc': .5, 'ap': .5}
    assert ranking([1, 0, 1, 0], [2, 0, 3, 1]) == {'auc': 1., 'ap': 1.}

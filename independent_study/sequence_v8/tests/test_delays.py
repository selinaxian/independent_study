import numpy as np
import pytest
from sequence_v8.core import delayed_inputs, CompetitivePatterns, generate, output_training_arrays

CFG = dict(hidden=8, weight_sum=2., epochs=3, usage_decay=.995,
           usage_penalty=.15, eta=.15, threshold=1.3)


def test_real_delay_coincidence_and_no_future_access():
    seq = np.zeros((1, 12, 3)); seq[0, 3, 0] = 1; seq[0, 5, 1] = 1
    x = delayed_inputs(seq, [5], range(5)).reshape(3,5)
    assert x[0,2] == x[1,0] == 1 and x.sum() == 2
    seq[0, 6:] = 1
    np.testing.assert_array_equal(x, delayed_inputs(seq, [5], range(5)).reshape(3,5))
    with pytest.raises(ValueError):
        delayed_inputs(seq, [5], [-1, 0])


def test_counts_cannot_distinguish_order_or_gap():
    data = generate(3, 104)
    counts = data['spikes'].sum(axis=1)
    hard = data['kinds'] < 5
    assert np.unique(counts[hard], axis=0).shape[0] == 1
    delayed = delayed_inputs(data['spikes'], data['anchors'], range(5))
    assert np.unique(delayed[hard], axis=0).shape[0] == 5


def test_single_arrival_cannot_trigger_inference_unit():
    model = CompetitivePatterns(12, CFG, 104)
    model.weight[:] = 1
    spikes, potential, reset = model.response(np.eye(12))
    assert not spikes.any() and (potential == 1).all()
    spikes, potential, reset = model.response(np.ones((1,12)))
    assert spikes.all() and not reset.any()


def test_hidden_learning_does_not_use_price_events():
    data = generate(20, 7)
    x = delayed_inputs(data['spikes'], data['anchors'], range(5))
    m1 = CompetitivePatterns(x.shape[1], CFG, 8).fit(x, 9)
    data['events'][:] = ~data['events']
    m2 = CompetitivePatterns(x.shape[1], CFG, 8).fit(x, 9)
    np.testing.assert_array_equal(m1.weight, m2.weight)
    assert np.linalg.norm(m1.weight-m1.initial_weight) > .1
    assert ((m1.weight >= 0) & (m1.weight <= 1)).all()
    assert (m1.weight.sum(axis=1) <= 2+1e-12).all()


def test_teacher_is_after_target_and_before_reverse():
    data = generate(2, 13)
    hidden = np.ones((len(data['target']), 8))
    spikes, event, segment = output_training_arrays(hidden, data)
    assert spikes.sum() == len(data['target'])*8
    for row, kind in enumerate(data['kinds']):
        event_times = np.flatnonzero(data['events'][row])
        if kind == 0:
            assert event_times[0] == data['anchors'][row]+1
        elif kind == 3:
            assert event_times[0] == data['anchors'][row]-2
        else:
            assert not len(event_times)

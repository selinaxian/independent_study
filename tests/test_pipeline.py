"""Focused scientific correctness checks: causality, boundaries and plasticity."""

import numpy as np
import pandas as pd
import pytest
import torch

from snn_quant.evaluation import evaluate, select_ridge
from snn_quant.features import (FEATURE_NAMES, attach_segments, build_features,
                                build_windows, fit_scales, normalize)
from snn_quant.network import CompetitiveSTDP, NetworkConfig, signed_channels, stdp_delta


@pytest.fixture
def bars():
    times = pd.DatetimeIndex([])
    for day in ("2026-01-05", "2026-01-06", "2026-01-07"):
        for start in ("09:31", "13:01"):
            times = times.append(pd.date_range(f"{day} {start}", periods=120, freq="min"))
    idx = np.arange(len(times))
    close = 6000 + np.cumsum(np.sin(idx / 2.7) + 0.2)
    opening = close - np.cos(idx / 3)
    frame = pd.DataFrame({"datetime": times, "open": opening,
                          "high": np.maximum(opening, close) + 0.8,
                          "low": np.minimum(opening, close) - 0.7,
                          "close": close, "volume": 100 + idx % 19,
                          "open_interest": 100000 + 10 * np.sin(idx / 4)})
    return attach_segments(frame)


def test_future_changes_cannot_change_past_features(bars):
    original = build_features(bars)
    changed = bars.copy()
    changed.loc[60:, ["open", "high", "low", "close"]] *= 2
    changed.loc[60:, ["volume", "open_interest"]] *= 3
    perturbed = build_features(changed)
    np.testing.assert_allclose(original.loc[:59, FEATURE_NAMES], perturbed.loc[:59, FEATURE_NAMES], equal_nan=True)
    assert not np.isclose(original.loc[58, "future_return_bps"], perturbed.loc[58, "future_return_bps"])


def test_reference_levels_exclude_current_bar(bars):
    original = build_features(bars)
    changed = bars.copy()
    changed.loc[45, "high"] += 500
    changed.loc[45, "low"] -= 500
    current = build_features(changed)
    assert current.loc[45, "reference_high"] == original.loc[45, "reference_high"]
    assert current.loc[45, "reference_low"] == original.loc[45, "reference_low"]
    assert current.loc[46, "reference_high"] != original.loc[46, "reference_high"]


def test_window_history_and_targets_stay_inside_sessions(bars):
    features = build_features(bars)
    data = build_windows(features, 10, "2026-01-05", "2026-01-06")
    assert len(data.x) == 6 * 85
    assert data.x.shape[1:] == (10, 10)
    assert set(data.metadata.split) == {"train", "validation", "test"}
    for _, segment in features.groupby("segment_id"):
        meta = data.metadata[data.metadata.segment_id == segment.segment_id.iloc[0]]
        assert meta.datetime.min() == segment.datetime.iloc[30]
        assert meta.target_datetime.max() == segment.datetime.iloc[-1]
        assert (meta.target_datetime - meta.datetime).eq(pd.Timedelta(minutes=5)).all()


def test_known_contract_change_and_gap_reset_histories(bars):
    changed = bars.iloc[:120].copy()
    changed["contract_id"] = np.where(np.arange(120) < 60, "IC2601", "IC2602")
    changed = attach_segments(changed.drop(index=95).reset_index(drop=True))
    assert changed.segment_id.nunique() == 3
    features = build_features(changed)
    for _, segment in features.groupby("segment_id"):
        assert not segment.feature_valid.iloc[:21].any()


def test_normalization_fits_only_unique_training_minutes(bars):
    features = build_features(bars)
    original_scale = fit_scales(features, "2026-01-05")
    features.loc[features.trading_date > "2026-01-05", FEATURE_NAMES] *= 1000
    np.testing.assert_array_equal(original_scale, fit_scales(features, "2026-01-05"))


def test_zero_divisions_never_make_valid_infinite_features(bars):
    flat = bars.copy()
    flat[["open", "high", "low", "close"]] = 6000.0
    flat[["volume", "open_interest"]] = 0.0
    features = build_features(flat)
    assert not np.isinf(features[FEATURE_NAMES].to_numpy()).any()
    assert not features.feature_valid.any()


def test_signed_encoding_preserves_sign_and_exclusivity():
    x = torch.tensor([[[0.7, -0.2, 0.0]]])
    encoded = signed_channels(x)
    torch.testing.assert_close(encoded, torch.tensor([[[0.7, 0.0, 0.0, 0.2, 0.0, 0.0]]]))


def test_stdp_respects_order_and_excludes_simultaneous_pairs():
    cfg = NetworkConfig()
    w = torch.tensor([[0.5]])
    z, one = torch.zeros((1, 1)), torch.ones((1, 1))
    _, pre, post = stdp_delta(w, z, z, one, z, cfg)
    positive, _, _ = stdp_delta(w, pre, post, z, one, cfg)
    _, pre, post = stdp_delta(w, z, z, z, one, cfg)
    negative, _, _ = stdp_delta(w, pre, post, one, z, cfg)
    simultaneous, _, _ = stdp_delta(w, z, z, one, one, cfg)
    assert positive.item() > 0
    assert negative.item() < 0
    assert simultaneous.item() == 0
    # Compare to the expected single earlier-spike decay, not an implementation copy.
    assert positive.item() == pytest.approx(cfg.eta_plus * 0.5 * (1 - 1 / cfg.tau_pre))


def test_stdp_batch_size_does_not_change_update_scale():
    cfg = NetworkConfig()
    w, pre = torch.tensor([[0.4]]), torch.tensor([[0.7]])
    post, x, y = torch.tensor([[0.2]]), torch.tensor([[1.0]]), torch.tensor([[1.0]])
    small = stdp_delta(w, pre, post, x, y, cfg)[0]
    large = stdp_delta(w, pre.repeat(7, 1), post.repeat(7, 1), x.repeat(7, 1), y.repeat(7, 1), cfg)[0]
    torch.testing.assert_close(small, large)


def test_training_changes_synapses_but_inference_is_frozen_and_reproducible(tmp_path):
    torch.set_num_threads(1)
    cfg = NetworkConfig(hidden=8, epochs=1, steps_per_minute=4, batch_size=8,
                        threshold=0.3, max_spike_probability=0.6, weight_sum=1.5)
    windows = np.random.default_rng(17).uniform(-1, 1, size=(32, 6, 4)).astype(np.float32)
    net = CompetitiveSTDP(4, cfg, 17)
    initial = net.weight.clone()
    history = net.fit(windows)
    assert not torch.equal(initial, net.weight)
    assert history[-1]["mean_spikes_per_sample"] > 0
    assert (net.weight >= 0).all() and (net.weight <= cfg.weight_max).all()
    state = net.state_dict()
    prediction = net.transform(windows, 123)
    np.testing.assert_array_equal(prediction, net.transform(windows, 123))
    for name in state:
        assert torch.equal(state[name], net.state_dict()[name])
    path = tmp_path / "model.pt"
    net.save(path, ["a", "b", "c", "d"])
    reloaded = CompetitiveSTDP(4, cfg, 999)
    reloaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    np.testing.assert_array_equal(prediction, reloaded.transform(windows, 123))
    control = CompetitiveSTDP(4, cfg, 17)
    control.fit(windows, learn_weights=False)
    assert torch.equal(control.weight, initial)


def test_ridge_scaling_and_selection_use_training_and_validation_only():
    rng = np.random.default_rng(25)
    x = rng.normal(size=(100, 4))
    y = 2 * x[:, 0] - x[:, 1]
    model, candidates = select_ridge(x[:70], y[:70], x[70:90], y[70:90], [0.0001, 0.01, 1.0])
    np.testing.assert_allclose(model.mean, x[:70].mean(axis=0))
    assert model.alpha == min(candidates, key=lambda c: c["validation_mse_bps2"])["alpha"]
    assert np.mean((model.predict(x[90:]) - y[90:]) ** 2) < 0.01


def test_constant_prediction_has_undefined_ic():
    scores = evaluate(np.array([-1, 0, 1, 2.0]), np.ones(4), ["2026-01-05"] * 4)
    assert scores["pearson_ic"] is None
    assert scores["direction_accuracy"] == pytest.approx(2 / 3)

"""Shape, causality and arithmetic checks on synthetic nights."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import sosfilt

from arcasleep_watchtree.data.cohorts import four_class
from arcasleep_watchtree.features import constants as C
from arcasleep_watchtree.features.baseline import low_baseline
from arcasleep_watchtree.features.breathing import goertzel_power
from arcasleep_watchtree.features.preprocess import biquads, nominal_rate, to_25hz
from arcasleep_watchtree.features.table import (
    EPOCHS_PER_MIN,
    bout_timing,
    compute_features,
    feature_names,
    stream_costs,
    window_costs,
)
from arcasleep_watchtree.features.windows import RingWindow, window_stat

FORWARD_EPOCHS = C.FORWARD_MAX_MIN * EPOCHS_PER_MIN


def synthetic_night(n_epochs: int, seed: int = 0, hr: bool = True):
    """A still wrist with 0.25 Hz breathing, bursts of movement, and a heart rate near 60."""
    rng = np.random.default_rng(seed)
    n = n_epochs * C.SAMPLES_PER_EPOCH
    t = np.arange(n) / C.FS
    acc = np.array([0.1, 0.2, 0.97]) + rng.normal(0, 0.004, (n, 3))
    acc += 0.003 * np.sin(2 * np.pi * 0.25 * t)[:, None]
    for start in rng.integers(0, n - 2000, size=n_epochs // 40):
        acc[start : start + 600] += rng.normal(0, 0.3, (600, 3))
    hr_t = np.arange(n_epochs * C.EPOCH_S, dtype=float)
    hr_bpm = 60 + 3 * np.sin(hr_t / 900) + rng.normal(0, 1, hr_t.size)
    if not hr:
        hr_t, hr_bpm = np.empty(0), np.empty(0)
    return acc, np.ones(n, bool), hr_t, hr_bpm


def test_table_has_one_row_per_epoch_and_named_features():
    n_epochs = 120
    table = compute_features(*synthetic_night(n_epochs))
    assert len(table) == n_epochs
    assert list(table.columns) == ["epoch", *feature_names()]
    assert len(set(feature_names())) == len(feature_names())
    assert 60 <= len(feature_names()) <= 150
    assert not np.isinf(table[feature_names()].to_numpy()).any()


def test_feature_does_not_change_when_data_after_the_forward_reach_changes():
    n_epochs, k = 400, 150
    acc, valid, hr_t, hr_bpm = synthetic_night(n_epochs)
    before = compute_features(acc, valid, hr_t, hr_bpm)

    # Epoch k reaches to the end of epoch k + 90; leave 5 s of margin for the resampler-free path.
    cut = (k + 1 + FORWARD_EPOCHS) * C.SAMPLES_PER_EPOCH + 5 * C.FS
    rng = np.random.default_rng(1)
    acc2 = acc.copy()
    acc2[cut:] = np.array([0.5, -0.3, 0.8]) + rng.normal(0, 0.2, acc2[cut:].shape)
    hr2 = hr_bpm.copy()
    hr2[hr_t >= cut / C.FS] += 25.0
    after = compute_features(acc2, valid, hr_t, hr2)

    names = feature_names()
    np.testing.assert_array_equal(before.loc[:k, names].to_numpy(), after.loc[:k, names].to_numpy())
    assert not np.array_equal(
        before.loc[k + FORWARD_EPOCHS :, names].to_numpy(),
        after.loc[k + FORWARD_EPOCHS :, names].to_numpy(),
    )


def test_missing_heart_rate_is_flagged_not_filled():
    table = compute_features(*synthetic_night(100, hr=False))
    assert (table["hr_missing"] == 1).all()
    assert table["hr"].isna().all() and table["hr_rel_mean_b15"].isna().all()
    assert table["jerk"].notna().all()

    acc, valid, hr_t, hr_bpm = synthetic_night(100)
    keep = (hr_t < 1500) | (hr_t >= 1800)
    gapped = compute_features(acc, valid, hr_t[keep], hr_bpm[keep])
    assert (gapped.loc[50:59, "hr_missing"] == 1).all() and gapped.loc[50:59, "hr"].isna().all()
    assert gapped.loc[10, "hr_missing"] == 0


def test_invalid_epochs_have_no_movement_features():
    acc, valid, hr_t, hr_bpm = synthetic_night(60)
    valid[20 * C.SAMPLES_PER_EPOCH : 22 * C.SAMPLES_PER_EPOCH] = False
    table = compute_features(acc, valid, hr_t, hr_bpm)
    assert table.loc[20:21, ["jerk", "lfm", "angle", "breath_bpm"]].isna().all().all()
    assert table.loc[22, "jerk"] > 0


def test_time_since_last_bout_is_capped_and_ignores_recording_start():
    quiet = np.zeros(1000)
    since, until = bout_timing(quiet)
    assert (since == C.TSB_CAP_MIN).all() and (until == C.FORWARD_MAX_MIN).all()

    act = np.zeros(1000)
    act[100] = 1
    since, until = bout_timing(act)
    assert since[99] == C.TSB_CAP_MIN and since[100] == 0 and since[110] == 5
    assert since[999] == C.TSB_CAP_MIN
    assert until[99] == 0.5 and until[10] == C.FORWARD_MAX_MIN and until[100] == C.FORWARD_MAX_MIN


@pytest.mark.parametrize("stat", ["mean", "std", "max"])
@pytest.mark.parametrize("n", [2, 10, 60])
def test_windows_match_the_ring_buffer_and_forward_is_a_delayed_backward(stat, n):
    rng = np.random.default_rng(3)
    values = rng.normal(size=200)
    values[rng.integers(0, 200, 15)] = np.nan
    ring, streamed = RingWindow(n), []
    for v in values:
        ring.push(v)
        streamed.append(getattr(ring, stat)())
    back = window_stat(values, stat, "b", n)
    np.testing.assert_allclose(back, streamed, atol=1e-9, equal_nan=True)
    forward = window_stat(values, stat, "f", n)
    np.testing.assert_allclose(forward[: 200 - n], back[n:], atol=1e-9, equal_nan=True)


def test_biquad_cascade_equals_the_direct_form_recursion():
    rng = np.random.default_rng(4)
    x = 1.0 + rng.normal(size=(500, 3))
    state = np.zeros((C.SOS_1.shape[0], 2, 3))
    from scipy.signal import sosfilt_zi

    state[:] = sosfilt_zi(C.SOS_1)[:, :, None] * x[0]
    out = np.empty_like(x)
    for n in range(x.shape[0]):
        v = x[n].copy()
        for s, (b0, b1, b2, _, a1, a2) in enumerate(C.SOS_1):
            y = b0 * v + state[s, 0]
            state[s, 0] = b1 * v - a1 * y + state[s, 1]
            state[s, 1] = b2 * v - a2 * y
            v = y
        out[n] = v
    np.testing.assert_allclose(biquads(C.SOS_1, x), out, atol=1e-12)
    assert abs(biquads(C.SOS_1, np.ones((200, 3)))[-1, 0] - 1.0) < 1e-9
    assert sosfilt(C.SOS_1, np.ones(10)).shape == (10,)


def test_goertzel_bins_equal_the_dft_power():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, C.BREATH_SAMPLES, 3))
    expected = (np.abs(np.fft.fft(x, axis=1)[:, C.BREATH_BINS, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(goertzel_power(x), expected, rtol=1e-8)


def test_breathing_rate_of_a_still_synthetic_wrist_is_read_from_the_peak():
    table = compute_features(*synthetic_night(60))
    assert abs(table["breath_bpm"].median() - 15.0) < 1.0


def test_low_baseline_follows_the_lowest_blocks():
    series = np.concatenate([np.full(240, 50.0), np.full(20, 90.0), np.full(20, 52.0)])
    base = low_baseline(series)
    assert base[-1] == pytest.approx(50.0, abs=1e-6)
    assert base[0] == 50.0


def test_resampling_a_jittered_stream_keeps_a_slow_sine_and_marks_gaps():
    rng = np.random.default_rng(6)
    t = np.arange(0, 120, 0.02) + rng.normal(0, 0.0005, 6000)
    t = np.sort(t)
    keep = (t < 40) | (t > 45)
    t = t[keep]
    sine = 0.01 * np.sin(2 * np.pi * 0.3 * t)
    xyz = np.column_stack([sine, np.zeros_like(t), np.ones_like(t)])
    assert nominal_rate(t) == 50
    out, valid = to_25hz(t, xyz, 0.0, 4)
    grid = np.arange(out.shape[0]) / C.FS
    inside = (grid > 5) & (grid < 35)
    np.testing.assert_allclose(
        out[inside, 0], 0.01 * np.sin(2 * np.pi * 0.3 * grid[inside]), atol=2e-4
    )
    assert not valid[(grid > 41) & (grid < 44)].any() and valid[(grid > 0.5) & (grid < 39)].all()


def test_empatica_rate_stream_reaches_25hz():
    t = np.arange(0, 90, 1 / 32)
    out, valid = to_25hz(
        t, np.column_stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]), 0.0, 3
    )
    assert out.shape == (3 * C.SAMPLES_PER_EPOCH, 3)
    assert abs(out[1000, 2] - 1.0) < 1e-3


def test_four_class_collapses_n1_n2_and_n3_n4():
    raw = np.array([0, 1, 2, 3, 4, 5, -1, -2])
    np.testing.assert_array_equal(four_class(raw), [0, 1, 1, 2, 2, 3, -1, -1])


def test_every_feature_has_a_cost_and_the_total_is_within_the_watch_budget():
    lines = stream_costs() + window_costs()
    assert all(line.ops_per_epoch > 0 for line in lines)
    assert sum(line.ops_per_epoch for line in lines) < 23640 * C.EPOCH_S
    assert len(window_costs()) + 20 == len(feature_names())

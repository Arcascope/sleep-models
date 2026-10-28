"""Weak labels outside the PSG, the window sampler and the overlapping-window average."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from arcasleep.data.constants import GAP_LABEL, OUTSIDE_PSG_LABEL  # noqa: E402
from arcasleep.data.weak_labels import (  # noqa: E402
    SLEEP,
    UNLABELED,
    WAKE,
    WEAK_LABEL_RULES,
    WeakLabelConfig,
    WeakLabels,
    label_recording,
)
from arcasleep.predict import (  # noqa: E402
    overlapping_window_logits,
    window_starts,
    window_taper,
)
from arcasleep.training.balanced import (  # noqa: E402
    balanced_cross_entropy,
    weighted_balanced_cross_entropy,
)
from arcasleep.training.windows import WindowMix, WindowRecording, WindowSampler  # noqa: E402

BOTH = ["daytime_wake", "daytime_sleep"]


def anglez_grid(epochs: int, still: slice) -> np.ndarray:
    """A z-angle series that moves everywhere except the ``still`` epochs."""
    rng = np.random.default_rng(0)
    grid = np.cumsum(rng.normal(0, 8.0, size=(epochs, 6)), axis=1) % 60.0
    grid[still] = 10.0
    return grid


def labels_with_psg(epochs: int, first: int, last: int) -> np.ndarray:
    y = np.full(epochs, OUTSIDE_PSG_LABEL, dtype=int)
    y[first:last] = np.tile([0, 1, 2, 3], (last - first) // 4 + 1)[: last - first]
    return y


def test_weak_labels_never_touch_the_psg_span() -> None:
    y = labels_with_psg(2400, 900, 1300)
    grid = anglez_grid(2400, slice(800, 1500))
    plain = label_recording(y, None)
    weak = label_recording(y, grid, BOTH)
    inside = np.zeros(2400, dtype=bool)
    inside[900:1300] = True
    assert np.array_equal(weak.labels[inside], plain.labels[inside])
    assert (weak.weights[inside] == 1).all()
    assert not weak.sleep_wake[inside].any()
    assert (plain.labels[~inside] == UNLABELED).all()
    assert weak.in_psg.sum() == 400


def test_wake_and_sleep_rules_are_exclusive_and_only_label_what_they_say() -> None:
    y = labels_with_psg(2400, 900, 1300)
    grid = anglez_grid(2400, slice(800, 1500))
    weights = WeakLabelConfig(strong=1.0, weak=0.3, sleep=0.25)
    wake = WEAK_LABEL_RULES["daytime_wake"](grid, weights)
    sleep = WEAK_LABEL_RULES["daytime_sleep"](grid, weights)
    assert not ((wake.labels == WAKE) & (sleep.labels == SLEEP)).any()
    assert (sleep.labels[sleep.sleep_wake] == SLEEP).all()
    assert set(np.unique(sleep.weights[sleep.sleep_wake])) <= {np.float32(0.25)}
    assert not wake.sleep_wake.any()
    assert (sleep.labels == SLEEP).any() and (wake.labels == WAKE).any()
    combined = label_recording(y, grid, BOTH, weights)
    assert combined.sleep_wake.sum() == int(sleep.sleep_wake[~combined.in_psg].sum())


def test_min_sib_bout_minutes_zero_labels_at_least_as_much_sleep() -> None:
    grid = anglez_grid(2400, slice(800, 1500))
    short = WEAK_LABEL_RULES["daytime_sleep"](grid, WeakLabelConfig(min_sib_bout_minutes=20.0))
    every = WEAK_LABEL_RULES["daytime_sleep"](grid, WeakLabelConfig(min_sib_bout_minutes=0.0))
    assert (every.sleep_wake | ~short.sleep_wake).all()
    huge = WEAK_LABEL_RULES["daytime_sleep"](grid, WeakLabelConfig(min_sib_bout_minutes=10_000.0))
    assert not huge.sleep_wake.any()


def test_a_custom_rule_can_label_sleep_wake_only(monkeypatch) -> None:
    def every_fifth(grid, weights):
        n = grid.shape[0]
        labels = np.where(np.arange(n) % 5 == 0, SLEEP, UNLABELED).astype(np.int32)
        return WeakLabels(labels, (labels == SLEEP).astype(np.float32) * 0.1, labels == SLEEP)

    monkeypatch.setitem(WEAK_LABEL_RULES, "every_fifth", every_fifth)
    y = labels_with_psg(100, 40, 60)
    out = label_recording(y, np.zeros((100, 6)), ["every_fifth"])
    assert out.sleep_wake[:40][::5].all() and not out.sleep_wake[40:60].any()


def recording(epochs: int, first: int, last: int, rules=()) -> WindowRecording:
    y = labels_with_psg(epochs, first, last)
    grid = anglez_grid(epochs, slice(first - 100, last + 100)) if rules else None
    targets = label_recording(y, grid, list(rules))
    x = np.random.default_rng(epochs).normal(size=(epochs, 5, 2)).astype(np.float32)
    x[:, -1] = 0.0
    return WindowRecording(x, targets)


def sampler(rules=(), **kwargs) -> WindowSampler:
    recs = [recording(2400, 900, 1300, rules), recording(1500, 300, 700, rules)]
    return WindowSampler(recs, 960, **kwargs)


def test_sampler_shapes_and_seed_determinism() -> None:
    s = sampler(rules=BOTH)
    a = s.sample(np.random.default_rng(3))
    b = s.sample(np.random.default_rng(3))
    c = s.sample(np.random.default_rng(4))
    assert a.x.shape == (2, 960, 5, 2) == (len(a.kinds), 960, 5, 2)
    assert a.labels.shape == a.weights.shape == a.sleep_wake.shape == (2, 960)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.labels, b.labels)
    assert a.kinds == b.kinds
    assert not np.array_equal(a.starts, c.starts) or a.kinds != c.kinds


def test_every_window_has_a_supervised_epoch_and_the_mix_is_followed() -> None:
    s = sampler(rules=BOTH, mix=WindowMix(night=0, edge=0, day=1, uniform=0))
    batch = s.sample(np.random.default_rng(0), 40)
    assert (((batch.labels >= 0) & (batch.weights > 0)).sum(axis=1) > 0).all()
    assert "day" in set(batch.kinds)
    for rec, start, kind in zip(batch.recordings, batch.starts, batch.kinds):
        first, last = s.recordings[rec].psg_span()
        if kind == "day":
            assert start + 960 <= first or start >= last
        else:
            assert kind == "night"


def test_a_recording_with_few_supervised_epochs_still_yields_supervised_windows() -> None:
    s = WindowSampler([recording(5000, 2400, 2404, ())], 960, mix=WindowMix(night=0, edge=0, day=0, uniform=1), max_attempts=1)
    batch = s.sample(np.random.default_rng(0), 30)
    assert (((batch.labels >= 0) & (batch.weights > 0)).sum(axis=1) > 0).all()


def test_day_windows_fall_back_to_night_without_weak_labels() -> None:
    s = sampler(mix=WindowMix(night=0, edge=0, day=1, uniform=0))
    batch = s.sample(np.random.default_rng(0), 6)
    assert set(batch.kinds) == {"night"}


def test_edge_windows_contain_a_psg_edge_and_night_windows_cover_the_span() -> None:
    edge = sampler(mix=WindowMix(night=0, edge=1, day=0, uniform=0)).sample(np.random.default_rng(1), 20)
    s = sampler(mix=WindowMix(night=1, edge=0, day=0, uniform=0))
    night = s.sample(np.random.default_rng(1), 20)
    for batch, kind in ((edge, "edge"), (night, "night")):
        assert set(batch.kinds) == {kind}
    for rec, start in zip(edge.recordings, edge.starts):
        first, last = s.recordings[rec].psg_span()
        assert start < first < start + 960 or start < last < start + 960
    for rec, start in zip(night.recordings, night.starts):
        first, last = s.recordings[rec].psg_span()
        assert start <= first and last <= start + 960


def test_short_recordings_are_padded_with_absent_epochs_and_the_gap_label() -> None:
    s = WindowSampler([recording(300, 50, 250)], 960)
    batch = s.sample(np.random.default_rng(0), 2)
    assert (batch.labels[:, 300:] == GAP_LABEL).all()
    assert (batch.x[:, 300:, -1, 0] == 1.0).all()


def test_centered_windows_are_deterministic() -> None:
    s = sampler()
    assert np.array_equal(s.centered().starts, s.centered().starts)
    assert s.centered().x.shape[0] == 2


def test_taper_is_symmetric_positive_and_peaks_in_the_middle() -> None:
    taper = window_taper(960)
    assert np.allclose(taper, taper[::-1])
    assert (taper > 0).all() and taper.argmax() in (479, 480)
    assert taper[0] < 1e-4 < taper[480]


def test_overlapping_windows_weight_the_middle_and_reduce_to_one_window() -> None:
    n, window, step = 50, 20, 5
    inputs = np.zeros((n, 3, 1), dtype=np.float32)
    seen = []

    def infer(batch):
        seen.append(len(batch))
        # Every window reports its own index in its first logit.
        return np.stack([np.full((window, 2), float(len(seen) * 100 + i)) for i in range(len(batch))])

    out = overlapping_window_logits(infer, inputs, n, window, step, 1, batch_size=4)
    assert out.shape == (n, 2) and np.isfinite(out).all()
    starts = window_starts(n, window, step)
    assert starts[-1] == n - window and starts[0] == 0
    short = overlapping_window_logits(lambda b: np.ones((len(b), window, 2)), inputs[:8], 8, window, step, 1)
    assert short.shape == (8, 2) and np.allclose(short, 1.0)
    # an epoch covered by two windows is their taper-weighted mean
    values = {0: 0.0, 1: 10.0}
    calls = iter([0, 1])
    two = overlapping_window_logits(
        lambda b: np.stack([np.full((window, 1), values[next(calls)]) for _ in b]),
        inputs[:30], 30, window, 10, 1, batch_size=1,
    )
    taper = window_taper(window)
    epoch = 15
    expected = (taper[15] * 0.0 + taper[5] * 10.0) / (taper[15] + taper[5])
    assert two[epoch, 0] == pytest.approx(expected)


def test_weighted_loss_with_unit_weights_matches_the_unweighted_loss() -> None:
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(2, 30, 5)), dtype=jnp.float32)
    labels = jnp.asarray(rng.integers(-1, 4, size=(2, 30)))
    plain = balanced_cross_entropy(logits, labels, 5, 0.999)
    ones = jnp.ones(labels.shape)
    dummy = jnp.zeros((2, 30, 2))
    weighted = weighted_balanced_cross_entropy(
        logits, labels, ones, jnp.zeros(labels.shape, dtype=bool), dummy, 5, 0.999
    )
    assert float(weighted) == pytest.approx(float(plain), rel=1e-6)


def test_weights_scale_the_loss_and_sleep_wake_sleep_accepts_any_stage() -> None:
    logits = jnp.asarray(np.random.default_rng(1).normal(size=(1, 8, 5)), dtype=jnp.float32)
    log_probs = jnp.log(jnp.stack([jnp.full((1, 8), 0.2), jnp.full((1, 8), 0.8)], axis=-1))
    labels = jnp.asarray([[0, 0, 1, 1, 2, 3, 1, -1]])
    sw = jnp.asarray([[False, False, True, True, False, False, True, False]])
    full = weighted_balanced_cross_entropy(logits, labels, jnp.ones((1, 8)), sw, log_probs, 4, 0.999)
    half = weighted_balanced_cross_entropy(logits, labels, jnp.full((1, 8), 0.5), sw, log_probs, 4, 0.999)
    assert float(half) < float(full)
    # a sleep/wake-only sleep epoch is scored on the summed sleep probability, whatever the stage logits
    other = logits.at[:, :, 1:4].set(0.0)
    a = weighted_balanced_cross_entropy(logits, labels, jnp.ones((1, 8)), sw, log_probs, 4, 0.999)
    b = weighted_balanced_cross_entropy(other, labels, jnp.ones((1, 8)), sw, log_probs, 4, 0.999)
    assert np.isfinite(float(a)) and np.isfinite(float(b))


def test_weak_sleep_with_no_stage_in_the_batch_is_weighted_by_its_own_count() -> None:
    # A day window holds weak sleep and no PSG stage; its per-epoch weight must be the
    # effective-number class weight of those epochs, not the weight of a class seen once.
    epochs, beta, weight = 960, 0.999, 0.3
    logits = jnp.zeros((1, epochs, 5))
    log_probs = jnp.log(jnp.stack([jnp.full((1, epochs), 0.25), jnp.full((1, epochs), 0.75)], axis=-1))
    labels = jnp.ones((1, epochs), dtype=jnp.int32)
    sleep = jnp.ones((1, epochs), dtype=bool)
    weights = jnp.full((1, epochs), weight)
    loss = float(weighted_balanced_cross_entropy(logits, labels, weights, sleep, log_probs, 4, beta))
    count = epochs * weight
    class_weight = (1 - beta) / (1 - beta**count)
    expected = class_weight * weight * -np.log(0.75)
    assert loss == pytest.approx(expected, rel=1e-4)

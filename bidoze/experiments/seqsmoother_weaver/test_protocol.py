"""Scientific guardrails for time alignment, leakage, and HR aggregation."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from prepare import extract, features, resample
from report import child_errors
from train import MODEL, normalization, objective, sequence_ends


def test_known_modulated_carrier_and_matching_reference_windows():
    fs = 100
    t = np.arange(121 * fs) / fs
    # A fast wrist-vibration carrier with 72-bpm amplitude modulation.
    vibration = (1 + 0.5 * np.sin(2 * np.pi * 1.2 * t)) * np.sin(2 * np.pi * 10 * t)
    xyz = np.column_stack([vibration, np.zeros(len(t)), np.zeros(len(t))])
    rt = np.arange(121, dtype=float)
    ry = 50 + rt
    frame, psd, freq = extract(xyz, np.ones(len(t), bool), 0, rt, ry, np.ones(len(rt)), pd.DataFrame())
    assert np.allclose(frame.peak_wide, 72, atol=1.1)
    assert frame.iloc[0].reference == np.mean(ry[:60])
    assert frame.iloc[1].reference == np.mean(ry[15:75])
    assert np.allclose(np.diff(frame.endpoint), 15)
    before = features(frame, psd, freq)
    frame["reference"] = -9999
    frame["reference_quality"] = 0
    frame["psg_stage"] = "W"
    assert np.array_equal(before, features(frame, psd, freq))


def test_missing_samples_are_not_interpolated_into_valid_windows():
    t = np.arange(80 * 100) / 100
    keep = (t < 20) | (t > 21)
    d = pd.DataFrame({"actigraphtime":t[keep], "actigraphx":np.zeros(keep.sum()),
                     "actigraphy":np.ones(keep.sum()), "actigraphz":np.zeros(keep.sum())})
    xyz, valid, origin, audit = resample(d, "actigraph")
    assert audit["gap_count"] == 1
    assert not valid[2050]
    assert valid[100]
    assert np.isfinite(xyz).all()


def test_sequence_context_respects_gaps():
    t = np.arange(100) * 15.
    t[50:] += 60
    ends = sequence_ends(t)
    assert 49 in ends
    assert not any(end in ends for end in range(50, 89))
    assert 89 in ends


def test_training_normalization_never_uses_test_children():
    records = [{"meta":{"participant":"train"}, "arrays":{"X":np.array([[1., 3.], [3., 5.]])}},
               {"meta":{"participant":"test"}, "arrays":{"X":np.full((100, 2), 9999.)}}]
    mean, std = normalization(records, ["train"])
    assert np.allclose(mean, [2, 4])
    assert np.allclose(std, [1, 1])


def test_equal_recordings_within_child_then_equal_children():
    frame = pd.DataFrame({"participant":["a"] * 101 + ["b"],
        "recording":["a_n1"] * 100 + ["a_n2", "b_n1"],
        "reference":[0.] * 102, "peak":[0.] * 100 + [10., 2.]})
    child = child_errors(frame, "peak", np.ones(len(frame), bool))
    assert child["a"] == 5
    assert child.mean() == 3.5


def test_padding_cannot_change_loss():
    p = jnp.array([[60., 70.]])
    y = jnp.array([[62., 72.]])
    q = jnp.ones_like(p)
    valid = jnp.ones_like(p)
    base = objective(p, y, q, valid)
    doubled = objective(jnp.concatenate([p, jnp.full_like(p, 999.)]),
        jnp.concatenate([y, jnp.full_like(y, -999.)]),
        jnp.concatenate([q, q]), jnp.concatenate([valid, jnp.zeros_like(valid)]))
    assert np.allclose(base, doubled)


def test_architecture_count_and_finite_bounded_output():
    p = MODEL.init(jax.random.PRNGKey(0), jnp.zeros((1, 40, 26)))["params"]
    assert sum(x.size for x in jax.tree_util.tree_leaves(p)) == 106625
    y = np.asarray(MODEL.apply({"params":p}, jnp.zeros((1, 40, 26))))
    assert y.shape == (1, 40)
    assert np.isfinite(y).all()
    assert np.all((y > 30) & (y < 120))

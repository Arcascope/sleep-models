"""Score one raw recording with a tree model: wrist accelerometer (and optional heart rate) to stages.

The recording is ``t, x, y, z`` with ``t`` in seconds and acceleration in g, at whatever rate the
device recorded; it is placed on the tree's 25 Hz grid by its own timestamps
(:func:`features.to_25hz`). Epochs are 30 s from ``origin`` (default: the first sample). A model
whose inputs name ``enhanced_actigraphy`` signals gets them from
:func:`enhanced_actigraphy.cirqa_signals`; any other model gets :func:`features.epoch_signals`.

The recording is scored as it is: window features that run past its start or end are missing
(:func:`features.blank_incomplete`), as in the trees' unpadded training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arcasleep_tree import enhanced_actigraphy, features, tree

STAGES = ("wake", "light", "deep", "rem")


def feature_table(
    accel: np.ndarray,
    inputs: dict,
    heart_rate: np.ndarray | None = None,
    origin: float | None = None,
) -> pd.DataFrame:
    """The feature table of one recording: ``epoch``, ``epoch_start_s`` and the columns of ``inputs``.

    ``accel`` is ``(n, 4)``: seconds, then x, y, z in g. ``heart_rate`` is ``(m, 2)``: seconds and
    beats per minute, on the same clock, or None.
    """
    accel = np.asarray(accel, np.float64)
    accel = accel[np.argsort(accel[:, 0], kind="stable")]
    t0 = float(accel[0, 0]) if origin is None else float(origin)
    accel = accel[accel[:, 0] >= t0]
    n_epochs = int((accel[-1, 0] - t0) // features.EPOCH_S) + 1
    acc25, valid25 = features.to_25hz(accel[:, 0], accel[:, 1:4], t0, n_epochs)
    if heart_rate is None or len(heart_rate) == 0:
        hr_t, hr_bpm = np.empty(0), np.empty(0)
    else:
        hr = np.asarray(heart_rate, np.float64)
        hr = hr[np.isfinite(hr).all(axis=1)]
        hr_t, hr_bpm = hr[:, 0] - t0, hr[:, 1]
    signals = None
    if inputs.get("signals") == enhanced_actigraphy.SIGNALS_KEY:
        signals = enhanced_actigraphy.cirqa_signals(acc25, valid25)
    table = features.compute_features(acc25, valid25, hr_t, hr_bpm, inputs, signals)
    table = features.blank_incomplete(table, inputs)
    table.insert(1, "epoch_start_s", t0 + features.EPOCH_S * table.epoch.to_numpy())
    return table


def predict(
    model: dict,
    accel: np.ndarray,
    heart_rate: np.ndarray | None = None,
    origin: float | None = None,
) -> pd.DataFrame:
    """One row per epoch: ``epoch``, ``epoch_start_s``, the four stage probabilities and ``stage``.

    ``stage`` is :func:`tree.predict` (with the model's stage offsets when it has them).
    """
    table = feature_table(accel, model["inputs"], heart_rate, origin)
    p = tree.probabilities(model, table)
    out = table[["epoch", "epoch_start_s"]].copy()
    for i, name in enumerate(STAGES):
        out[f"p_{name}"] = p[:, i]
    out["stage"] = tree.predict(model, table)
    return out

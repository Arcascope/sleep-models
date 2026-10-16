"""Shared pieces of the distant-context transplant diagnostic (research log 2026-10-16 §3).

A *target* is a 30 min (60 epoch) PSG-labelled interval of a SENSE recording. Its ±90 min
(180 epoch) buffers and the target itself are never changed. A *donor* is another person's
SENSE recording; a *carrier* is a sequence of the donor's pooled feature rows (one per
30 s epoch, gap row included) written into the target recording's distant observed epochs.
Every view is built from the recording's padded model input, so the padding and every
epoch outside the replaced positions are the arrays the saved predictions were scored on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.data.batching import GAP_ROW, pad_recording_input, recording_input

X_LENGTH = 3000
TARGET = 60
BUFFER = 180
SLOT = 10
MIN_REMOTE = 720
DOSES = {"60min": 120, "6h": 720, "all": None}
SLEEP_STAGES = ("light", "deep", "rem")
STAGES = ("wake",) + SLEEP_STAGES
DRAWS = 3


@dataclass
class Recording:
    name: str
    x: np.ndarray  # (X_LENGTH, F + 1, C) padded pooled model input, float64
    n: int  # scored epochs (≤ X_LENGTH)
    psg: np.ndarray  # (n,) PSG stage name, "" where unlabelled
    observed: np.ndarray  # (n,) gap row == 0: every frame of the epoch present
    csv_observed: np.ndarray  # (n,) prediction CSV is_gap == 0: at least one frame present
    is_spt: np.ndarray  # (n,) GGIR sleep-period window flag from the prediction CSV


def load_recording(cache: Path, predictions: Path, name: str) -> Recording:
    z = np.load(cache / f"SENSE_subject_{name}_features.npz", allow_pickle=True)
    n_epochs = int(z["y"].shape[1])
    pooled = recording_input(z["X"][0], n_epochs, 1)
    x = pad_recording_input(pooled, X_LENGTH, 1)
    frame = pd.read_csv(predictions / f"SENSE_{name}.csv", keep_default_na=False)
    n = min(n_epochs, X_LENGTH)
    if len(frame) < n or not np.array_equal(frame["epoch"].to_numpy()[:n], np.arange(n)):
        raise ValueError(f"{name}: prediction CSV does not cover epochs 0..{n - 1}")
    frame = frame.iloc[:n]
    return Recording(
        name=name,
        x=x,
        n=n,
        psg=frame["psg"].astype(str).to_numpy(),
        observed=x[:n, GAP_ROW, 0] == 0.0,
        csv_observed=frame["is_gap"].to_numpy() == 0,
        is_spt=frame["is_spt"].to_numpy() == 1,
    )


def eligible_starts(rec: Recording, observed: np.ndarray) -> list[int]:
    """Target starts on the 60-epoch grid from epoch 0 that meet the fixed rule."""
    labelled = np.isin(rec.psg, STAGES)
    starts = []
    for start in range(0, rec.n - TARGET + 1, TARGET):
        left, right = start - BUFFER, start + TARGET + BUFFER
        if left < 0 or right > rec.n:
            continue
        if not observed[left:right].all() or not labelled[start:start + TARGET].all():
            continue
        if observed[:left].sum() + observed[right:].sum() < MIN_REMOTE:
            continue
        starts.append(start)
    return starts


def remote_order(rec: Recording, start: int) -> np.ndarray:
    """Observed epochs outside the target and buffers, nearest to the buffers first.

    Distance is counted outward from each buffer's edge; at equal distance the earlier
    (left) epoch comes first. Dose d replaces the first d epochs of this order, so the
    doses are nested.
    """
    left, right = start - BUFFER, start + TARGET + BUFFER
    idx = np.flatnonzero(rec.observed)
    idx = idx[(idx < left) | (idx >= right)]
    dist = np.where(idx < left, left - idx, idx - right + 1)
    side = (idx >= right).astype(int)
    return idx[np.lexsort((side, dist))]


def slots(rec: Recording, start: int) -> list[np.ndarray]:
    """Distant observed epochs cut into ≤ 10-epoch slots.

    Cells are 10 epochs counted outward from each buffer edge; an unobserved epoch splits
    a cell into separate slots. Each slot is in time order.
    """
    left, right = start - BUFFER, start + TARGET + BUFFER
    order = remote_order(rec, start)
    keys = {}
    for t in np.sort(order):
        cell = ("L", (left - 1 - t) // SLOT) if t < left else ("R", (t - right) // SLOT)
        keys.setdefault(cell, []).append(int(t))
    out = []
    for run in keys.values():
        run = np.array(run)
        breaks = np.flatnonzero(np.diff(run) != 1) + 1
        out.extend(np.split(run, breaks))
    return sorted(out, key=lambda s: s[0])


def donor_blocks(rec: Recording) -> dict[str, np.ndarray]:
    """Start epochs of 10-epoch blocks on the grid from epoch 0, fully observed and all
    PSG wake (``wake``) or all PSG light/deep/REM (``sleep``)."""
    out = {"wake": [], "sleep": []}
    for b in range(0, rec.n - SLOT + 1, SLOT):
        if not rec.observed[b:b + SLOT].all():
            continue
        stage = rec.psg[b:b + SLOT]
        if (stage == "wake").all():
            out["wake"].append(b)
        elif np.isin(stage, SLEEP_STAGES).all():
            out["sleep"].append(b)
    return {k: np.array(v, dtype=int) for k, v in out.items()}


def build_view(target: Recording, positions: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The target's padded input with ``rows`` written at ``positions``."""
    x = target.x.copy()
    x[positions] = rows
    return x


def log_odds(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def conditional_sleep(proba: np.ndarray) -> np.ndarray:
    """Sleep probability given not gap, from the five probabilities (wake, light, deep, REM, gap)."""
    stages = proba[..., :4]
    return stages[..., 1:].sum(-1) / stages.sum(-1)

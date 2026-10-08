"""Align BiDoze prediction epochs with raw PSG rows by matching stage labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["wake", "light", "deep", "rem"]
SLEEPACCEL_CODES = {0: "wake", 1: "light", 2: "light", 3: "deep", 4: "deep", 5: "rem"}
CLINICAL_CODES = {0: "wake", 1: "light", 2: "light", 3: "deep", 4: "rem"}


def raw_psg(data: Path, cohort: str, sid: str) -> pd.DataFrame:
    """PSG rows ``(time, stage)`` on the accelerometer's clock; unscored rows have stage ''."""
    if cohort == "SleepAccel":
        p = np.loadtxt(data / "SleepAccel/cleaned_psg" / f"{sid}_cleaned_psg.out")
        t, code, codes = p[:, 0], p[:, 1], SLEEPACCEL_CODES
    else:
        p = pd.read_csv(data / "SleepAccel-Clinical/cleaned_psg" / f"{sid}.csv")
        t, code, codes = p.TIMESTAMP.to_numpy(float), p.PSG.to_numpy(float), CLINICAL_CODES
    stage = [codes.get(int(c), "") if np.isfinite(c) else "" for c in code]
    return pd.DataFrame({"time": t, "stage": stage})


def epoch_times(pred_psg: pd.Series, psg: pd.DataFrame, max_shift: int = 200) -> tuple[np.ndarray, int, float]:
    """Raw-clock start time of each prediction epoch.

    The shift maximizes agreement between the prediction file's ``psg`` column and the raw PSG
    stages over epochs both score. Returns ``(times, shift, agreement)``.
    """
    a = pred_psg.fillna("").to_numpy(str)
    b = psg.stage.to_numpy(str)
    best = (-1.0, 0)
    for s in range(-max_shift, max_shift + 1):
        i = np.arange(len(a))
        j = i + s
        ok = (j >= 0) & (j < len(b))
        both = ok.copy()
        both[ok] = (a[i[ok]] != "") & (b[j[ok]] != "")
        if both.sum() < 100:
            continue
        agree = float(np.mean(a[i[both]] == b[j[both]]))
        n_lab = np.sum(a != "")
        # Prefer full overlap of labelled epochs among equal agreements.
        score = agree * min(1.0, both.sum() / n_lab)
        if score > best[0]:
            best = (score, s)
    score, s = best
    t0, step = psg.time.iloc[0], 30.0
    assert np.allclose(np.diff(psg.time), step), "PSG rows are not on a 30-s grid"
    return t0 + step * (np.arange(len(a)) + s), s, score

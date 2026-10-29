"""Per-epoch heart rate, with a missing flag.

Heart-rate samples (about 1 Hz, or every 5 s on some devices) outside 30–200 bpm are dropped. An
epoch with no valid sample has NaN heart rate and ``hr_n == 0``; nothing is filled in, so no
later data reaches an earlier epoch.
"""

from __future__ import annotations

import numpy as np

from arcasleep_watchtree.features import constants as C


def epoch_heart_rate(
    t: np.ndarray, bpm: np.ndarray, n_epochs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, within-epoch standard deviation and number of valid samples per epoch.

    ``t`` is seconds from the start of epoch 0.
    """
    t = np.asarray(t, dtype=np.float64)
    bpm = np.asarray(bpm, dtype=np.float64)
    keep = np.isfinite(bpm) & (bpm >= C.HR_MIN_BPM) & (bpm <= C.HR_MAX_BPM)
    keep &= (t >= 0) & (t < n_epochs * C.EPOCH_S)
    k = (t[keep] // C.EPOCH_S).astype(int)
    count = np.bincount(k, minlength=n_epochs).astype(np.float64)
    total = np.bincount(k, weights=bpm[keep], minlength=n_epochs)
    total_sq = np.bincount(k, weights=bpm[keep] ** 2, minlength=n_epochs)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        var = total_sq / count - (total / count) ** 2
    std = np.where(count > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return mean, std, count

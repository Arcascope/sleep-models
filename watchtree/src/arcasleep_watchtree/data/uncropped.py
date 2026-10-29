"""Keep the whole wrist recording, not only the PSG span.

Epochs before and after the PSG span have real data, so windows are warm at the first scored
epoch and full at the last. Epochs outside the PSG span carry label ``OUTSIDE_PSG`` (-3); epochs
inside it carry the PSG's labels, masked (-2) where the accelerometer has too few samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pisces_lite.datasets import align_trim, mask_data
from pisces_lite.datasets.constants import PSG_COL, TIMESTAMP_COL

OUTSIDE_PSG = -3
PSG_DT = 30


def align_uncropped(
    accel: pd.DataFrame, psg: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Trim ``accel`` to whole epochs on the PSG's epoch phase; label every epoch.

    Returns the trimmed stream, the labels (``OUTSIDE_PSG`` outside the span that
    ``align_trim`` keeps) and the time of epoch 0.
    """
    accel = accel.sort_values(by=TIMESTAMP_COL)
    _, inside = align_trim(accel, psg)
    if inside.empty:
        raise ValueError("accelerometer and PSG do not overlap")
    phase = float(psg[TIMESTAMP_COL].min()) % PSG_DT
    first, last = accel[TIMESTAMP_COL].iloc[0], accel[TIMESTAMP_COL].iloc[-1]
    start = phase + PSG_DT * np.ceil((first - phase) / PSG_DT)
    end = phase + PSG_DT * np.floor((last - phase) / PSG_DT)
    accel = accel[(accel[TIMESTAMP_COL] >= start) & (accel[TIMESTAMP_COL] <= end)]
    n_epochs = int(round((end - start) / PSG_DT)) + 1
    from_ = int(round((inside[TIMESTAMP_COL].iloc[0] - start) / PSG_DT))
    labels = np.full(n_epochs, OUTSIDE_PSG, dtype=int)
    masked = mask_data(accel, inside)
    labels[from_ : from_ + len(inside)] = masked[PSG_COL].to_numpy()
    return accel, labels, float(start)

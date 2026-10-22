"""Check the E4-to-annotation alignment used by ``convert.py``.

Usage: ``python check_alignment.py <AAUWSS raw dir> <converted dir>``

1. For each subject, find the E4 ``BVP.csv`` samples that equal the first
   epoch of the authors' aligned PPG pickle; the implied shift between E4
   unix time and the pickle's wall-clock time must equal the Europe/Copenhagen
   UTC offset (3600 s or 7200 s).
2. For each subject, correlate per-epoch movement (std of the accelerometer
   magnitude) with a scored-wake indicator at lags of -10 to +10 epochs; the
   peak should be at lag 0.
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOCAL_TZ = ZoneInfo("Europe/Copenhagen")
MATCH_SAMPLES = 40
MAX_LAG_EPOCHS = 10


def pickle_shift_seconds(raw: Path, number: int) -> tuple[float, float]:
    """Return (pickle wall clock - E4 unix as UTC wall clock, expected UTC offset), in s."""
    d = pickle.load(open(raw / "aligned_sleep_data_set" / "ppg" / f"subject_{number:02d}_ppg.pkl", "rb"))
    first = d.iloc[0, :1920].to_numpy(float)
    bvp = pd.read_csv(raw / "empatica" / f"subject_{number:02d}" / "BVP.csv", header=None)[0].to_numpy()
    start, values = bvp[0], bvp[2:]
    pattern = first[:MATCH_SAMPLES]
    for c in np.where(np.abs(values - pattern[0]) < 1e-6)[0]:
        if c + MATCH_SAMPLES <= len(values) and np.allclose(values[c : c + MATCH_SAMPLES], pattern, atol=1e-6):
            unix = start + c / 64.0
            pickle_clock = pd.Timestamp(d["time"].iloc[0][0])
            shift = (pickle_clock - pd.Timestamp(unix, unit="s")).total_seconds()
            expected = datetime.fromtimestamp(unix, LOCAL_TZ).utcoffset().total_seconds()
            return shift, expected
    raise ValueError(f"subject {number}: pickle epoch not found in BVP.csv")


def movement_lag_profile(converted: Path, sid: str) -> dict[int, float]:
    acc = np.loadtxt(converted / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
    psg = np.loadtxt(converted / "cleaned_psg" / f"{sid}_cleaned_psg.out")
    n = len(psg)
    epoch = (acc[:, 0] // 30).astype(int)
    mag = np.linalg.norm(acc[:, 1:], axis=1)
    move = pd.Series(mag).groupby(epoch).std().reindex(range(n))
    wake = pd.Series(np.where(psg[:, 1] < 0, np.nan, (psg[:, 1] == 0).astype(float)))
    out = {}
    for lag in range(-MAX_LAG_EPOCHS, MAX_LAG_EPOCHS + 1):
        out[lag] = float(move.corr(wake.shift(lag)))
    return out


def main(raw: Path, converted: Path) -> None:
    for n in range(1, 14):
        shift, expected = pickle_shift_seconds(raw, n)
        prof = movement_lag_profile(converted, f"s{n:02d}")
        best = max(prof, key=prof.get)
        print(
            f"s{n:02d} pickle shift {shift:.0f} s, Copenhagen offset {expected:.0f} s, "
            f"movement-wake r at lag 0 {prof[0]:.3f}, peak lag {best} epochs (r {prof[best]:.3f})"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

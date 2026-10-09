"""Which white-noise level per axis gives the Galaxy Watch floor (§6, §9 of the Oct 9 log)?

Synthetic nights of white Gaussian noise (σ per axis, per sample) on a regular grid, measured
exactly as ``devices.py`` and ``floor_summary.py`` do: the 2–6 Hz median of the trace spectrum per
frame, the quietest frame per 30-s epoch, the 10th percentile over epochs.

Usage: python floor_calibrate.py <out csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from devices import trace_db

HOURS = 2.0


def floor_db(sigma_g: float, fs: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, HOURS * 3600, 1 / fs)
    times, f, db = trace_db(t, rng.normal(0.0, sigma_g, (3, t.size)))
    level = np.median(db[:, (f >= 2.0) & (f <= 6.0)], axis=1)
    epoch = (times // 30).astype(int)
    quietest = pd.Series(level).groupby(epoch).min()
    return float(quietest.quantile(0.1))


def main(out: Path) -> None:
    rows = [{"fs": fs, "sigma_mg": s, "floor_db": floor_db(s / 1000, fs, i)}
            for fs in (50.0, 1000 / 60) for i, s in enumerate((1.0, 2.0, 3.0, 4.0))]
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False, float_format="%.3f")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""The release's jerk, and the same jerk divided by the time step.

The release's jerk (pisces-lite → ``senpy.compute_jerk(use_diff=True)``) is ``‖a[i] − a[i−1]‖``
with 0 for the first sample; it is not divided by ``t[i] − t[i−1]``. ``jerk_mode`` replaces
``senpy.compute_jerk`` for the release's own feature pipeline, so only the jerk changes:

- ``release``: unchanged.
- ``dt``: ``‖a[i] − a[i−1]‖ / (t[i] − t[i−1])``, senpy's ``divide_by_dt=True``.
- ``median_dt``: ``‖a[i] − a[i−1]‖ / median(Δt)``, one constant per recording. In log-PSD this
  differs from ``release`` by exactly ``−2·log(median Δt)``, so ``dt`` against ``median_dt`` is
  the effect of timestamp irregularity alone.

Time steps are taken as senpy takes them: timestamps truncated to whole microseconds.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import senpy

MODES = ("release", "dt", "median_dt")
_COMPUTE_JERK = senpy.compute_jerk
COLUMNS = ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"]


def steps_s(t: np.ndarray) -> np.ndarray:
    """Time steps in seconds after senpy's truncation to whole microseconds."""
    return np.diff((np.asarray(t, np.float64) * 1e6).astype(np.int64)) * 1e-6


def _median_dt_jerk(timestamps, x, y, z, ts_unit="s", use_diff=True):
    out = _COMPUTE_JERK(timestamps, x, y, z, ts_unit=ts_unit, use_diff=use_diff)
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6}[ts_unit]
    step = float(np.median(steps_s(np.asarray(timestamps, np.float64) * scale)))
    return senpy.JerkData(timestamps_us=out.timestamps_us, jerk=np.asarray(out.jerk) / step)


def _dt_jerk(timestamps, x, y, z, ts_unit="s", use_diff=True):
    return _COMPUTE_JERK(timestamps, x, y, z, ts_unit=ts_unit, use_diff=use_diff, divide_by_dt=True)


@contextlib.contextmanager
def jerk_mode(mode: str):
    """Within the block, ``senpy.compute_jerk`` computes the jerk of ``mode``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    replacement = {"release": _COMPUTE_JERK, "dt": _dt_jerk, "median_dt": _median_dt_jerk}[mode]
    senpy.compute_jerk = replacement
    try:
        yield
    finally:
        senpy.compute_jerk = _COMPUTE_JERK


def timing(t: np.ndarray) -> dict:
    """Sampling rate and irregularity of one recording's (cleaned) timestamps; jitter is over steps ≤ 2× the median."""
    dt = steps_s(t)
    med = float(np.median(dt))
    regular = dt[dt <= 2 * med]  # jitter, without the gaps
    return {"samples": int(t.size), "median_fs": 1.0 / med, "gaps_over_1s": int(np.sum(dt > 1.0)),
            "jitter_cv": float(np.std(regular) / np.mean(regular)),
            "dt_off_10pct": float(np.mean(np.abs(dt / med - 1) > 0.1)),
            "dt_p01_ratio": float(np.percentile(dt, 1) / med), "dt_p99_ratio": float(np.percentile(dt, 99) / med),
            "dt_zero_us": int(np.sum(dt <= 0))}


def nights(root: Path, cohorts: list[str]) -> list[tuple[str, str]]:
    out = []
    for cohort in cohorts:
        if cohort == "SleepAccel":
            for p in sorted((root / cohort / "cleaned_accelerometer").glob("*_cleaned_motion.out")):
                out.append((cohort, p.name.split("_")[0]))
        else:
            for p in sorted((root / cohort / "cleaned_psg").glob("*.csv")):
                if (root / cohort / "cleaned_accelerometer" / p.name).exists():
                    out.append((cohort, p.stem))
    return out


def load_night(root: Path, cohort: str, stem: str) -> tuple[np.ndarray, pd.DataFrame | None, float]:
    """``(t, x, y, z)`` rows, the PSG (``TIMESTAMP``, ``PSG``) where its coding is the 0–4 one, and the origin."""
    if cohort == "SleepAccel":
        accel = np.loadtxt(root / cohort / "cleaned_accelerometer" / f"{stem}_cleaned_motion.out")
        psg_t = np.loadtxt(root / cohort / "cleaned_psg" / f"{stem}_cleaned_psg.out")[:, 0]
        return accel, None, float(psg_t[0] % 30.0)
    accel = pd.read_csv(root / cohort / "cleaned_accelerometer" / f"{stem}.csv", usecols=COLUMNS,
                        dtype="float64")[COLUMNS].to_numpy()
    psg = pd.read_csv(root / cohort / "cleaned_psg" / f"{stem}.csv")
    return accel, psg, float(psg.TIMESTAMP.iloc[0] % 30.0)

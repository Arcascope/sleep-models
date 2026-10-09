"""Spectrograms of every SURF and Somerset night, for the boat-noise zoology.

Per night, from the cleaned accelerometer (no resampling; senpy NUFFT on the raw timestamps):

- the magnitude spectrogram: PSD of ||a|| (detrended per window);
- the jerk spectrogram: PSD of senpy's scalar jerk (``compute_jerk``);
- ENMO, max(||a|| − 1, 0): its minimum and maximum sample in each 2-s bin (no averaging);
- PSG stage per 30-s epoch.

Windows are 10 s at a 2-s step, as in the BiDoze input (``config/inputs.json``), and the
frequency axis runs from 0 to the recording's Nyquist frequency (25 Hz for SURF, about 8 Hz for
Somerset), beyond the model's 0.1–6 Hz band.

Writes ``<out>/spectra/<cohort>_<id>.npz`` (log10 PSD as float16); ``figures.py`` draws them.

Usage: python survey.py <data root> <out dir> [workers] [cohort,cohort,...]  (default SURF,Somerset)
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import senpy

COHORTS = ["SURF", "Somerset"]
WINDOW_S = 10.0
STEP_S = 2.0
EPS = 1e-12


def spectrogram(t: np.ndarray, x: np.ndarray, fmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(times, frequencies, log10 PSD)``; rows without enough samples are NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        s = senpy.compute_nufft_spectrogram(t, x, WINDOW_S, WINDOW_S - STEP_S, kind="psd", empty_windows="keep",
                                            origin_s=0.0)
    keep = s.frequencies <= fmax
    return s.times + (s.origin_s or 0.0), s.frequencies[keep], np.log10(s.Sxx[:, keep] + EPS)


def load_accel(root: Path, cohort: str, stem: str) -> np.ndarray:
    """``(t, x, y, z)`` rows sorted by time; rows repeating an earlier timestamp are dropped.

    Three SURF nights store every sample twice, which would give zero time steps.
    """
    a = pd.read_csv(root / cohort / "cleaned_accelerometer" / f"{stem}.csv").to_numpy(float)
    a = a[np.argsort(a[:, 0], kind="stable")]
    return a[np.r_[True, np.diff(a[:, 0]) > 0]]


def night(job: tuple[Path, str, Path]) -> str:
    root, name, out = job
    cohort, stem = name.split("/")
    target = out / "spectra" / f"{cohort}_{stem}.npz"
    if target.exists():
        return f"{name}: cached"
    t, x, y, z = load_accel(root, cohort, stem).T
    fs = 1.0 / np.median(np.diff(t))
    fmax = min(25.6, fs / 2)
    mag = senpy.compute_magnitude(x, y, z)
    tm, f, mag_psd = spectrogram(t, mag, fmax)
    j = senpy.compute_jerk(t, x, y, z)
    tj, fj, jerk_psd = spectrogram(np.asarray(j.timestamps_s, float), np.asarray(j.jerk, float), fmax)
    enmo = np.maximum(mag - 1.0, 0.0)
    bins, index = np.unique(np.floor(t / STEP_S).astype(int), return_inverse=True)
    enmo_t = bins * STEP_S
    enmo_max = np.full(len(bins), -np.inf)
    enmo_min = np.full(len(bins), np.inf)
    np.maximum.at(enmo_max, index, enmo)
    np.minimum.at(enmo_min, index, enmo)
    psg_path = root / cohort / "cleaned_psg" / f"{stem}.csv"
    psg = pd.read_csv(psg_path).to_numpy(float) if psg_path.exists() else np.zeros((0, 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, fs=fs, mag_t=tm, jerk_t=tj, f=f, fj=fj, mag=mag_psd.astype(np.float16),
                        jerk=jerk_psd.astype(np.float16), enmo_t=enmo_t, enmo_max=enmo_max.astype(np.float32), enmo_min=enmo_min.astype(np.float32),
                        psg_t=psg[:, 0], psg=psg[:, 1])
    return f"{name}: fs {fs:.1f} Hz, {t[-1] / 3600:.1f} h"


def main(root: Path, out: Path, workers: int, cohorts: list[str]) -> None:
    names = [f"{c}/{p.stem}" for c in cohorts for p in sorted((root / c / "cleaned_accelerometer").glob("*.csv"))]
    with ProcessPoolExecutor(workers) as pool:
        for line in pool.map(night, [(root, n, out) for n in names]):
            print(line, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8,
         sys.argv[4].split(",") if len(sys.argv) > 4 else COHORTS)

"""Galaxy Watch against Apple Watch on the same SURF004 nights, and both against land.

SURF004 wore both watches on four nights. The Apple Watch files hold CoreMotion's
``userAcceleration``: CoreMotion's gravity estimate is subtracted, so ``|a| − 1`` and the
magnitude spectrum are not comparable with the Galaxy Watch or with land. Every measure here is
instead taken on the *trace* spectrum: the sum of the three axes' PSDs. The trace does not depend
on the watch's orientation, and above about 0.5 Hz it does not depend on whether gravity was
subtracted.

Per 10-s frame (2-s step, senpy NUFFT on the raw timestamps), from the trace spectrum in dB:

- ``hull_db``: the largest bin in 0.1–0.5 Hz (the boat's own motion, shared by both watches);
- ``level_db``: the median bin in 2–6 Hz (the floor that masks quiet wake, §4 of the log);
- ``high_db``: the median bin in 15–24 Hz (above the wrist's own movement; sensor noise and
  vibration).

Per PSG epoch, the largest and the smallest of the 15 frames centred in it. Nothing is
averaged across frames.

Each night is cached in ``<out>/nights/``. Writes ``<out>/epochs.csv.gz`` (one row per night and PSG epoch) and ``<out>/quantization.csv``
(the smallest step between distinct values of each axis).

A night written ``cohort/stem@k`` keeps every k-th sample before the spectrum (no anti-alias
filter), to compare one device at a lower sample rate. At 16.7 Hz and below, ``high_db`` is NaN.

Usage: python devices.py <data root> <out> <night list> [workers]
  <night list>: a text file of ``cohort/stem`` or ``cohort/stem@k`` lines.
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import senpy

from survey import STEP_S, WINDOW_S, load_accel

EPS = 1e-14
PSG_STAGES = {0: "wake", 1: "light", 2: "light", 3: "deep", 4: "rem", 5: "rem"}
BANDS = {"hull_db": (0.1, 0.5, "max"), "level_db": (2.0, 6.0, "median"), "high_db": (15.0, 24.0, "median")}


def trace_db(t: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = None
    for x in axes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            s = senpy.compute_nufft_spectrogram(t, x, WINDOW_S, WINDOW_S - STEP_S, kind="psd",
                                                empty_windows="keep", origin_s=0.0)
        total = s.Sxx if total is None else total + s.Sxx
    return s.times + (s.origin_s or 0.0), s.frequencies, 10 * np.log10(total + EPS)


def night(job: tuple[Path, str, Path]) -> tuple[pd.DataFrame, dict]:
    root, name, out = job
    cohort, stem = name.split("/")
    stem, _, k = stem.partition("@")
    step = int(k or 1)
    label = f"{stem}@{step}" if step > 1 else stem
    cache = out / "nights" / f"{cohort}_{label}"
    if cache.with_suffix(".csv.gz").exists():
        return pd.read_csv(cache.with_suffix(".csv.gz")), pd.read_json(cache.with_suffix(".json"), typ="series").to_dict()
    a = load_accel(root, cohort, stem)[::step]
    t = a[:, 0]
    times, f, db = trace_db(t, a[:, 1:].T)
    frame = {}
    for key, (lo, hi, how) in BANDS.items():
        band = (f >= lo) & (f <= hi)
        if not band.any():
            frame[key] = np.full(db.shape[0], np.nan)
            continue
        frame[key] = db[:, band].max(axis=1) if how == "max" else np.median(db[:, band], axis=1)
    psg = pd.read_csv(root / cohort / "cleaned_psg" / f"{stem}.csv")
    rows = []
    for e, stage in zip(psg.TIMESTAMP.to_numpy(), psg.PSG.to_numpy()):
        inside = (times >= e) & (times < e + 30)
        if stage not in PSG_STAGES or inside.sum() < 14 or not np.isfinite(frame["level_db"][inside]).all():
            continue
        row = {"cohort": cohort, "night": label, "epoch_s": e, "psg": PSG_STAGES[stage]}
        for key, v in frame.items():
            row[f"{key}_max"] = float(v[inside].max())
            row[f"{key}_min"] = float(v[inside].min())
        rows.append(row)
    steps = {f"q_{c}": float(np.min(np.diff(np.unique(a[:200_000, k])))) for k, c in enumerate("xyz", start=1)}
    info = {"cohort": cohort, "night": label, "fs": float(1 / np.median(np.diff(t)))} | steps
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cache.with_suffix(".csv.gz"), index=False, float_format="%.3f")
    pd.Series(info).to_json(cache.with_suffix(".json"))
    return pd.DataFrame(rows), info


def main(root: Path, out: Path, names: list[str], workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(night, [(root, n, out) for n in names]))
    pd.concat([r[0] for r in results], ignore_index=True).to_csv(out / "epochs.csv.gz", index=False,
                                                                 float_format="%.3f")
    pd.DataFrame([r[1] for r in results]).to_csv(out / "quantization.csv", index=False)
    print(f"{len(names)} nights -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).read_text().split(),
         int(sys.argv[4]) if len(sys.argv) > 4 else 12)

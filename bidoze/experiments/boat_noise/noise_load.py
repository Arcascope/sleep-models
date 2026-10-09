"""Per-epoch noise load of every night, and whether it goes with wake epochs scored as sleep.

Every quantity is computed within one 10-s frame (10-s window, 2-s step, as in ``survey.py``) and
compared only with other bins of that same frame; no spectrum is averaged, smoothed or taken a
median of across frames, and nothing is referenced to a whole-night level. Over the model's band
(0.1–6 Hz), per frame and channel (|a| and jerk):

- ``prominence``: each bin's power (dB) above the median of the 7 bins centred on it (0.7 Hz) in
  the same frame; a *line* is a bin in 0.5–6 Hz at ≥ ``LINE_DB`` above that median;
- ``hull_db`` (|a| only): the largest bin in 0.1–0.5 Hz above the median bin in 0.5–2 Hz of the
  same frame;
- ``level_db``: the median bin in 2–6 Hz, an absolute level that rises with movement.

Per PSG epoch, from the 15 frames centred inside it: the share of frames with a line (|a|, jerk),
the most lines in one frame, the largest ``hull_db`` and the largest ``level_db``; also the largest
ENMO sample in the epoch. These are counts and maxima of per-frame values, not averages of spectra.

The released adult model's call for the epoch is joined from ``<predictions>/<night>.csv``
(``predict.py`` format), matched on the epoch's start time.

Writes ``<out>/epochs_<cohort>.csv.gz`` (one row per PSG epoch with a prediction).

Usage: python noise_load.py <out> <spectra dir> <predictions dir> <cohort label> [workers]
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

LINE_DB = 10.0
FMIN, FMAX = 0.1, 6.0
PSG_STAGES = {0: "wake", 1: "light", 2: "light", 3: "deep", 4: "rem"}
FRAMES_PER_EPOCH = 15


def frame_features(f: np.ndarray, log10_psd: np.ndarray, hull: bool) -> dict[str, np.ndarray]:
    band = (f >= FMIN - 1e-9) & (f <= FMAX + 1e-9)
    f, db = f[band], 10.0 * log10_psd[:, band].astype(np.float64)
    prominence = db - median_filter(db, size=(1, 7), mode="nearest")
    lines = (prominence >= LINE_DB) & (f >= 0.5)
    out = {"lines": lines.sum(axis=1), "level_db": np.median(db[:, f >= 2.0], axis=1)}
    if hull:
        out["hull_db"] = db[:, f <= 0.5].max(axis=1) - np.median(db[:, (f > 0.5) & (f <= 2.0)], axis=1)
    return out


def night(job: tuple[Path, Path, str]) -> pd.DataFrame | None:
    spectra, predictions, cohort = job
    stem = spectra.stem.split("_", 1)[1]
    pred_path = predictions / f"{stem}.csv"
    if not pred_path.exists():
        return None
    d = np.load(spectra)
    mag = frame_features(d["f"], d["mag"], hull=True)
    jerk = frame_features(d["fj"], d["jerk"], hull=False)
    psg_t, psg = d["psg_t"], d["psg"].astype(int)
    p = pd.read_csv(pred_path)
    pred = p.set_index(np.rint(p.start_s.to_numpy()).astype(int))
    rows = []
    for t, stage in zip(psg_t, psg):
        if stage not in PSG_STAGES or int(round(t)) not in pred.index:
            continue
        call = pred.loc[int(round(t))]
        if call.stage == "gap":
            continue
        inside = (d["mag_t"] >= t) & (d["mag_t"] < t + 30)
        inside_j = (d["jerk_t"] >= t) & (d["jerk_t"] < t + 30)
        e = (d["enmo_t"] >= t) & (d["enmo_t"] < t + 30)
        if inside.sum() < FRAMES_PER_EPOCH - 1 or not np.isfinite(mag["level_db"][inside]).all():
            continue
        rows.append({
            "night": stem, "epoch_s": t, "psg": PSG_STAGES[stage],
            "call": call.stage, "p_wake": call.wake,
            "line_frac_mag": float(np.mean(mag["lines"][inside] > 0)),
            "line_frac_jerk": float(np.mean(jerk["lines"][inside_j] > 0)),
            "lines_max_mag": int(mag["lines"][inside].max()),
            "lines_max_jerk": int(jerk["lines"][inside_j].max()),
            "hull_db": float(mag["hull_db"][inside].max()),
            "level_db": float(mag["level_db"][inside].max()),
            "enmo_max": float(d["enmo_max"][e].max()) if e.any() else np.nan,
        })
    out = pd.DataFrame(rows)
    out.insert(0, "cohort", cohort)
    return out


def main(out: Path, spectra: Path, predictions: Path, cohort: str, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(spectra.glob(f"{cohort}_*.npz"))
    with ProcessPoolExecutor(workers) as pool:
        frames = [f for f in pool.map(night, [(s, predictions, cohort) for s in files]) if f is not None]
    table = pd.concat(frames, ignore_index=True)
    target = out / f"epochs_{cohort}.csv.gz"
    table.to_csv(target, index=False, float_format="%.4f")
    print(f"{cohort}: {table.night.nunique()} nights, {len(table)} epochs -> {target}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 12)

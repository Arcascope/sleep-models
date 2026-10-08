"""One row per BiDoze-scored epoch: the release's cross-validated output, PSG, and three HRs.

Per 30-s PSG epoch starting at T (raw clock), each HR describes the centred 60-s window
[T − 15, T + 45):
- ``hr_seqsmoother``: SeqSmoother HR of the window ending at T + 45 (``hr_windows.csv.gz``).
  SleepAccel: the fold model that held the subject's fold out; SleepAccel-Clinical: the model
  trained on all SleepAccel.
- ``hr_jerk``: peak of the jerk power in 0.9–1.3 Hz (54–78 bpm, 1-bpm bins); jerk from senpy's
  ``compute_jerk``, power from ``compute_nufft_spectrogram`` on the raw timestamps.
  ``jerk_clarity`` is the peak's power over the band mean; ``stillness`` the SD of acceleration
  magnitude in the window.
- ``hr_watch`` (SleepAccel only): median of the Apple Watch's recorded HR in the window.

Model output: the 3-seed mean of the ``matched`` arm's 5-way output. SleepAccel subjects come
from the fold models that held them out; SleepAccel-Clinical from the released model (``final``).

Usage: python epoch_hr.py --data <pisces2-data> --arm <adult-devices run>/matched --run <run> [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import senpy

from align import STAGES, epoch_times, raw_psg

SEEDS = (0, 1, 2)
BAND = (0.9, 1.3)


def load_output(path: Path) -> tuple[pd.Series, np.ndarray]:
    d = pd.read_csv(path)
    stage = d[STAGES].to_numpy(float) * (1.0 - d["gap_head"].to_numpy(float))[:, None]
    return d["psg"], np.column_stack([stage, d["gap_head"].to_numpy(float)])


def raw_accel(data: Path, cohort: str, sid: str) -> np.ndarray:
    if cohort == "SleepAccel":
        return np.loadtxt(data / "SleepAccel/cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
    return pd.read_csv(data / "SleepAccel-Clinical/cleaned_accelerometer" / f"{sid}.csv").to_numpy(float)


def jerk_peaks(acc: np.ndarray, starts: np.ndarray) -> pd.DataFrame:
    """Jerk-band peak HR, clarity and stillness for 60-s windows starting at ``starts`` (30-s grid)."""
    acc = acc[np.argsort(acc[:, 0], kind="stable")]
    t = acc[:, 0]
    jerk = senpy.compute_jerk(t, acc[:, 1], acc[:, 2], acc[:, 3])
    origin = float(starts[0])
    spec = senpy.compute_nufft_spectrogram(jerk.timestamps_us / 1e6, jerk.jerk, window_s=60.0, overlap_s=30.0,
                                           kind="power", target_fs=4.0, origin_s=origin, empty_windows="keep",
                                           min_samples=1500)
    f = spec.frequencies
    band = (f >= BAND[0] - 1e-9) & (f <= BAND[1] + 1e-9)
    index = np.round((starts - origin) / 30.0).astype(int)
    rows = dict(hr_jerk=np.full(len(starts), np.nan), jerk_clarity=np.full(len(starts), np.nan))
    lookup = {int(w): i for i, w in enumerate(spec.window_index)}
    for e, w in enumerate(index):
        i = lookup.get(int(w))
        if i is None or not spec.valid[i]:
            continue
        bp = spec.Sxx[i, band]
        if not np.isfinite(bp).all() or bp.mean() <= 0:
            continue
        k = int(np.argmax(bp))
        rows["hr_jerk"][e] = f[band][k] * 60
        rows["jerk_clarity"][e] = bp[k] / bp.mean()
    mag = np.linalg.norm(acc[:, 1:4], axis=1)
    c1, c2 = np.r_[0, np.cumsum(mag)], np.r_[0, np.cumsum(mag ** 2)]
    lo, hi = np.searchsorted(t, starts), np.searchsorted(t, starts + 60)
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c1[hi] - c1[lo]) / n
        var = (c2[hi] - c2[lo]) / n - mean ** 2
    rows["stillness"] = np.where(n >= 1500, np.sqrt(np.maximum(var, 0)), np.nan)
    return pd.DataFrame(rows)


def watch_hr(data: Path, sid: str, starts: np.ndarray) -> np.ndarray:
    hr = np.loadtxt(data / "SleepAccel/cleaned_heartrate" / f"{sid}_cleaned_hr.out")
    hr = hr[np.isfinite(hr).all(axis=1)]
    hr = hr[np.argsort(hr[:, 0], kind="stable")]
    out = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        a, b = np.searchsorted(hr[:, 0], [s, s + 60])
        if b > a:
            out[i] = np.median(hr[a:b, 1])
    return out


def subject(job: tuple) -> pd.DataFrame:
    cohort, sid, fold, paths, data, windows = job
    outputs = [load_output(p) for p in paths]
    psg_col, proba = outputs[0][0], np.mean([o[1] for o in outputs], axis=0)
    psg = raw_psg(data, cohort, sid)
    times, shift, agreement = epoch_times(psg_col, psg)
    if agreement < 0.99:
        raise ValueError(f"{cohort} {sid}: PSG alignment agreement {agreement:.3f}")
    starts = times - 15.0
    frame = pd.DataFrame(dict(cohort=cohort, subject=sid, fold=fold, epoch=np.arange(len(times)), time=times,
                              psg=psg_col.map({s: i for i, s in enumerate(STAGES)}).fillna(-1).astype(int)))
    for i, name in enumerate(["p_wake", "p_light", "p_deep", "p_rem", "p_gap"]):
        frame[name] = proba[:, i]
    frame = pd.concat([frame, jerk_peaks(raw_accel(data, cohort, sid), starts)], axis=1)
    w = windows.sort_values("endpoint")
    target = times + 45.0
    j = np.searchsorted(w.endpoint.to_numpy(), target).clip(1, max(len(w) - 1, 1))
    ends = w.endpoint.to_numpy()
    nearest = np.where(np.abs(ends[j - 1] - target) <= np.abs(ends[j] - target), j - 1, j)
    hr = w.seqsmoother.to_numpy()[nearest]
    frame["hr_seqsmoother"] = np.where(np.abs(ends[nearest] - target) <= 7.5, hr, np.nan)
    frame["hr_watch"] = watch_hr(data, sid, starts) if cohort == "SleepAccel" else np.nan
    print(f"{cohort} {sid}: {len(frame)} epochs, shift {shift}", flush=True)
    return frame


def jobs(arm: Path, data: Path, windows: pd.DataFrame) -> list[tuple]:
    out = []
    for k in range(5):
        for p in sorted((arm / f"fold{k}/predictions/fold{k}_seed_0").glob("BiDozeSleepAccelFold*.csv")):
            sid = p.stem.split("_", 1)[1]
            paths = [arm / f"fold{k}/predictions/fold{k}_seed_{s}" / p.name for s in SEEDS]
            out.append(("SleepAccel", sid, k, paths, data, windows[windows.participant.eq(sid)]))
    final = arm / "final/predictions"
    for p in sorted((final / "final_seed_0").glob("SleepAccel-Clinical_*.csv")):
        sid = p.stem.split("_", 1)[1]
        paths = [final / f"final_seed_{s}" / p.name for s in SEEDS]
        out.append(("SleepAccel-Clinical", sid, -1, paths, data, windows[windows.participant.eq(sid)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    windows = pd.read_csv(a.run / "hr_windows.csv.gz", dtype={"participant": str})
    with ProcessPoolExecutor(a.workers) as pool:
        frames = list(pool.map(subject, jobs(a.arm, a.data, windows)))
    pd.concat(frames, ignore_index=True).to_csv(a.run / "epochs.csv.gz", index=False)


if __name__ == "__main__":
    main()

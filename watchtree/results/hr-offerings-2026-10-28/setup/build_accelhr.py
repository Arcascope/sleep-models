"""Accelerometer-derived pulse rate per night (step 1 of the accelerometer-HR work).

For every kept night of the four tree sources, for W in 10, 20, 30 s: one estimate per 5 s (bpm, quality; see accel_hr.py),
and per scored epoch the movement (mean jerk of the 25 Hz stream) and the device HR mean over the same window [e - W, e).
Writes $OUT/accelhr/series/<source>/<night>.parquet (window end 30 + 5 j s) and $OUT/accelhr/epochs/<source>/<night>.parquet.
Resumable. One thread per worker.
"""
import argparse, multiprocessing, os, time, warnings
from concurrent.futures import ProcessPoolExecutor

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
import pandas as pd

from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, load_night, subject_ids
from arcasleep_watchtree.uncropped import RAW_ROOT, load_night_uncropped, sync_table
import accel_hr as A

_STATE: dict = {}


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset

    c = COHORTS[name]
    _STATE.update(cohort=c, name=name, data_set=load_dataset(DATA_ROOT / c.directory))
    if name == "wearanize":
        _STATE["sync"] = sync_table(DATA_ROOT / RAW_ROOT)


def _one(args):
    subject, reported, out = args
    sp, ep = out / "series" / f"{reported}.parquet", out / "epochs" / f"{reported}.parquet"
    if sp.exists() and ep.exists():
        return reported, "cached", 0.0
    t0 = time.time()
    c = _STATE["cohort"]
    if _STATE["name"] == "wearanize":
        night = load_night_uncropped(subject, reported, _STATE["sync"], DATA_ROOT / c.directory)
    else:
        night = load_night(c, subject, reported, _STATE["data_set"])
    ne = night.labels.size
    ser = {}
    for W in A.WINDOWS_S:
        b, q = A.accel_hr(night.acc25, night.valid25, ne, W)
        ser[f"bpm{W}"], ser[f"q{W}"] = b.astype(np.float32), q.astype(np.float32)
    ser = pd.DataFrame(ser)
    ser.insert(0, "j", np.arange(len(ser)))
    labels = night.labels
    epochs = np.flatnonzero((labels >= 0) & (labels <= 5))
    acc = night.acc25[: ne * 750]
    jerk = np.r_[0.0, np.linalg.norm(np.diff(acc, axis=0), axis=1)]
    jerk = np.where(night.valid25[: ne * 750], jerk, np.nan).reshape(ne, 750)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mov = np.nanmean(jerk, axis=1)
    t, b = np.asarray(night.hr_t, float), np.asarray(night.hr_bpm, float)
    keep = np.isfinite(t) & np.isfinite(b) & (b >= 30) & (b <= 200)
    t, b = t[keep], b[keep]
    o = np.argsort(t, kind="stable"); t, b = t[o], b[o]
    cs = np.r_[0.0, np.cumsum(b)]
    ends = (epochs + 1) * 30.0
    e = pd.DataFrame({"epoch": epochs, "stage5": labels[epochs].astype(np.int8),
                      "y4": np.array([0, 1, 1, 2, 2, 3])[labels[epochs].astype(int)].astype(np.int8), "move": mov[epochs]})
    for W in A.WINDOWS_S:
        lo, hi = np.searchsorted(t, ends - W, "left"), np.searchsorted(t, ends, "left")
        n = hi - lo
        e[f"dev{W}"] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        e[f"ndev{W}"] = n
        j = 6 * epochs
        ok = j < len(ser)
        e[f"est{W}"] = np.where(ok, ser[f"bpm{W}"].to_numpy()[np.minimum(j, len(ser) - 1)], np.nan)
        e[f"q{W}"] = np.where(ok, ser[f"q{W}"].to_numpy()[np.minimum(j, len(ser) - 1)], np.nan)
    e.insert(0, "subject", reported)
    for p in (sp, ep):
        p.parent.mkdir(parents=True, exist_ok=True)
    ser.to_parquet(sp.with_suffix(".tmp"), index=False); sp.with_suffix(".tmp").rename(sp)
    e.to_parquet(ep.with_suffix(".tmp"), index=False); ep.with_suffix(".tmp").rename(ep)
    return reported, "built", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", default="sleepaccel,wearanize,weaver_garmin,weaver_fitbit")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from bpm_common import OUT, source_tables

    t_all = time.time()
    for name in a.sources.split(","):
        sel = set(source_tables()[name].subject)
        ids, reported = subject_ids(COHORTS[name])
        todo = [(s, reported[s]) for s in ids if reported[s] in sel]
        if a.limit:
            todo = todo[: a.limit]
        out = OUT / "accelhr"
        sub = out
        ctx = multiprocessing.get_context("spawn")
        built, tt = 0, []
        with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_init, initargs=(name,)) as pool:
            for rep, status, sec in pool.map(_one, [(s, r, out / name) for s, r in todo]):
                built += status == "built"
                if status == "built":
                    tt.append(sec)
        print(f"{name}: nights {len(todo)} built {built} mean s/night {np.mean(tt) if tt else 0:.1f} elapsed {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()

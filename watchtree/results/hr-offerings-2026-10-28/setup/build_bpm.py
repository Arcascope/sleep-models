"""Build the BPM feature files, one parquet per night (resumable: existing files are skipped).

    python build_bpm.py [--workers 6]            all kept nights of the four tree sources, into $EXP4_OUT/features/<source>/
    EXP4_SMOKE=1 python build_bpm.py ...         only the smoke nights, into $EXP4_OUT (smoke dir) /features

Nights are loaded as the tree's feature tables were built (cropped loader for SleepAccel and Weaver, the uncropped Wearanize+
loader for wearanize, so epoch indices match). Only epochs with a PSG stage get rows. Each worker uses one thread.
"""
import argparse, multiprocessing, os, time, warnings
from concurrent.futures import ProcessPoolExecutor

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
import pandas as pd

from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, load_night, subject_ids
from arcasleep_watchtree.uncropped import RAW_ROOT, load_night_uncropped, sync_table
import bpm_features as BF

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
    path = out / f"{reported}.parquet"
    if path.exists():
        return reported, "cached", 0.0
    t0 = time.time()
    c = _STATE["cohort"]
    if _STATE["name"] == "wearanize":
        night = load_night_uncropped(subject, reported, _STATE["sync"], DATA_ROOT / c.directory)
    else:
        night = load_night(c, subject, reported, _STATE["data_set"])
    labels = night.labels
    epochs = np.flatnonzero((labels >= 0) & (labels <= 5))
    f = BF.night_features(night.hr_t, night.hr_bpm, labels.size, epochs, _STATE["name"])
    y4 = np.array([0, 1, 1, 2, 2, 3])[labels[epochs].astype(int)]
    f.insert(0, "y4", y4.astype(np.int8))
    f.insert(0, "subject", reported)
    f.to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").rename(path)
    return reported, "built", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sources", default="wearanize,sleepaccel,weaver_garmin,weaver_fitbit")
    a = ap.parse_args()
    from bpm_common import FEAT_DIR, SMOKE, smoke_subjects, source_tables

    keep = smoke_subjects() if SMOKE else None
    t_all = time.time()
    for name in a.sources.split(","):
        sel = set(source_tables()[name].subject)
        if keep is not None:
            sel &= keep
        ids, reported = subject_ids(COHORTS[name])
        todo = [(s, reported[s]) for s in ids if reported[s] in sel]
        assert len(todo) == len(sel), (name, len(todo), len(sel))
        out = FEAT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        ctx = multiprocessing.get_context("spawn")
        built, tt = 0, []
        with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_init, initargs=(name,)) as pool:
            for rep, status, sec in pool.map(_one, [(s, r, out) for s, r in todo]):
                built += status == "built"
                if status == "built":
                    tt.append(sec)
        print(f"{name}: nights {len(todo)} built {built} cached {len(todo) - built} "
              f"mean s/night {np.mean(tt) if tt else 0:.1f} elapsed {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()

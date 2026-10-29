"""Per night, every estimator of ahr2.py for W = 30, 60, 120 s at every scored epoch, plus movement and device HR.
Writes $R/est/<source>/<night>.parquet. Resumable; one thread per worker."""
import argparse, multiprocessing, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd
from pathlib import Path
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.uncropped import RAW_ROOT, sync_table
import ahr2 as A
from loadnight import load_with_rows

R = Path(os.environ["R"])
WINDOWS = (30, 60, 120)
METHODS = ("peak", "hsum", "hps", "sacf")
VIT = ((0.05, "a"), (0.15, "b"), (0.5, "c"))
_S = {}


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    c = COHORTS[name]
    _S.update(name=name, c=c, ds=load_dataset(DATA_ROOT / c.directory),
              sync=sync_table(DATA_ROOT / RAW_ROOT) if name == "wearanize" else None)


def _place(v, W, ne):
    """window k -> epoch e = k + W/30 - 1"""
    out = np.full(ne, np.nan, np.float32)
    off = W // 30 - 1
    out[off: off + len(v)] = v
    return out


def _one(args):
    subject, reported, out = args
    dst = out / f"{reported}.parquet"
    if dst.exists():
        return reported, "cached", 0.0
    t0 = time.time()
    night, _ = load_with_rows(_S["name"], subject, reported, _S["ds"], _S["sync"])
    ne = night.labels.size
    labels = night.labels
    ep = np.flatnonzero((labels >= 0) & (labels <= 5))
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
    ends = (ep + 1) * 30.0
    d = {"subject": reported, "epoch": ep, "stage5": labels[ep].astype(np.int8),
         "y4": np.array([0, 1, 1, 2, 2, 3])[labels[ep].astype(int)].astype(np.int8), "move": mov[ep]}
    for W in WINDOWS:
        lo, hi = np.searchsorted(t, ends - W, "left"), np.searchsorted(t, ends, "left")
        n = hi - lo
        d[f"dev{W}"] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        d[f"ndev{W}"] = n
    for W in WINDOWS:
        sp, fr = A.spectra(night.acc25, night.valid25, ne, W)
        cols = {}
        if sp is not None:
            for wh in (0, 1):
                for ty in A.TYPES:
                    S = A.whiten(sp[ty], fr) if wh else sp[ty]
                    for m in METHODS:
                        sc = A.scores(S, fr, m)
                        cols[f"{ty}.{m}.{wh}.{W}"] = A.pick(sc, lin=(m != "hps"))
                    if wh and ty in ("sum", "mag"):
                        sh = A.scores(S, fr, "hsum")
                        em = sh / np.nanmean(sh, axis=1, keepdims=True)
                        for lam, tag in VIT:
                            cols[f"{ty}.vit{tag}.{wh}.{W}"] = A.viterbi_filter(em, lam)
        for band, sos in ((3, A._SOS_3), (8, A._SOS_8)):
            cols[f"acf.acf{band}.0.{W}"] = A.acf_est(night.acc25, night.valid25, ne, W, sos)
        for k, (bp, q) in cols.items():
            d[f"est.{k}"] = _place(bp, W, ne)[ep]
            d[f"q.{k}"] = _place(q, W, ne)[ep]
    df = pd.DataFrame(d)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst.with_suffix(".tmp"), index=False); dst.with_suffix(".tmp").rename(dst)
    return reported, "built", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sources", default="sleepaccel,weaver_fitbit,weaver_garmin,wearanize")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from bpm_common import source_tables
    for name in a.sources.split(","):
        sel = set(source_tables()[name].subject)
        ids, rep = subject_ids(COHORTS[name])
        todo = [(s, rep[s]) for s in ids if rep[s] in sel]
        if a.limit:
            todo = todo[: a.limit]
        ctx = multiprocessing.get_context("spawn")
        tt = []
        with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_init, initargs=(name,)) as pool:
            for r, st, sec in pool.map(_one, [(s, r, R / "est" / name) for s, r in todo]):
                if st == "built": tt.append(sec)
        print(f"{name}: nights {len(todo)} built {len(tt)} mean s/night {np.mean(tt) if tt else 0:.1f}", flush=True)

if __name__ == "__main__":
    main()

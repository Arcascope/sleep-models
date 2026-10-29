"""Per CV night: autocorrelation curves for the rate-free arms (ratefree.py: R1, R2, R2L, R1c, each at the 0.67-8 and 0.67-3 Hz edges), per-window peak and
forward-only Viterbi estimates (emission 10 r, lambda 0.5 / 1 / 2). Writes $N/est/<cohort>/<night>.parquet keyed (subject, epoch)."""
import os, sys, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np, pandas as pd
import ahr2 as A, ratefree as RF
from common2 import R
from splits import night_folds
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.preprocess import nominal_rate
from loadnight import load_with_rows
N = Path(os.environ["N"]); _S = {}
LAMS = (0.5, 1.0, 2.0)


def lam_tag(l): return str(l).replace(".", "p")


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    c = COHORTS[name]; sync = None
    if name == "wearanize":
        from arcasleep_watchtree.uncropped import sync_table, RAW_ROOT; sync = sync_table(DATA_ROOT / RAW_ROOT)
    _S.update(name=name, ds=load_dataset(DATA_ROOT / c.directory), sync=sync)


def _one(args):
    subject, rep, name = args
    dst = N / "est" / name / f"{rep}.parquet"
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(R / "est" / name / f"{rep}.parquet", columns=["epoch"])
    night, rows = load_with_rows(name, subject, rep, _S["ds"], _S["sync"])
    ne = night.labels.size; t = rows[:, 0]; t0 = night.t0
    valid_ep = RF.coverage(t, t0, ne) >= 0.8
    idx = np.flatnonzero(night.valid25[:ne * 750])
    C, nyq = RF.curves_all(t, rows[:, 1:4], t0, ne, valid_ep, with_control=(t0 + idx / 25.0, night.acc25[idx]))
    ep = m.epoch.to_numpy(); d = {"subject": rep, "epoch": ep, "nyq": nyq[ep].astype(np.float32), "nominal": nominal_rate(t)}
    for k, c in C.items():
        pk = np.where(np.isfinite(c).all(axis=1), A.GRID_BPM[np.nanargmax(np.where(np.isfinite(c), c, -9), axis=1)], np.nan)
        d[f"pk.{k}"] = pk[ep].astype(np.float32)
        for lam in LAMS:
            bp, _ = A.viterbi_filter(10.0 * c, lam)
            d[f"est.{k}_l{lam_tag(lam)}"] = bp[ep].astype(np.float32)
    pd.DataFrame(d).to_parquet(dst.with_suffix(".tmp"), index=False); dst.with_suffix(".tmp").rename(dst)
    return 1


if __name__ == "__main__":
    name = sys.argv[1]
    f = night_folds(); keep = set(f[f.cohort == name].subject)
    ids, rep = subject_ids(COHORTS[name])
    todo = [(s, rep[s], name) for s in ids if rep[s] in keep]
    if len(sys.argv) > 2: todo = todo[:int(sys.argv[2])]
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 12)), mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(name,)) as pool:
        list(pool.map(_one, todo))
    print(name, len(todo), flush=True)

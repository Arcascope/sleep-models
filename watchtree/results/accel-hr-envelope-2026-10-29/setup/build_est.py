"""Per night: curves for the current estimator, Hilbert-envelope and squared+low-pass envelope estimators, every candidate setting;
per-window peak (no tracking) and forward-only Viterbi estimate (emission 10 r, jump penalty lam per bpm). Writes $N/est/<cohort>/<night>.parquet
keyed (subject, epoch) like accel-hr-v2-2026-10-28/est3."""
import os, sys, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np, pandas as pd
import ahr2 as A
import envlib as E
from common2 import R
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.uncropped import RAW_ROOT, sync_table
from loadnight import load_with_rows
N = Path(os.environ["N"]); _S = {}; HP = os.environ.get("HP") == "1"; SUB = "est_hp" if HP else "est"


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    c = COHORTS[name]
    _S.update(name=name, ds=load_dataset(DATA_ROOT / c.directory), sync=sync_table(DATA_ROOT / RAW_ROOT) if name == "wearanize" else None)


def _one(args):
    subject, rep, out = args
    dst = out / f"{rep}.parquet"
    if dst.exists():
        return 0
    e0 = pd.read_parquet(R / "est" / _S["name"] / f"{rep}.parquet", columns=["epoch"])
    night, _ = load_with_rows(_S["name"], subject, rep, _S["ds"], _S["sync"])
    ne = night.labels.size
    C = E.all_curves(night.acc25, night.valid25, ne, hp=HP)
    ep = e0.epoch.to_numpy(); d = {"subject": rep, "epoch": ep}
    for k, c in C.items():
        pk = np.where(np.isfinite(c).all(axis=1), A.GRID_BPM[np.nanargmax(np.where(np.isfinite(c), c, -9), axis=1)], np.nan)
        d[f"pk.{k}"] = pk[ep].astype(np.float32)
    for k, c in C.items():
        for lam in ((1.0,) if k == "cur" else E.LAMS):
            bp, _ = A.viterbi_filter(10.0 * c, lam)
            d[f"est.{k}_l{E.lam_tag(lam)}"] = bp[ep].astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(d).to_parquet(dst.with_suffix(".tmp"), index=False); dst.with_suffix(".tmp").rename(dst)
    return 1


if __name__ == "__main__":
    from bpm_common import source_tables
    for name in sys.argv[1].split(","):
        sel = set(source_tables()[name].subject)
        ids, rep = subject_ids(COHORTS[name])
        todo = [(s, rep[s], N / SUB / name) for s in ids if rep[s] in sel]
        if len(sys.argv) > 2: todo = todo[:int(sys.argv[2])]
        with ProcessPoolExecutor(int(os.environ.get("NPROC", 12)), mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(name,)) as pool:
            list(pool.map(_one, todo))
        print(name, len(todo), flush=True)

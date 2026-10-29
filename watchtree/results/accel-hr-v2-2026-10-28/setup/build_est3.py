"""Third pass: stronger jump penalties for the autocorrelation-curve tracker (acfvd 1, acfve 2, acfvf 4 per bpm). Per night, W = 30, 60, 120, bands 3 and 8 Hz: peak of the autocorrelation
curve on the 1 bpm grid (acfg), and forward-only Viterbi tracking of the curve (emission 10 x autocorrelation, jump penalty
lam per bpm: acfva 0.05, acfvb 0.15, acfvc 0.5). Writes $R/est2/<cohort>/<night>.parquet keyed (subject, epoch)."""
import os, warnings, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np, pandas as pd
import ahr2 as A
from common2 import R, COH
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.uncropped import RAW_ROOT, sync_table
from loadnight import load_with_rows
_S = {}
VIT = ((1.0, "d"), (2.0, "e"), (4.0, "f"))


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    c = COHORTS[name]
    _S.update(name=name, ds=load_dataset(DATA_ROOT / c.directory), sync=sync_table(DATA_ROOT / RAW_ROOT) if name == "wearanize" else None)


def _place(v, W, ne):
    out = np.full(ne, np.nan, np.float32); off = W // 30 - 1; out[off: off + len(v)] = v; return out


def _one(args):
    subject, rep, out = args
    dst = out / f"{rep}.parquet"
    if dst.exists():
        return 0
    e0 = pd.read_parquet(R / "est" / _S["name"] / f"{rep}.parquet", columns=["epoch"])
    night, _ = load_with_rows(_S["name"], subject, rep, _S["ds"], _S["sync"])
    ne = night.labels.size
    d = {"subject": rep, "epoch": e0.epoch.to_numpy()}
    for W in (30, 60, 120):
        for band, sos in ((3, A._SOS_3), (8, A._SOS_8)):
            C = A.acf_curves(night.acc25, night.valid25, ne, W, sos)
            for lam, tag in VIT:
                bp, q = A.viterbi_filter(10.0 * C, lam, raw_quality=True)
                q = q / 10.0  # autocorrelation value at the tracked rate
                d[f"est.acf.acfv{tag}{band}.0.{W}"] = _place(bp, W, ne)[e0.epoch]; d[f"q.acf.acfv{tag}{band}.0.{W}"] = _place(q, W, ne)[e0.epoch]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(d).to_parquet(dst.with_suffix(".tmp"), index=False); dst.with_suffix(".tmp").rename(dst)
    return 1


if __name__ == "__main__":
    import sys
    from bpm_common import source_tables
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else COH
    for name in names:
        sel = set(source_tables()[name].subject)
        ids, rep = subject_ids(COHORTS[name])
        todo = [(s, rep[s], R / "est3" / name) for s in ids if rep[s] in sel]
        with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(name,)) as pool:
            list(pool.map(_one, todo))
        print(name, len(todo), flush=True)

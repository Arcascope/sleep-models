"""Individual-window spectra at the PSG-sleep, stillest-decile epochs (movement cutpoint from the fold-0 training sleep epochs of
the cohort, as common2.py). Stores sum-PSD rows (the saved freqs array is the W = 120 grid; the W = 60 grid is 0.4 + k/60 Hz) (0.4-12 Hz) for W = 60 and 120 s plus per-window pulse-line statistics:
whitened PSD (running median over +-1.5 Hz) at the device HR f, 2f, 3f and at control frequencies, per window (no averaging
over windows). Writes $R/still/<cohort>/<night>.npz."""
import os, warnings, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np, pandas as pd
import ahr2 as A
from common2 import R, COH, STILL_Q
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.uncropped import RAW_ROOT, sync_table
from loadnight import load_with_rows
_S = {}


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    c = COHORTS[name]
    _S.update(name=name, ds=load_dataset(DATA_ROOT / c.directory), sync=sync_table(DATA_ROOT / RAW_ROOT) if name == "wearanize" else None)


def _one(args):
    subject, rep, thr, out = args
    dst = out / f"{rep}.npz"
    if dst.exists():
        return 0
    est = pd.read_parquet(R / "est" / _S["name"] / f"{rep}.parquet")
    still = est[(est.stage5 >= 1) & (est.stage5 <= 5) & (est.move <= thr)]
    night, _ = load_with_rows(_S["name"], subject, rep, _S["ds"], _S["sync"])
    ne = night.labels.size
    res = {"epoch": still.epoch.to_numpy()}
    for W in (60, 120):
        sp, fr = A.spectra(night.acc25, night.valid25, ne, W)
        res["freqs"] = fr.astype(np.float32)
        if sp is None:
            continue
        k = still.epoch.to_numpy() + 1 - W // 30
        okk = k >= 0
        kk = np.where(okk, k, 0)
        for ty in ("sum", "mag"):
            S = sp[ty][kk]; S[~okk] = np.nan
            res[f"S_{ty}_{W}"] = S.astype(np.float32)
        res[f"dev{W}"] = still[f"dev{W}"].to_numpy()
    np.savez_compressed(dst, **res)
    return 1


if __name__ == "__main__":
    from common2 import load_est
    from bpm_common import source_tables
    from splits import split0
    sp = split0()
    for name in COH:
        sel = set(source_tables()[name].subject)
        ids, rep = subject_ids(COHORTS[name])
        # movement cutpoint: fold-0 training sleep epochs
        tr = sp[name][0]
        mv = pd.concat([pd.read_parquet(R / "est" / name / f"{r}.parquet", columns=["subject", "stage5", "move"]) for r in tr])
        thr = mv[(mv.stage5 >= 1) & (mv.stage5 <= 5)].move.quantile(STILL_Q)
        todo = [(s, rep[s], thr, R / "still" / name) for s in ids if rep[s] in sel]
        (R / "still" / name).mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(name,)) as pool:
            list(pool.map(_one, todo))
        print(name, len(todo), flush=True)

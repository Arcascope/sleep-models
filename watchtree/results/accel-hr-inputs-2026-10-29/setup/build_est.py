"""Per night: autocorrelation curves for arms A-E, per-window peak and forward-only Viterbi estimates (emission 10 r, lam in 0.5/1/2).
Writes $N/est/<cohort>/<night>.parquet keyed (subject, epoch); also $N/dt/<cohort>/<night>.npy (timestamp-step sample) and, for the
resampling check, $N/psd/<cohort>/<night>.npz (per-window NUFFT PSDs of the raw samples and the 25 Hz stream, still PSG-sleep windows)."""
import os, sys, warnings, zlib
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np, pandas as pd
import ahr2 as A
import nativelib as NL
from common2 import R
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.uncropped import RAW_ROOT, sync_table
from loadnight import load_with_rows
N = Path(os.environ["N"]); _S = {}
PER_NIGHT = {"sleepaccel": 8, "wearanize": 2}


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    c = COHORTS[name]
    _S.update(name=name, ds=load_dataset(DATA_ROOT / c.directory), sync=sync_table(DATA_ROOT / RAW_ROOT) if name == "wearanize" else None)


def _psd_windows(rows, night, meta, name, k_pick, rng):
    """NUFFT PSD (per axis, summed) of the raw samples on their own timestamps and of the valid 25 Hz samples, windows = epochs."""
    from senpy import compute_nufft_spectrogram
    t = rows[:, 0]; out = {"raw": [], "r25": [], "epoch": []}
    t25 = night.t0 + np.arange(night.labels.size * 750) / 25.0
    for k in k_pick:
        a, b = night.t0 + 30.0 * k, night.t0 + 30.0 * (k + 1)
        i0, i1 = np.searchsorted(t, [a, b]); m25 = np.arange(k * 750, (k + 1) * 750)
        m25 = m25[night.valid25[m25]]
        if i1 - i0 < 0.8 * 30 * NL_FS[0] or m25.size < 600:
            continue
        S = {}
        for key, tt, X in (("raw", t[i0:i1], rows[i0:i1, 1:4]), ("r25", t25[m25], night.acc25[m25])):
            P = 0
            for ax in range(3):
                r = compute_nufft_spectrogram(tt, X[:, ax], 30.0, 0.0, kind="psd", origin_s=a, empty_windows="keep", min_samples=100)
                if not r.valid[0]:
                    P = None; break
                sel = r.frequencies <= 12.5; P = P + r.Sxx[0][sel]; f = r.frequencies[sel]
            S[key] = (f, P)
        if S["raw"][1] is None or S["r25"][1] is None:
            continue
        out["raw"].append(S["raw"][1]); out["r25"].append(S["r25"][1]); out["epoch"].append(k); out["f_raw"] = S["raw"][0]; out["f_25"] = S["r25"][0]
    return out


NL_FS = [0]


def _one(args):
    subject, rep, name = args
    dst = N / "est" / name / f"{rep}.parquet"
    if dst.exists():
        return 0
    m = pd.read_parquet(R / "est" / name / f"{rep}.parquet", columns=["epoch", "stage5", "move"])
    night, rows = load_with_rows(name, subject, rep, _S["ds"], _S["sync"])
    ne = night.labels.size
    C = NL.curves_25(night.acc25, night.valid25, ne)
    Cn, fs = NL.curves_native(rows, night.t0, ne); C.update(Cn); NL_FS[0] = fs
    ep = m.epoch.to_numpy(); d = {"subject": rep, "epoch": ep}
    for k, c in C.items():
        pk = np.where(np.isfinite(c).all(axis=1), A.GRID_BPM[np.nanargmax(np.where(np.isfinite(c), c, -9), axis=1)], np.nan)
        d[f"pk.{k}"] = pk[ep].astype(np.float32)
        for lam in NL.LAMS:
            bp, _ = A.viterbi_filter(10.0 * c, lam)
            d[f"est.{k}_l{NL.lam_tag(lam)}"] = bp[ep].astype(np.float32)
    # timestamp steps and PSD windows
    for sub in ("dt", "psd", "est"):
        (N / sub / name).mkdir(parents=True, exist_ok=True)
    t = rows[:, 0]; dt = np.diff(t); rng = np.random.default_rng(zlib.crc32(rep.encode()))
    np.save(N / "dt" / name / f"{rep}.npy", np.concatenate([[fs, dt.size, (dt > 1.0).sum(), (np.abs(dt * fs - 1) > 0.01).sum()], rng.choice(dt, min(dt.size, 20000), replace=False)]))
    sl = np.flatnonzero(((m.stage5 >= 1) & (m.stage5 <= 5)).to_numpy())
    if sl.size:
        mv = m["move"].to_numpy()[sl]; still = sl[mv <= np.quantile(mv, 0.25)]
        pick = np.sort(rng.choice(still, min(PER_NIGHT[name], still.size), replace=False))
        P = _psd_windows(rows, night, None, name, ep[pick], rng)
        if P["epoch"]:
            np.savez(N / "psd" / name / f"{rep}.npz", raw=np.array(P["raw"]), r25=np.array(P["r25"]), epoch=np.array(P["epoch"]), f_raw=P["f_raw"], f_25=P["f_25"])
    pd.DataFrame(d).to_parquet(dst.with_suffix(".tmp"), index=False); dst.with_suffix(".tmp").rename(dst)
    return 1


if __name__ == "__main__":
    from bpm_common import source_tables
    for name in sys.argv[1].split(","):
        sel = set(source_tables()[name].subject)
        ids, rep = subject_ids(COHORTS[name])
        todo = [(s, rep[s], name) for s in ids if rep[s] in sel]
        if len(sys.argv) > 2: todo = todo[:int(sys.argv[2])]
        with ProcessPoolExecutor(int(os.environ.get("NPROC", 12)), mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(name,)) as pool:
            list(pool.map(_one, todo))
        print(name, len(todo), flush=True)

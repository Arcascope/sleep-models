"""ahr_* columns from the best estimator (--combo, e.g. acf.acfvd8.0.30): one estimate per scored epoch (window ending at the epoch end).
ahr_bpm, ahr_q: the estimate and its quality (autocorrelation value at the tracked rate) at the epoch; values with quality < --keep
are NaN (gate 0 = none). The 5- and 10-min variability columns (bpm_features.night_features, prefix "ahr", cohort "accelhr")
are computed on the series of non-NaN estimates at their epoch-end times (e - 1 ms, data before e only); the series has dt 30 s,
so the spectral columns and sample entropy are NaN by that function's own rules."""
import argparse
import numpy as np, pandas as pd
from common2 import R, COH
from bpm_common import SRCS
import bpm_features2 as BF
ap = argparse.ArgumentParser(); ap.add_argument("--combo", required=True); ap.add_argument("--keep", type=float, default=1.0)
ap.add_argument("--out", default="features_ahr"); a = ap.parse_args()
from splits import split0
sp = split0()


def gate_of(name):
    """Quality value that keeps the top --keep share of the cohort's fold-0 training-participant sleep epochs (1 = no gate)."""
    if a.keep >= 1:
        return -np.inf
    parts = []
    for r in sp[name][0]:
        d = pd.read_parquet(R / "est" / name / f"{r}.parquet", columns=["stage5"]); e3 = pd.read_parquet(R / "est3" / name / f"{r}.parquet")
        parts.append(e3.loc[(d.stage5 >= 1).to_numpy() & (d.stage5 <= 5).to_numpy(), f"q.{a.combo}"])
    g = float(pd.concat(parts).dropna().quantile(1 - a.keep))
    print(name, "gate", g, flush=True)
    return g


for name in SRCS:
    gate = gate_of(name)
    out = R / a.out / name; out.mkdir(parents=True, exist_ok=True)
    e2 = {p.stem: p for p in (R / "est2" / name).glob("*.parquet")}
    for p in sorted((R / "est" / name).glob("*.parquet")):
        dst = out / p.name
        if dst.exists():
            continue
        d = pd.read_parquet(p)
        e3 = pd.read_parquet(R / "est3" / name / p.name)
        bpm = e3[f"est.{a.combo}"].to_numpy(float); q = e3[f"q.{a.combo}"].to_numpy(float)
        bpm = np.where(q >= gate, bpm, np.nan)
        ep = d.epoch.to_numpy(); ne = int(ep.max()) + 1
        t = (ep + 1) * 30.0 - 1e-3
        f = BF.night_features(t, bpm, ne, ep, "accelhr", "ahr")
        f.insert(0, "ahr_q", q.astype(np.float32)); f.insert(0, "ahr_bpm", bpm.astype(np.float32))
        f.insert(0, "y4", d.y4.to_numpy()); f.insert(0, "subject", p.stem)
        f.to_parquet(dst, index=False)
    print(name, len(list(out.glob("*.parquet"))), flush=True)

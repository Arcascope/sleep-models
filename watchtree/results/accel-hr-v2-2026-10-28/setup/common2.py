import os, glob
import numpy as np, pandas as pd
from pathlib import Path
R = Path(os.environ["R"])
COH = ("sleepaccel", "weaver_fitbit", "weaver_garmin", "wearanize")
STILL_Q = 0.10


def load_est(cohorts=COH):
    from splits import split0
    sp = split0(); parts = []
    for n in cohorts:
        d = pd.concat([pd.read_parquet(p) for p in sorted((R / "est" / n).glob("*.parquet"))], ignore_index=True)
        e2 = pd.concat([pd.read_parquet(p) for p in sorted((R / "est2" / n).glob("*.parquet"))], ignore_index=True) if (R / "est2" / n).exists() else None
        if e2 is not None:
            d = d.merge(e2, on=["subject", "epoch"], how="left")
        e3 = pd.concat([pd.read_parquet(p) for p in sorted((R / "est3" / n).glob("*.parquet"))], ignore_index=True) if (R / "est3" / n).exists() else None
        if e3 is not None:
            d = d.merge(e3, on=["subject", "epoch"], how="left")
        d["cohort"] = n
        tr, te = sp[n]
        d["split"] = np.where(d.subject.isin(te), "test", np.where(d.subject.isin(tr), "train", "none"))
        d["sleep"] = (d.stage5 >= 1) & (d.stage5 <= 5)
        thr = d.loc[(d.split == "train") & d.sleep, "move"].quantile(STILL_Q)
        d["still"] = d.sleep & (d.move <= thr)
        d["thr_still"] = thr
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def combos(d):
    return [c[4:] for c in d.columns if c.startswith("est.")]


def mae_row(est, ref, const):
    ok = est.notna() & ref.notna()
    e = (est - ref)[ok]
    return dict(n=int(ok.sum()), mae=e.abs().mean() if ok.any() else np.nan, within5=(e.abs() <= 5).mean() if ok.any() else np.nan,
                bias=e.mean() if ok.any() else np.nan, const_mae=(ref[ok] - const).abs().mean() if ok.any() else np.nan)

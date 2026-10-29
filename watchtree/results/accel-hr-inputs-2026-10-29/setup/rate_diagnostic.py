"""SleepAccel all-sleep MAE (bpm, lambda 1) of A, D, E by how far the night's effective sampling rate (1 / median step) is from the nominal integer rate.
Groups, no night ids. Writes sleepaccel_rate_diagnostic.csv."""
import os, glob
import numpy as np, pandas as pd
from splits import night_folds
N = os.environ["N"]; R = os.environ["R"]
f = set(night_folds().query("cohort == 'sleepaccel'").subject); rows = []
for p in glob.glob(f"{N}/dt/sleepaccel/*.npy"):
    s = os.path.basename(p)[:-4]
    if s not in f: continue
    z = np.load(p); e = pd.read_parquet(f"{N}/est/sleepaccel/{s}.parquet"); m = pd.read_parquet(f"{R}/est/sleepaccel/{s}.parquet", columns=["stage5", "dev30"])
    sl = ((m.stage5 >= 1) & (m.stage5 <= 5)).to_numpy(); eff = 1 / np.median(z[4:]); dev = abs(eff / z[0] - 1)
    rows.append(dict(nominal=int(z[0]), rate_off=dev, shift_step_share=z[3] / z[1], n=sl.sum(), **{k: np.nanmean(abs(e[f"est.{k}_l1p0"] - m.dev30)[sl]) for k in "ADE"},
                     ok=(~np.isnan(e["est.A_l1p0"]) & sl & m.dev30.notna()).sum()))
d = pd.DataFrame(rows)
d["group"] = np.where(d.rate_off <= 0.002, "effective rate within 0.2% of nominal", "effective rate more than 0.2% from nominal")
out = d.groupby("group").apply(lambda g: pd.Series(dict(nights=len(g), A_mae=np.average(g.A, weights=g.n), D_mae=np.average(g.D, weights=g.n), E_mae=np.average(g.E, weights=g.n))), include_groups=False).reset_index()
out.to_csv(f"{N}/sleepaccel_rate_diagnostic.csv", index=False); print(out.round(2).to_string(index=False)); print(d.rate_off.describe())

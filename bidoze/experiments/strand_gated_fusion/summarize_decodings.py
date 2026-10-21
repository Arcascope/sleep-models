"""Tables for the two stage-divisor refits: per design, draw, method and decoding; paired against G x3."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

fits = Path(sys.argv[1])
out = fits.parent
COH = {"testSleepAccel": "SleepAccel", "testClinical": "Clinical"}
STG = ["wake", "light", "deep", "rem"]
frames = []
for csv in sorted(fits.glob("*/*/draw*.csv")):
    df = pd.read_csv(csv)
    df["family"], df["fit"] = csv.parts[-3], csv.parts[-2]
    frames.append(df)
df = pd.concat(frames)
df["cohort"] = df.dataset.map(COH)
keys = ["family", "fit", "draw", "method", "decoding", "cohort"]
agg = df.groupby(keys).agg(kappa_4stage=("kappa_4stage", "mean"), kappa_sleep_wake=("kappa_sleep_wake", "mean"),
                           **{f"hits{i}": (f"hits{i}", "sum") for i in range(4)},
                           **{f"total{i}": (f"total{i}", "sum") for i in range(4)}).reset_index()
for i, s in enumerate(STG):
    agg[f"recall_{s}"] = agg[f"hits{i}"] / agg[f"total{i}"]
agg = agg.drop(columns=[f"hits{i}" for i in range(4)] + [f"total{i}" for i in range(4)])
agg.to_csv(out / "decodings_per_draw.csv", index=False)
V = ["kappa_4stage", "kappa_sleep_wake"] + [f"recall_{s}" for s in STG]
g3 = agg[agg.family == "g3"].set_index(["draw", "method", "decoding", "cohort"])[V]
# paired differences against G x3 (its mean of probabilities under the same decoding), and against the mean
rows = []
for (fam, fit, method, dec, coh), g in agg.groupby(["family", "fit", "method", "decoding", "cohort"]):
    g = g.set_index("draw")
    r = dict(family=fam, fit=fit, method=method, decoding=dec, cohort=coh, n_draws=len(g))
    for v in V:
        r[v] = g[v].mean()
        if fam != "g3":
            ref = g3.xs(("mean", dec, coh), level=(1, 2, 3))[v].reindex(g.index)
            d = (g[v] - ref).dropna()
            r[f"{v}_minus_g3"], r[f"{v}_minus_g3_sd"] = d.mean(), d.std()
    rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(out / "decodings_summary.csv", index=False)
# the mean's own change from the original divisor
orig = agg[(agg.method == "mean") & (agg.decoding == "orig")].set_index(["family", "fit", "draw", "cohort"])
chg = []
for dec in ("prior", "kappa"):
    x = agg[(agg.method == "mean") & (agg.decoding == dec)].set_index(["family", "fit", "draw", "cohort"])
    d = (x["kappa_4stage"] - orig["kappa_4stage"]).reset_index()
    d["decoding"] = dec
    chg.append(d)
chg = pd.concat(chg)
chg.groupby(["family", "fit", "decoding", "cohort"]).kappa_4stage.agg(["mean", "std", "min", "max"]).reset_index().to_csv(
    out / "mean_kappa_change_with_refit.csv", index=False)

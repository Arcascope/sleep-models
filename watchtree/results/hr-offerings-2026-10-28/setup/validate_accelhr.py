"""Accelerometer-derived HR against device HR: bpm MAE and coverage over all scored epochs, by stage and movement quartile.

Epoch-level rows come from build_accelhr.py. Per cohort and window W: estimate = window ending at the epoch end; device =
mean device HR (30-200 bpm) over the same window [e - W, e) (>= 1 sample). Coverage = share of scored epochs with an
estimate (accelerometer valid) and a device value. Movement quartiles are cutpoints of the epoch jerk within each cohort.
Also written: the MAE of a constant prediction (the cohort's median device HR) on the same epochs, and MAE by quality decile.
No averaging of spectra; the only aggregation is over epochs.
"""
import numpy as np, pandas as pd
from bpm_common import OUT, SRCS

STAGES = {0: "wake", 1: "light", 2: "deep", 3: "REM"}
rows = []
for n in SRCS:
    p = sorted((OUT / "accelhr" / n / "epochs").glob("*.parquet"))
    d = pd.concat([pd.read_parquet(x) for x in p], ignore_index=True)
    d["cohort"] = n
    d["mq"] = pd.qcut(d.move, 4, labels=["Q1 (least movement)", "Q2", "Q3", "Q4 (most movement)"])
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
D.drop(columns="subject").to_parquet(OUT / "accelhr_epochs_all.parquet")  # no night ids in the shared copy
res_c, res_s, res_m, res_q = [], [], [], []


def stats(g, W):
    e = g[f"est{W}"] - g[f"dev{W}"]
    ok = e.notna()
    const = np.abs(g.loc[ok, f"dev{W}"] - g[f"dev{W}"].median()).mean()
    return {"epochs": len(g), "nights": g.subject.nunique(), "coverage_est": g[f"est{W}"].notna().mean(),
            "coverage_dev": (g[f"ndev{W}"] > 0).mean(), "coverage_both": ok.mean(), "n_compared": int(ok.sum()),
            "mae": e[ok].abs().mean() if ok.any() else np.nan, "median_ae": e[ok].abs().median() if ok.any() else np.nan,
            "bias": e[ok].mean() if ok.any() else np.nan, "within_5bpm": (e[ok].abs() <= 5).mean() if ok.any() else np.nan,
            "const_mae": const if ok.any() else np.nan, "est_median": g[f"est{W}"].median(), "dev_median": g[f"dev{W}"].median(),
            "quality_median": g[f"q{W}"].median()}


for n, g in D.groupby("cohort"):
    for W in (10, 20, 30):
        res_c.append({"cohort": n, "window_s": W} | stats(g, W))
        for k, name in STAGES.items():
            res_s.append({"cohort": n, "window_s": W, "stage": name} | stats(g[g.y4 == k], W))
        for q, gg in g.groupby("mq", observed=True):
            res_m.append({"cohort": n, "window_s": W, "movement_quartile": q} | stats(gg, W))
        gq = g[g[f"est{W}"].notna() & (g[f"ndev{W}"] > 0)].copy()
        gq["qd"] = pd.qcut(gq[f"q{W}"], 10, labels=False, duplicates="drop") + 1
        for q, gg in gq.groupby("qd"):
            res_q.append({"cohort": n, "window_s": W, "quality_decile": int(q), "quality_lo": gg[f"q{W}"].min(),
                          "quality_hi": gg[f"q{W}"].max()} | stats(gg, W))
for name, r in (("by_cohort", res_c), ("by_stage", res_s), ("by_movement_quartile", res_m), ("by_quality_decile", res_q)):
    pd.DataFrame(r).to_csv(OUT / f"accelhr_validation_{name}.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(res_c).round(3).to_string())

"""Tables from the per-fit outputs of fuse.py: per draw and per design, means, gate, paired differences."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

fits = Path(sys.argv[1])
out = fits.parent
M = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity"]
COH = {"testSleepAccel": "SleepAccel", "testClinical": "Clinical"}
rows, infos = [], []
for csv in sorted(fits.glob("*/*/draw*.csv")):
    df = pd.read_csv(csv)
    df["family"], df["fit"] = csv.parts[-3], csv.parts[-2]
    rows.append(df)
    infos.append(json.loads(csv.with_suffix(".json").read_text()))
df = pd.concat(rows)
df["cohort"] = df.dataset.map(COH)
per = df.groupby(["family", "fit", "draw", "method", "cohort"])[M].mean().reset_index()
per.to_csv(out / "per_draw_metrics.csv", index=False)
pd.DataFrame([{k: v for k, v in i.items() if k not in ("weights_by_stage", "cv_loss", "epochs_by_stage")}
              for i in infos]).to_csv(out / "fit_info.csv", index=False)

g3 = per[(per.family == "g3") & (per.method == "mean")].set_index(["draw", "cohort"])[M]
lines = []
def md(t):
    t = t.round(3)
    return "\n".join(["| " + " | ".join(map(str, t.columns)) + " |", "|" + ":---|" * len(t.columns)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in t.to_numpy()])

summ = []
for (fam, fit), g in per.groupby(["family", "fit"]):
    m = g[g.method == "mean"].set_index(["draw", "cohort"])[M]
    ga = g[g.method == "gate"].set_index(["draw", "cohort"])[M]
    for coh in COH.values():
        for k in M:
            r = dict(family=fam, fit=fit, cohort=coh, metric=k, n_draws=len(m.xs(coh, level=1)))
            mm, gg = m.xs(coh, level=1)[k], ga.xs(coh, level=1)[k]
            r.update(mean=mm.mean(), gate=gg.mean())
            d = gg - mm
            r.update(gate_minus_mean=d.mean(), gate_minus_mean_sd=d.std())
            r["draws_gate_higher"] = int((d > 0).sum())
            if fam != "g3":
                gref = g3.xs(coh, level=1)[k].reindex(mm.index)
                for name, x in (("mean", mm), ("gate", gg)):
                    dd = (x - gref).dropna()
                    r[f"{name}_minus_g3"], r[f"{name}_minus_g3_sd"] = dd.mean(), dd.std()
            summ.append(r)
pd.DataFrame(summ).to_csv(out / "paired_summary.csv", index=False)

# gate weights by stage
wrows = []
for i in infos:
    for coh, stg in i["weights_by_stage"].items():
        for st, w in stg.items():
            for s, x in enumerate(w):
                wrows.append(dict(family=i["family"], fit=i["design"], draw=i["draw"], cohort=COH[coh],
                                  stage=["wake", "light", "deep", "rem"][int(st)], strand=s, weight=x,
                                  uniform=1 / i["strands"]))
w = pd.DataFrame(wrows)
w.to_csv(out / "gate_weights_by_stage.csv", index=False)

# markdown tables
def table(t):
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}".replace("-", "−"))
    return md(t.astype(object)).replace("−", "−")

parts = ["# Strand-gated fusion tables\n"]
short = {"kappa_4stage": "κ4", "kappa_sleep_wake": "κ s/w", "sleep_sensitivity": "sens", "wake_specificity": "spec"}
for (fam, fit), g in per.groupby(["family", "fit"]):
    parts.append(f"\n## {fam} {fit}: per draw (mean over test recordings; gate vs mean of probabilities)\n")
    t = g[g.method.isin(["mean", "gate"])].pivot_table(index=["draw", "cohort"], columns="method", values=M)
    t.columns = [f"{short[a]} {b}" for a, b in t.columns]
    t = t[[f"{short[k]} {m}" for k in M for m in ("mean", "gate")]].reset_index()
    parts.append(table(t))
s = pd.DataFrame(summ)
parts.append("\n## Paired summary over draws (gate − mean; mean and gate − BiDoze-G ×3 of the same draw)\n")
parts.append(table(s))
parts.append("\n## Gate weights by PSG stage (test epochs, mean over draws of per-fit values; strand order is arbitrary per fit)\n")
mw = w.groupby(["family", "fit", "cohort", "stage", "strand"]).weight.mean().reset_index()
parts.append(table(mw.pivot_table(index=["family", "fit", "cohort", "strand"], columns="stage", values="weight")[["wake", "light", "deep", "rem"]].reset_index()))
(out / "tables.md").write_text("\n".join(parts))

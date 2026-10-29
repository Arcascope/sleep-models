"""Step 3: coverage vs accuracy as a function of a quality gate, all PSG-sleep epochs (any movement), test participants.
Candidates: per cohort and method family, the combination with the lowest training-participant MAE in the stillest decile.
Gate levels are quantiles of the training participants' sleep-epoch quality (keep the top 100/50/20/10/5 %)."""
import os
import numpy as np, pandas as pd
from common2 import *
T = pd.read_csv(R / "step2_all_combos.csv")
d = load_est()
FAM = {"peak": ("peak",), "hsum": ("hsum",), "hps": ("hps",), "sacf": ("sacf",), "acf": ("acf3", "acf8"), "acf_curve": ("acfg3", "acfg8"),
       "viterbi_spec": ("vita", "vitb", "vitc"), "viterbi_acf": tuple(f"acfv{t}{b}" for t in "abcdef" for b in (3, 8))}
KEEP = (1.0, 0.5, 0.2, 0.1, 0.05)
cand = {}
for n in COH:
    t = T[(T.cohort == n) & (T.split == "train")].dropna(subset=["mae"])
    for fam, pre in FAM.items():
        tt = t[t.method.isin(pre)]
        if len(tt):
            r = tt.sort_values("mae").iloc[0]
            cand[(n, fam)] = f"{r.type}.{r.method}.{int(r.whiten)}.{int(r.W)}"
# one estimator for all cohorts: lowest mean training-participant MAE (stillest decile) over the two CV cohorts, Wearanize+ and SleepAccel
tt = T[(T.split == "train") & T.cohort.isin(["wearanize", "sleepaccel"])].dropna(subset=["mae"])
tt = tt.assign(combo=tt.type + "." + tt.method + "." + tt.whiten.astype(str) + "." + tt.W.astype(str))
best = tt.groupby("combo").mae.mean().sort_values()
best.head(15).to_csv(R / "step3_chosen_rank.csv")
CHOSEN = best.index[0]
print("chosen", CHOSEN)
for n in COH:
    cand[(n, "chosen")] = CHOSEN
pd.Series({f"{k[0]}|{k[1]}": v for k, v in cand.items()}).to_csv(R / "step3_candidates.csv")
rows, strat = [], []
for (n, fam), c in cand.items():
    W = int(c.split(".")[-1])
    g = d[(d.cohort == n) & d.sleep]
    const = g[(g.split == "train")][f"dev{W}"].median()
    thr_q = g[g.split == "train"][f"q.{c}"].dropna()
    te = g[g.split == "test"].copy()
    te["est"], te["q"], te["ref"] = te[f"est.{c}"], te[f"q.{c}"], te[f"dev{W}"]
    nall = len(te)
    mq = pd.qcut(g[g.split == "train"].move, 4, retbins=True)[1]
    te["mq"] = pd.cut(te.move, [-np.inf, *mq[1:-1], np.inf], labels=["Q1 (least movement)", "Q2", "Q3", "Q4 (most movement)"])
    te["stage"] = te.y4.map({0: "wake", 1: "light", 2: "deep", 3: "REM"})
    for keep in KEEP:
        cut = thr_q.quantile(1 - keep) if keep < 1 else -np.inf
        sel = te[(te.q >= cut) & te.est.notna() & te.ref.notna()]
        e = (sel.est - sel.ref)
        rows.append(dict(cohort=n, family=fam, combo=c, keep_train_quantile=keep, share_all_sleep_epochs=len(sel) / nall,
                         n=len(sel), participants=sel.subject.nunique(), mae=e.abs().mean(), within5=(e.abs() <= 5).mean(),
                         bias=e.mean(), const_mae=(sel.ref - const).abs().mean()))
        if keep in (1.0, 0.2, 0.05):
            for key in ("stage", "mq"):
                for lv, s2 in sel.groupby(key, observed=True):
                    e2 = s2.est - s2.ref
                    nlv = (te[key] == lv).sum()
                    strat.append(dict(cohort=n, family=fam, combo=c, keep_train_quantile=keep, by=key, level=lv, n=len(s2),
                                      share_of_level=len(s2) / nlv, mae=e2.abs().mean(), within5=(e2.abs() <= 5).mean(),
                                      const_mae=(s2.ref - const).abs().mean()))
pd.DataFrame(rows).to_csv(R / "step3_gate_curve.csv", index=False)
pd.DataFrame(strat).to_csv(R / "step3_gate_by_stage_movement.csv", index=False)
pd.set_option("display.width", 250)
r = pd.DataFrame(rows)
print(r[r.keep_train_quantile.isin([1.0, 0.2, 0.05])].round(3).to_string())

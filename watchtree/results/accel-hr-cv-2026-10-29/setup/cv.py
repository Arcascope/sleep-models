"""5-fold CV (BiDoze SleepAccel test folds) of the accelerometer HR estimator candidates; uses the per-night estimates of accel-hr-v2-2026-10-28."""
import os
import numpy as np, pandas as pd
from pathlib import Path
from common2 import combos, load_est, STILL_Q
from splits import night_folds
N = Path(os.environ["N"]); rng = np.random.default_rng(0)
d = load_est(("sleepaccel",))
f = night_folds(); f = f[f.cohort == "sleepaccel"].set_index("subject").fold
d["fold"] = d.subject.map(f)
d = d[d.fold.notna()].copy(); d["fold"] = d.fold.astype(int)
cs = combos(d); FIX = ["acf.acfvd8.0.30", "acf.acfvc8.0.30"]
W = lambda c: int(c.split(".")[-1])
print("nights", d.subject.nunique(), "per fold", d.groupby("fold").subject.nunique().to_dict())

def mae_of(c, g):
    ok = g[f"est.{c}"].notna() & g[f"dev{W(c)}"].notna()
    return (g.loc[ok, f"est.{c}"] - g.loc[ok, f"dev{W(c)}"]).abs().mean()

def run_arm(name, picker):
    """returns long frame of out-of-fold sleep epochs with est, ref, const, still, fold, subject, cand"""
    parts = []
    for k in range(5):
        tr = d[(d.fold != k) & d.sleep]; te = d[(d.fold == k) & d.sleep].copy()
        thr = tr.move.quantile(STILL_Q); trs = tr[tr.move <= thr]
        c = picker(tr, trs)
        te["cand"] = c; te["est"] = te[f"est.{c}"]; te["ref"] = te[f"dev{W(c)}"]
        te["const"] = tr[f"dev{W(c)}"].median(); te["still"] = te.move <= thr
        parts.append(te[["subject", "epoch", "fold", "cand", "est", "ref", "const", "still", "stage5"]])
    return pd.concat(parts, ignore_index=True)

arms = {}
sel_rows = []
for k in range(5):
    tr = d[(d.fold != k) & d.sleep]; trs = tr[tr.move <= tr.move.quantile(STILL_Q)]
    m_still = pd.Series({c: mae_of(c, trs) for c in cs}).dropna().sort_values()
    m_all = pd.Series({c: mae_of(c, tr) for c in cs}).dropna().sort_values()
    sel_rows.append(dict(fold=k, train_nights=tr.subject.nunique(), test_nights=d[d.fold == k].subject.nunique(),
        pick_still=m_still.index[0], mae_still=m_still.iloc[0], pick_all=m_all.index[0], mae_all=m_all.iloc[0],
        runnerup_still=m_still.index[1], runnerup_still_mae=m_still.iloc[1],
        train_still_mae_acfvd8=m_still[FIX[0]], train_still_mae_acfvc8=m_still[FIX[1]],
        train_all_mae_acfvd8=m_all[FIX[0]], train_all_mae_acfvc8=m_all[FIX[1]]))
sel = pd.DataFrame(sel_rows); sel.to_csv(N / "cv_selection.csv", index=False)
pk = {"still": dict(zip(sel.fold, sel.pick_still)), "all": dict(zip(sel.fold, sel.pick_all))}
def picker_for(rule):
    state = {"i": -1}
    def p(tr, trs):
        state["i"] += 1; return pk[rule][state["i"]]
    return p
arms["select_still"] = run_arm("s", picker_for("still"))
arms["select_all"] = run_arm("a", picker_for("all"))
for c in FIX:
    arms[f"fixed_{c}"] = run_arm(c, lambda tr, trs, c=c: c)

def stats(g):
    ok = g.est.notna() & g.ref.notna(); e = (g.est - g.ref)[ok]; ec = (g.const - g.ref)[ok]
    return dict(n=int(ok.sum()), nights=g[ok].subject.nunique(), mae=e.abs().mean(), within5=(e.abs() <= 5).mean(), bias=e.mean(),
                const_mae=ec.abs().mean(), const_within5=(ec.abs() <= 5).mean(), const_bias=ec.mean(), diff_mae=e.abs().mean() - ec.abs().mean())

def boot(g, B=5000):
    ok = g.est.notna() & g.ref.notna(); g = g[ok]
    a = g.assign(ae=(g.est - g.ref).abs(), ce=(g.const - g.ref).abs()).groupby("subject").agg(sa=("ae", "sum"), sc=("ce", "sum"), n=("ae", "size"))
    m = len(a); idx = rng.integers(0, m, (B, m))
    sa, sc, n = a.sa.values[idx].sum(1), a.sc.values[idx].sum(1), a.n.values[idx].sum(1)
    q = lambda x: np.percentile(x, [2.5, 97.5])
    return dict(mae_lo=q(sa / n)[0], mae_hi=q(sa / n)[1], const_mae_lo=q(sc / n)[0], const_mae_hi=q(sc / n)[1],
                diff_lo=q((sa - sc) / n)[0], diff_hi=q((sa - sc) / n)[1])

pooled, perfold, blocks, level = [], [], [], []
for name, o in arms.items():
    for sub, g in (("all_sleep", o), ("stillest_decile", o[o.still])):
        pooled.append(dict(arm=name, subset=sub, **stats(g), **boot(g)))
        for k, gk in g.groupby("fold"):
            perfold.append(dict(arm=name, subset=sub, fold=k, cand=gk.cand.iloc[0], **stats(gk)))
    # block averages (all PSG-sleep epochs)
    for mins, nb in ((0.5, 1), (1, 2), (5, 10), (10, 20)):
        g = o.dropna(subset=["est", "ref"]).copy(); g["blk"] = g.epoch // nb
        a = g.groupby(["subject", "blk"]).agg(est=("est", "mean"), ref=("ref", "mean"), k=("est", "size")).reset_index()
        a = a[a.k >= 0.8 * nb]
        e = a.est - a.ref
        a["er"] = a.est - a.groupby("subject").est.transform("median"); a["rr"] = a.ref - a.groupby("subject").ref.transform("median")
        blocks.append(dict(arm=name, block_min=mins, blocks=len(a), nights=a.subject.nunique(), mae=e.abs().mean(), within5=(e.abs() <= 5).mean(),
                           bias=e.mean(), within_night_r=np.corrcoef(a.er, a.rr)[0, 1]))
    nm = o.dropna(subset=["est", "ref"]).groupby("subject").agg(est=("est", "median"), ref=("ref", "median"))
    level.append(dict(arm=name, nights=len(nm), pearson_r=nm.est.corr(nm.ref), spearman_r=nm.est.corr(nm.ref, method="spearman")))
pd.DataFrame(pooled).to_csv(N / "cv_pooled.csv", index=False)
pd.DataFrame(perfold).to_csv(N / "cv_per_fold.csv", index=False)
pd.DataFrame(blocks).to_csv(N / "cv_blocks.csv", index=False)
pd.DataFrame(level).to_csv(N / "cv_night_level.csv", index=False)
pd.DataFrame([dict(nights=d.subject.nunique(), participants=d.subject.nunique(), epochs_sleep=int(d.sleep.sum()),
                   **{f"nights_fold{k}": int((f.reindex(d.subject.unique()) == k).sum()) for k in range(5)})]).to_csv(N / "cv_cohort.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for t in ("selection", "pooled", "per_fold", "blocks", "night_level", "cohort"):
    print(t); print(pd.read_csv(N / (f"cv_{t}.csv")).round(3).to_string())

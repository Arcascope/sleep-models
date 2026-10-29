"""5-fold CV (BiDoze test folds) of the envelope HR estimators vs the current one: per fold pick the candidate (band, cutoff, lambda) with the
lowest training MAE vs device HR over training PSG-sleep epochs ('all' rule) or over the training stillest decile ('still' rule); score out of fold.
usage: cv.py <cohort> (sleepaccel | wearanize); writes <cohort>_*.csv at the run dir top level. Bootstrap unit: night (SleepAccel: one night per
participant)."""
import os, sys
import numpy as np, pandas as pd
from pathlib import Path
from splits import night_folds
HP = os.environ.get("HP") == "1"; TAG = "hp_" if HP else ""
N = Path(os.environ["N"]); R = Path(os.environ["R"]); coh = sys.argv[1]; STILL_Q = 0.10; B = 5000
rng = np.random.default_rng(0)
f = night_folds(); f = f[f.cohort == coh].set_index("subject").fold
parts = []
for p in sorted((N / "est" / coh).glob("*.parquet")):
    s = p.stem
    if s not in f.index: continue
    e = pd.read_parquet(p)
    if HP: e = e.merge(pd.read_parquet(N / "est_hp" / coh / p.name), on=["subject", "epoch"])
    m = pd.read_parquet(R / "est" / coh / p.name, columns=["epoch", "stage5", "move", "dev30"])
    parts.append(e.merge(m, on="epoch"))
d = pd.concat(parts, ignore_index=True); d["fold"] = d.subject.map(f).astype(int)
d["sleep"] = (d.stage5 >= 1) & (d.stage5 <= 5)
FAM = {"hilhp": [c for c in d.columns if c.startswith("est.hilhp_")], "lphp": [c for c in d.columns if c.startswith("est.lphp_")]} if HP else \
      {"hil": [c for c in d.columns if c.startswith("est.hil_")], "lp": [c for c in d.columns if c.startswith("est.lp_")]}
CUR = "est.cur_l1p0"
pkcol = lambda c: "pk." + c[4:].rsplit("_l", 1)[0]
print(coh, "nights", d.subject.nunique(), {k: int(v) for k, v in d.groupby("fold").subject.nunique().items()}, "candidates", {k: len(v) for k, v in FAM.items()})

def mae(c, g):
    ok = g[c].notna() & g.dev30.notna(); return (g.loc[ok, c] - g.loc[ok, "dev30"]).abs().mean()

arms, sel_rows = {}, []
def make_arm(cands_by_fold):
    parts = []
    for k in range(5):
        tr = d[(d.fold != k) & d.sleep]; te = d[(d.fold == k) & d.sleep].copy(); thr = tr.move.quantile(STILL_Q)
        c = cands_by_fold[k]; ok = tr[c].notna() & tr.dev30.notna()
        te["cand"] = c; te["est"] = te[c]; te["pk"] = te[pkcol(c)]; te["ref"] = te.dev30; te["const"] = tr.dev30.median()
        te["train_bias"] = (tr.loc[ok, c] - tr.loc[ok, "dev30"]).mean(); te["still"] = te.move <= thr
        parts.append(te[["subject", "epoch", "fold", "cand", "est", "pk", "ref", "const", "train_bias", "still"]])
    return pd.concat(parts, ignore_index=True)
for fam, cs in FAM.items():
    pick = {"all": {}, "still": {}}
    for k in range(5):
        tr = d[(d.fold != k) & d.sleep]; trs = tr[tr.move <= tr.move.quantile(STILL_Q)]
        ma = pd.Series({c: mae(c, tr) for c in cs}).sort_values(); ms = pd.Series({c: mae(c, trs) for c in cs}).sort_values()
        pick["all"][k], pick["still"][k] = ma.index[0], ms.index[0]
        sel_rows.append(dict(estimator=fam, fold=k, train_nights=tr.subject.nunique(), test_nights=d[d.fold == k].subject.nunique(),
                             pick_all=ma.index[0][4:], train_mae_all=ma.iloc[0], pick_still=ms.index[0][4:], train_still_mae=ms.iloc[0],
                             train_mae_all_current=mae(CUR, tr), train_still_mae_current=mae(CUR, trs)))
    for rule in ("all", "still"):
        arms[f"{fam}_{rule}"] = make_arm(pick[rule])
arms["current"] = make_arm({k: CUR for k in range(5)})
cst = arms["current"].copy(); cst["est"] = cst.const; cst["pk"] = np.nan; cst["train_bias"] = 0.0; cst["cand"] = "const"; arms["const"] = cst
pd.DataFrame(sel_rows).to_csv(N / f"{TAG}{coh}_selection.csv", index=False)
cur = arms["current"].set_index(["subject", "epoch"])

def valid(g): return g.est.notna() & g.ref.notna()
def stats(g, bc=False):
    g = g[valid(g)]; e = g.est - g.ref - (g.train_bias if bc else 0)
    return dict(n=len(g), nights=g.subject.nunique(), mae=e.abs().mean(), within5=(e.abs() <= 5).mean(), bias=e.mean())

def boot(g, name):
    """participant bootstrap: MAE, bias-corrected MAE, paired MAE difference vs current on the epochs where both exist"""
    g = g[valid(g)].copy(); g["ae"] = (g.est - g.ref).abs(); g["be"] = (g.est - g.ref - g.train_bias).abs()
    a = g.groupby("subject").agg(sa=("ae", "sum"), sb=("be", "sum"), n=("ae", "size"))
    out = {}
    idx = rng.integers(0, len(a), (B, len(a)))
    n = a.n.values[idx].sum(1); q = lambda x: np.percentile(x, [2.5, 97.5])
    out["mae_lo"], out["mae_hi"] = q(a.sa.values[idx].sum(1) / n); out["bc_mae_lo"], out["bc_mae_hi"] = q(a.sb.values[idx].sum(1) / n)
    if name != "current":
        c = cur.reindex(pd.MultiIndex.from_frame(g[["subject", "epoch"]]))
        okc = c.est.notna().to_numpy(); h = g[okc].copy(); h["ce"] = (c.est.to_numpy()[okc] - c.ref.to_numpy()[okc]); h["ce"] = h.ce.abs()
        pa = h.groupby("subject").agg(sa=("ae", "sum"), sc=("ce", "sum"), n=("ae", "size")).reindex(a.index).fillna(0)
        j = idx; nn = pa.n.values[j].sum(1); dd = (pa.sa.values[j].sum(1) - pa.sc.values[j].sum(1)) / nn
        out["diff_mae"] = (pa.sa.sum() - pa.sc.sum()) / pa.n.sum(); out["diff_lo"], out["diff_hi"] = q(dd)
    return out

pooled, perfold, blocks, level, peaks = [], [], [], [], []
for name, o in arms.items():
    for sub, g in (("all_sleep", o), ("stillest_decile", o[o.still])):
        pooled.append(dict(arm=name, subset=sub, **stats(g), **{f"bc_{k}": v for k, v in stats(g, True).items() if k in ("mae", "within5", "bias")}, **boot(g, name)))
        for k, gk in g.groupby("fold"):
            perfold.append(dict(arm=name, subset=sub, fold=k, cand=gk.cand.iloc[0][4:], **stats(gk), bc_mae=stats(gk, True)["mae"]))
    for mins, nb in ((5, 10),):
        g = o[valid(o)].copy(); g["blk"] = g.epoch // nb
        a = g.groupby(["subject", "blk"]).agg(est=("est", "mean"), ref=("ref", "mean"), k=("est", "size")).reset_index(); a = a[a.k >= 0.8 * nb]
        e = a.est - a.ref
        a["er"] = a.est - a.groupby("subject").est.transform("median"); a["rr"] = a.ref - a.groupby("subject").ref.transform("median")
        blocks.append(dict(arm=name, block_min=mins, blocks=len(a), nights=a.subject.nunique(), mae=e.abs().mean(), within5=(e.abs() <= 5).mean(),
                           bias=e.mean(), within_night_r=np.corrcoef(a.er, a.rr)[0, 1]))
    nm = o[valid(o)].groupby("subject").agg(est=("est", "median"), ref=("ref", "median"))
    level.append(dict(arm=name, nights=len(nm), pearson_r=nm.est.corr(nm.ref), spearman_r=nm.est.corr(nm.ref, method="spearman")))
    if name != "const":
        p = o[o.pk.notna() & o.ref.notna()]
        peaks.append(dict(arm=name, windows=len(p), share_peak_ge150=(p.pk >= 150).mean(), share_peak_within3=((p.pk - p.ref).abs() <= 3).mean(),
                          share_peak_within3_still=((p.pk - p.ref).abs() <= 3)[p.still].mean()))
for nme, rows in (("pooled", pooled), ("per_fold", perfold), ("blocks", blocks), ("night_level", level), ("peaks", peaks)):
    pd.DataFrame(rows).to_csv(N / f"{TAG}{coh}_{nme}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for t in ("selection", "pooled", "per_fold", "blocks", "night_level", "peaks"):
    print(t); print(pd.read_csv(N / f"{TAG}{coh}_{t}.csv").round(3).to_string())

"""5-fold CV (BiDoze test folds) of the five input arms A-E (nativelib.py), tracker lambda 1 fixed ('_l1') or selected per fold from {0.5, 1, 2} by the
lowest training-fold PSG-sleep MAE vs device HR ('_sel'); scored out of fold. Paired differences are vs arm A in the same lambda mode.
usage: cv.py <cohort> (sleepaccel | wearanize); writes <cohort>_*.csv at the run dir top level. Bootstrap unit: participant (SleepAccel: one night
per participant; Wearanize+: the night id's participant prefix), 5,000 draws, seed 0."""
import os, sys
import numpy as np, pandas as pd
from pathlib import Path
from splits import night_folds
N = Path(os.environ["N"]); R = Path(os.environ["R"]); coh = sys.argv[1]; STILL_Q = 0.10; B = 5000
rng = np.random.default_rng(0)
ARMS = ("A", "B", "C", "D", "E"); LAMS = ("0p5", "1p0", "2p0")
f = night_folds(); f = f[f.cohort == coh].set_index("subject").fold
parts = []
for p in sorted((N / "est" / coh).glob("*.parquet")):
    s = p.stem
    if s not in f.index: continue
    e = pd.read_parquet(p)
    m = pd.read_parquet(R / "est" / coh / p.name, columns=["epoch", "stage5", "move", "dev30"])
    parts.append(e.merge(m, on="epoch"))
d = pd.concat(parts, ignore_index=True); d["fold"] = d.subject.map(f).astype(int)
d["part"] = d.subject.str.split("_").str[0] if coh == "wearanize" else d.subject
d["sleep"] = (d.stage5 >= 1) & (d.stage5 <= 5)
print(coh, "nights", d.subject.nunique(), "participants", d.part.nunique(), {k: int(v) for k, v in d.groupby("fold").subject.nunique().items()})

def mae(c, g):
    ok = g[c].notna() & g.dev30.notna(); return (g.loc[ok, c] - g.loc[ok, "dev30"]).abs().mean()

arms, sel_rows = {}, []
def make_arm(arm, cands_by_fold):
    out = []
    for k in range(5):
        tr = d[(d.fold != k) & d.sleep]; te = d[(d.fold == k) & d.sleep].copy(); thr = tr.move.quantile(STILL_Q)
        c = cands_by_fold[k]
        te["lam"] = c.rsplit("_l", 1)[1]; te["est"] = te[c]; te["pk"] = te[f"pk.{arm}"]; te["ref"] = te.dev30; te["still"] = te.move <= thr
        out.append(te[["subject", "part", "epoch", "fold", "lam", "est", "pk", "ref", "still"]])
    return pd.concat(out, ignore_index=True)
for arm in ARMS:
    pick = {}
    for k in range(5):
        tr = d[(d.fold != k) & d.sleep]
        ms = pd.Series({f"est.{arm}_l{l}": mae(f"est.{arm}_l{l}", tr) for l in LAMS}).sort_values()
        pick[k] = ms.index[0]
        sel_rows.append(dict(arm=arm, fold=k, train_nights=tr.subject.nunique(), test_nights=d[d.fold == k].subject.nunique(), lam_selected=ms.index[0].rsplit("_l", 1)[1].replace("p", "."),
                             **{f"train_mae_l{l}": ms[f"est.{arm}_l{l}"] for l in LAMS}))
    arms[f"{arm}_l1"] = make_arm(arm, {k: f"est.{arm}_l1p0" for k in range(5)})
    arms[f"{arm}_sel"] = make_arm(arm, pick)
pd.DataFrame(sel_rows).to_csv(N / f"{coh}_selection.csv", index=False)

def valid(g): return g.est.notna() & g.ref.notna()
def stats(g):
    g = g[valid(g)]; e = g.est - g.ref
    return dict(n=len(g), nights=g.subject.nunique(), participants=g.part.nunique(), mae=e.abs().mean(), within5=(e.abs() <= 5).mean(), bias=e.mean())

def boot(g, base):
    g = g[valid(g)].copy(); g["ae"] = (g.est - g.ref).abs()
    a = g.groupby("part").agg(sa=("ae", "sum"), n=("ae", "size"))
    idx = rng.integers(0, len(a), (B, len(a))); q = lambda x: np.percentile(x, [2.5, 97.5])
    out = {}; out["mae_lo"], out["mae_hi"] = q(a.sa.values[idx].sum(1) / a.n.values[idx].sum(1))
    if base is not None:
        c = base.set_index(["subject", "epoch"]).reindex(pd.MultiIndex.from_frame(g[["subject", "epoch"]]))
        okc = (c.est.notna() & c.ref.notna()).to_numpy(); h = g[okc].copy(); h["ce"] = (c.est.to_numpy()[okc] - c.ref.to_numpy()[okc]); h["ce"] = h.ce.abs()
        pa = h.groupby("part").agg(sa=("ae", "sum"), sc=("ce", "sum"), n=("ae", "size")).reindex(a.index).fillna(0)
        nn = pa.n.values[idx].sum(1); dd = (pa.sa.values[idx].sum(1) - pa.sc.values[idx].sum(1)) / nn
        out["diff_mae"] = (pa.sa.sum() - pa.sc.sum()) / pa.n.sum(); out["diff_lo"], out["diff_hi"] = q(dd)
    return out

pooled, perfold, blocks, level, peaks = [], [], [], [], []
for name, o in arms.items():
    base = arms[("A_l1" if name.endswith("_l1") else "A_sel")] if not name.startswith("A_") else None
    for sub, g in (("all_sleep", o), ("stillest_decile", o[o.still])):
        bg = None if base is None else (base if sub == "all_sleep" else base[base.still])
        pooled.append(dict(arm=name, subset=sub, **stats(g), **boot(g, bg)))
        for k, gk in g.groupby("fold"):
            perfold.append(dict(arm=name, subset=sub, fold=k, **stats(gk)))
    g = o[valid(o)].copy(); g["blk"] = g.epoch // 10
    a = g.groupby(["subject", "blk"]).agg(est=("est", "mean"), ref=("ref", "mean"), k=("est", "size")).reset_index(); a = a[a.k >= 8]
    a["er"] = a.est - a.groupby("subject").est.transform("median"); a["rr"] = a.ref - a.groupby("subject").ref.transform("median")
    blocks.append(dict(arm=name, blocks=len(a), nights=a.subject.nunique(), within_night_r=np.corrcoef(a.er, a.rr)[0, 1]))
    nm = o[valid(o)].groupby("subject").agg(est=("est", "median"), ref=("ref", "median"))
    level.append(dict(arm=name, nights=len(nm), pearson_r=nm.est.corr(nm.ref), spearman_r=nm.est.corr(nm.ref, method="spearman")))
    p = o[o.pk.notna() & o.ref.notna()]
    peaks.append(dict(arm=name, windows=len(p), share_peak_ge150=(p.pk >= 150).mean(), share_peak_within3=((p.pk - p.ref).abs() <= 3).mean()))
for nme, rows in (("pooled", pooled), ("per_fold", perfold), ("blocks", blocks), ("night_level", level), ("peaks", peaks)):
    pd.DataFrame(rows).to_csv(N / f"{coh}_{nme}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for t in ("selection", "pooled", "per_fold", "blocks", "night_level", "peaks"):
    print(t); print(pd.read_csv(N / f"{coh}_{t}.csv").round(3).to_string())

"""Out-of-fold bias correction of the accelerometer HR estimate (acf.acfvd8.0.30) on SleepAccel: in each BiDoze fold, subtract
the training nights' mean error (estimate - device HR over PSG-sleep epochs) from the test nights' estimates.
Writes bias_correction.csv (per fold and pooled) and bias_by_night_summary.csv; CIs: participant bootstrap, 5,000 draws."""
import glob, os
import numpy as np, pandas as pd
from splits import night_folds
R = os.environ["R"]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
f = night_folds(); f = f[f.cohort == "sleepaccel"].set_index("subject").fold
rows = []
for p in sorted(glob.glob(f"{R}/est/sleepaccel/*.parquet")):
    s = os.path.basename(p)[:-8]
    if s not in f.index:
        continue
    d = pd.read_parquet(p, columns=["epoch", "stage5", "dev30"])
    d["est"] = pd.read_parquet(p.replace("/est/", "/est3/"), columns=["est.acf.acfvd8.0.30"]).iloc[:, 0].to_numpy()
    d["s"] = s; d["fold"] = f[s]; rows.append(d)
D = pd.concat(rows); D = D[(D.stage5 >= 1) & (D.stage5 <= 5)].dropna(subset=["est", "dev30"]); D["err"] = D.est - D.dev30
nb = D.groupby("s").err.mean()
pd.DataFrame([dict(nights=len(nb), median=nb.median(), q25=nb.quantile(.25), q75=nb.quantile(.75), min=nb.min(), max=nb.max(),
                   nights_negative=int((nb < 0).sum()))]).to_csv(os.path.join(OUT, "bias_by_night_summary.csv"), index=False)
out, res = [], []
for k in range(5):
    tr, te = D[D.fold != k], D[D.fold == k].copy()
    c = tr.err.mean(); te["e2"] = te.err - c; out.append(te)
    res.append(dict(fold=k, train_bias=c, test_bias=te.err.mean(), test_bias_corrected=te.e2.mean(),
                    mae=te.err.abs().mean(), mae_corrected=te.e2.abs().mean()))
O = pd.concat(out); rng = np.random.default_rng(0)
agg = O.groupby("s").agg(n=("e2", "size"), b=("err", "sum"), b2=("e2", "sum"), a=("err", lambda v: v.abs().sum()), a2=("e2", lambda v: v.abs().sum()))
boot = np.array([agg.iloc[rng.choice(len(agg), len(agg))].pipe(lambda x: [x.b2.sum() / x.n.sum(), x.a2.sum() / x.n.sum()]) for _ in range(5000)])
res.append(dict(fold="pooled", train_bias=np.nan, test_bias=O.err.mean(), test_bias_corrected=O.e2.mean(), mae=O.err.abs().mean(),
                mae_corrected=O.e2.abs().mean(), within5=(O.err.abs() <= 5).mean(), within5_corrected=(O.e2.abs() <= 5).mean(),
                bias_corrected_lo=np.percentile(boot[:, 0], 2.5), bias_corrected_hi=np.percentile(boot[:, 0], 97.5),
                mae_corrected_lo=np.percentile(boot[:, 1], 2.5), mae_corrected_hi=np.percentile(boot[:, 1], 97.5)))
pd.DataFrame(res).to_csv(os.path.join(OUT, "bias_correction.csv"), index=False)
print(pd.DataFrame(res).round(3).to_string(index=False))

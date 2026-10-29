"""Metrics and participant-bootstrap helpers shared by the redo scripts (definitions copied from the 2026-10-28 analysis code)."""
import warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

STEP = 0.5
DRAWS = 5000


def kappa(y, p):
    if len(np.unique(np.r_[y, p])) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cohen_kappa_score(y, p)


def bounds(s):
    edges = np.diff(np.r_[False, s, False].astype(int)); st = np.flatnonzero(edges == 1); en = np.flatnonzero(edges == -1)
    keep = en - st >= 20
    return (int(st[keep][0]), int(en[keep][-1])) if keep.any() else None


def measure(y, p, mask):
    """Per-night metrics on epochs of mask with a label and a call 0-3 (call 4 = gap, -1 = none are excluded)."""
    ok = mask & (y >= 0) & (p >= 0) & (p < 4); t = y[ok]; c = p[ok]; ts = t > 0; ps = c > 0
    fp = np.sum(ps & ~ts) * STEP; fn = np.sum(~ps & ts) * STEP
    a = bounds((y > 0) & ok); b = bounds((p > 0) & ok)
    r = dict(n_epochs=int(ok.sum()), ref_sleep_min=ts.sum() * STEP, false_sleep_min=fp, false_wake_min=fn, tst_error=fp - fn,
             tst_bias=fp - fn, kappa4=kappa(t, c), kappa_sw=kappa(ts, ps),
             sensitivity=float(ps[ts].mean()) if ts.any() else np.nan, specificity=float((~ps[~ts]).mean()) if (~ts).any() else np.nan)
    for name, sl in [('before', slice(0, a[0] if a else 0)), ('within', slice(a[0], a[1]) if a else slice(0, 0)), ('after', slice(a[1] if a else len(y), len(y)))]:
        m = ok[sl]; yy = y[sl]; pp = p[sl]
        r[f'{name}_false_sleep'] = np.sum(m & (yy == 0) & (pp > 0)) * STEP
        r[f'{name}_false_wake'] = np.sum(m & (yy > 0) & (pp == 0)) * STEP
    if a and b:
        r['waso_error'] = (np.sum(ok[b[0]:b[1]] & (p[b[0]:b[1]] == 0)) - np.sum(ok[a[0]:a[1]] & (y[a[0]:a[1]] == 0))) * STEP
    else:
        r['waso_error'] = np.nan
    return r


def cluster_ci(person, v, seed=29028, draws=DRAWS):
    """95% participant-bootstrap CI of the mean over nights (participants resampled, their nights pooled)."""
    d = pd.DataFrame({'p': np.asarray(person), 'v': np.asarray(v, float)}).dropna()
    g = d.groupby('p')['v'].agg(['sum', 'count']).to_numpy()
    if len(g) < 2:
        return [np.nan, np.nan]
    ix = np.random.default_rng(seed).integers(0, len(g), (draws, len(g))); s = g[ix].sum(1)
    return np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5]).tolist()


def paired(df, a, b, metric, absolute=False):
    """df: per-night frame with columns model, subject, person and the metric; mean of (a - b) over shared nights."""
    x = df[df.model == a].set_index('subject'); y0 = df[df.model == b].set_index('subject'); ix = x.index.intersection(y0.index)
    xa, ya = x.loc[ix, metric], y0.loc[ix, metric]
    d = (xa.abs() - ya.abs()) if absolute else (xa - ya)
    d = d.dropna()
    if len(d) < 2:
        return None
    ci = cluster_ci(x.loc[d.index, 'person'].to_numpy(), d.to_numpy())
    return dict(nights=len(d), participants=x.loc[d.index, 'person'].nunique(), mean_a=xa[d.index].mean(), mean_b=ya[d.index].mean(), delta=d.mean(), ci_lo=ci[0], ci_hi=ci[1],
                wilcoxon_p=wilcoxon(d).pvalue if (d != 0).any() and len(d) > 5 else np.nan)

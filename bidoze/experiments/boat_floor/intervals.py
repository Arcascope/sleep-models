"""Vessel-cluster intervals for existing paired boat comparisons; reads saved CSVs only.

Each comparison is reduced to one difference per participant (nights, then seeds, averaged
first). The participant-macro mean is bootstrapped by resampling vessels with replacement;
Somerset has no vessel identity below the cohort, so it gets a participant bootstrap instead.
The detectable difference is 2.8 bootstrap SDs (two-sided 5% test, 80% power, normal approx.).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DRAWS = 10000
SPEC_METRICS = ['descriptive_spec95', 'auroc', 'wake_specificity', 'still_wake_specificity', 'sleep_sensitivity', 'kappa_sleep_wake', 'kappa_4stage']


def cluster_interval(delta, clusters, rng):
    """delta, clusters: per-participant arrays. Returns mean, 95% interval, bootstrap SD."""
    ids = np.unique(clusters)
    members = [delta[clusters == c] for c in ids]
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(len(ids), size=len(ids))
        draws[i] = np.concatenate([members[j] for j in pick]).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi), float(draws.std())


def rows_for(label, wide, boats, metric, scale, rng):
    """wide: participant-indexed frame with columns 'new' and 'old'."""
    wide = wide.dropna()
    delta = (wide.new - wide.old).to_numpy() * scale
    surf = np.array([p in boats for p in wide.index])
    clusters = np.array([boats.get(p, p) for p in wide.index])
    kind = 'vessel' if surf.all() else 'participant'
    mean, lo, hi, sd = cluster_interval(delta, clusters, rng)
    p = wilcoxon(delta).pvalue if len(delta) > 1 and np.any(delta != 0) else np.nan
    return dict(comparison=label, metric=metric, participants=len(delta), clusters=len(np.unique(clusters)),
        bootstrap=kind, difference=mean, ci_low=lo, ci_high=hi, detectable=2.8 * sd, wilcoxon_p=p)


def main(a):
    rng = np.random.default_rng(23)
    boats = pd.read_csv(a.results / 'boat-baseline-2026-10-09/boats.csv').set_index('participant').boat.to_dict()
    out = []
    # Release-scale floor training: night × seed rows in percent; average seeds then nights.
    n = pd.read_csv(a.results / 'bidoze-floor-release-2026-10-10/scores/nights.csv')
    n['arm'] = n.model.str.split('-').str[0]
    for cohort in ('SURF', 'Somerset'):
        g = n[n.cohort == cohort].groupby(['participant', 'night', 'arm'])[['sleep_sensitivity', 'wake_specificity', 'kappa_sleep_wake', 'kappa_4stage']].mean()
        g = g.groupby(['participant', 'arm']).mean()
        for metric, scale in [('wake_specificity', 1), ('sleep_sensitivity', 1), ('kappa_sleep_wake', 1), ('kappa_4stage', 1)]:
            wide = g[metric].unstack('arm').rename(columns={'floor': 'new', 'release': 'old'})
            out.append(rows_for(f'floor-release − release ({cohort}, 3 seeds each)', wide, boats, metric, scale, rng))
    # Matched seed-0 scratch: trace − duplicate per outer group.
    t = pd.read_csv(a.results / 'boat-axes-learning-2026-10-12/analysis/training/subjects.csv')
    t = t[(t.role == 'test') & (t['mode'] == 'ggir') & t.covered & (t.model == 'adapted')]
    for fold in sorted(t.fold.unique()):
        f = t[t.fold == fold].pivot_table(index='participant', columns='arm', values=SPEC_METRICS)
        for metric in SPEC_METRICS:
            if (metric, 'trace') not in f or (metric, 'duplicate') not in f:
                continue
            wide = pd.DataFrame(dict(new=f[metric]['trace'], old=f[metric]['duplicate']))
            out.append(rows_for(f'trace − duplicate (group {fold}, seed 0)', wide, boats, metric, 100 if metric not in ('auroc', 'kappa_sleep_wake', 'kappa_4stage') else 1, rng))
    # Held-out vessel adaptation: adapted − initial, every SURF participant in its outer fold.
    e = pd.read_csv(a.results / 'boat-evidence-2026-10-11/analysis/subjects.csv')
    e = e[(e.role == 'test') & (e['mode'] == 'ggir') & e.covered]
    for arm in ('head', 'full'):
        for cohort in ('SURF', 'Somerset'):
            g = e[(e.arm == arm) & (e.cohort == cohort)].groupby(['participant', 'model'])[SPEC_METRICS].mean()
            if g.empty:
                continue
            for metric in ('descriptive_spec95', 'auroc', 'wake_specificity', 'sleep_sensitivity'):
                wide = g[metric].unstack('model').rename(columns={'adapted': 'new', 'initial': 'old'})
                seeds = sorted(e[(e.arm == arm) & (e.cohort == cohort)].seed.unique())
                out.append(rows_for(f'{arm} adaptation − initial seed ({cohort}, seeds {seeds})', wide, boats, metric, 100 if metric not in ('auroc',) else 1, rng))
    # Window layouts, group D seed 0: contained/three-window mean − native mean.
    w = pd.read_csv(a.results / 'boat-window-pooling-2026-10-12/analysis/subjects.csv')
    w = w[(w.role == 'test') & (w['mode'] == 'ggir') & w.covered & (w.model == 'adapted')]
    f = w.pivot_table(index='participant', columns='arm', values=SPEC_METRICS)
    for arm in ('contained_mean', 'nonoverlap_mean'):
        for metric in ('descriptive_spec95', 'auroc'):
            wide = pd.DataFrame(dict(new=f[metric][arm], old=f[metric]['native_mean']))
            out.append(rows_for(f'{arm} − native_mean (group D, seed 0)', wide, boats, metric, 100 if metric != 'auroc' else 1, rng))
    table = pd.DataFrame(out)
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / 'vessel_intervals.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 200, 'display.max_colwidth', 60):
        print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path); p.add_argument('out', type=Path)
    main(p.parse_args())

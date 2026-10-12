"""Ranking metrics for the release-scale floor models versus the release on boat nights.

Wake probability is averaged over each arm's three seeds; AUROC and descriptive specificity at
95% sleep sensitivity are computed per night on covered PSG epochs, then recording means within
participant and participant means, with vessel-cluster intervals for SURF.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from boat_floor.intervals import rows_for


def main(a):
    meta = pd.read_csv(a.data / 'inventory.csv')
    meta = meta[meta.cohort.isin(['SURF', 'Somerset'])]
    boats = pd.read_csv(a.results / 'boat-baseline-2026-10-09/boats.csv').set_index('participant').boat.to_dict()
    pred = a.results / 'bidoze-floor-release-2026-10-10/scores/predictions'
    rows = []
    for r in meta.to_dict('records'):
        stem = r['name'].split('_subject_', 1)[1]
        with np.load(a.data / 'nights' / f"{r['name']}.npz") as d:
            n = int(d['n']); truth = d['truth'][:n]
        for arm in ('release', 'floor'):
            frames = [pd.read_csv(pred / f'{arm}-seed{s}' / f'{stem}.csv') for s in range(3)]
            k = np.rint(frames[0].start_s / 30).astype(int).to_numpy()
            wake = np.mean([f.wake.to_numpy() for f in frames], axis=0)
            gap = frames[0].is_gap.to_numpy() == 1
            keep = (k >= 0) & (k < n) & ~gap
            y = truth[k[keep]]; ok = y >= 0; s = 1 - wake[keep][ok]; y = y[ok] > 0
            fpr, tpr, _ = roc_curve(y, s)
            rows.append(dict(name=r['name'], cohort=r['cohort'], participant=r['participant'], arm=arm,
                auroc=roc_auc_score(y, s), descriptive_spec95=1 - np.interp(.95, tpr, fpr), epochs=int(ok.sum())))
    nights = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True); nights.to_csv(a.out / 'floor_release_roc_nights.csv', index=False)
    subjects = nights.groupby(['cohort', 'participant', 'arm'])[['auroc', 'descriptive_spec95']].mean().reset_index()
    rng = np.random.default_rng(31); out = []
    for cohort, g in subjects.groupby('cohort'):
        for metric, scale in [('auroc', 1), ('descriptive_spec95', 100)]:
            w = g.pivot(index='participant', columns='arm', values=metric)
            out.append(dict(cohort=cohort, release=w.release.mean() * scale, floor=w.floor.mean() * scale) |
                rows_for('floor − release', pd.DataFrame(dict(new=w.floor, old=w.release)), boats, metric, scale, rng))
    t = pd.DataFrame(out); t.to_csv(a.out / 'floor_release_roc.csv', index=False)
    print(t.drop(columns=['comparison']).round(4).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data', type=Path); p.add_argument('results', type=Path); p.add_argument('out', type=Path)
    main(p.parse_args())

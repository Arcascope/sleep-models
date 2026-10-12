"""Single-feature still-epoch wake/sleep AUROC of the in-band epoch level, per night.

Reads the saved native epoch PSDs (boat-observation run). For each night and channel the
feature is the epoch's mean linear power over 2-6 Hz (and 0.1-0.5 Hz); AUROC is PSG wake
versus sleep on still, covered epochs. Subtracting a constant floor is monotone in this
feature, so it leaves these per-night values unchanged.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CHANNELS = ['x', 'y', 'z', 'mag', 'jerk']


def main(a):
    rows = []
    paths = sorted((a.observation / 'nights').glob('*/native_axis_psd.npz')) + sorted((a.observation / 'land').glob('*/native_axis_psd.npz'))
    for path in paths:
        name = path.parent.name
        cohort, rest = name.split('_subject_', 1)
        d = np.load(path)
        p = d['epoch_linear_psd']; f = d['frequencies']; truth = d['truth']; still = d['still']; covered = ~d['gap']
        p = np.concatenate([p, p[:, :, :3].sum(axis=2, keepdims=True)], axis=2)
        take = covered & (truth >= 0) & np.isfinite(p).all(axis=(1, 2))
        for subset, mask in [('still', take & still), ('all', take)]:
            y = truth[mask] == 0
            if y.sum() < 5 or (~y).sum() < 5:
                continue
            for lo, hi in [(2, 6), (0.1, 0.5)]:
                band = (f >= lo) & (f <= hi)
                for c, channel in enumerate(CHANNELS + ['trace']):
                    if channel in ('x', 'y', 'z'):
                        continue
                    rows.append(dict(name=name, cohort=cohort, participant=rest.split('_')[0], subset=subset, band=f'{lo:g}-{hi:g}',
                        channel=channel, wake=int(y.sum()), sleep=int((~y).sum()), auroc=roc_auc_score(y, p[mask][:, band, c].mean(axis=1))))
    nights = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    nights.to_csv(a.out / 'ceiling_nights.csv', index=False)
    subjects = nights.groupby(['cohort', 'participant', 'subset', 'band', 'channel']).auroc.mean().reset_index()
    summary = subjects.groupby(['cohort', 'subset', 'band', 'channel']).auroc.agg(['mean', 'count']).reset_index()
    summary.to_csv(a.out / 'ceiling_summary.csv', index=False)
    print(summary.pivot_table(index=['subset', 'band', 'channel'], columns='cohort', values='mean').round(3).to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('observation', type=Path); p.add_argument('out', type=Path)
    main(p.parse_args())

"""Inner-selected floor subtraction versus native, with vessel-cluster intervals.

For each outer boat group the subtraction strength is the one with the highest inner-group
participant-macro descriptive specificity at 95% sensitivity (ties keep native); the outer
group is then scored with that strength. Every strength is also reported descriptively.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from boat_floor.intervals import rows_for

METRICS = ['sleep_sensitivity', 'wake_specificity', 'still_wake_specificity', 'kappa_sleep_wake', 'kappa_4stage', 'auroc', 'descriptive_spec95', 'clamped_fraction']
SCALE = dict(sleep_sensitivity=100, wake_specificity=100, still_wake_specificity=100, descriptive_spec95=100, auroc=1, kappa_sleep_wake=1, kappa_4stage=1, clamped_fraction=100)


def main(a):
    subjects = pd.read_csv(a.run / 'analysis/subjects.csv')
    subjects = subjects[subjects['mode'] == 'ggir']
    splits = json.loads((a.data / 'splits.json').read_text())
    boats = pd.read_csv(a.results / 'boat-baseline-2026-10-09/boats.csv').set_index('participant').boat.to_dict()
    boat = subjects[subjects.cohort.isin(['SURF', 'Somerset'])]
    by_group = boat.groupby(['group', 'variant']).descriptive_spec95.mean().unstack('variant')
    order = ['native'] + sorted(v for v in by_group.columns if v != 'native')
    selected = {}
    for outer, plan in splits.items():
        inner = by_group.loc[plan['inner_group'], order]
        selected[outer] = dict(inner_group=plan['inner_group'], variant=str(inner.idxmax()),
            inner_spec95=inner.to_dict())
    (a.run / 'analysis/selection.json').write_text(json.dumps(selected, indent=2))
    pick = boat[boat.apply(lambda r: r.variant == selected[r.group]['variant'], axis=1)].assign(variant='selected')
    table = pd.concat([subjects, pick])
    summary = table.groupby(['cohort', 'variant'])[METRICS].mean()
    summary['participants'] = table.groupby(['cohort', 'variant']).size()
    for m, s in SCALE.items():
        summary[m] = summary[m] * s
    summary.round(4).to_csv(a.run / 'analysis/report_summary.csv')
    rng = np.random.default_rng(29); rows = []
    for cohort in ('SURF', 'Somerset', 'SleepAccel-Clinical'):
        c = table[table.cohort == cohort]
        for variant in [v for v in c.variant.unique() if v != 'native']:
            for metric in ['descriptive_spec95', 'auroc', 'wake_specificity', 'still_wake_specificity', 'sleep_sensitivity', 'kappa_sleep_wake', 'kappa_4stage']:
                wide = c.pivot_table(index='participant', columns='variant', values=metric)
                if variant not in wide or 'native' not in wide:
                    continue
                pair = pd.DataFrame(dict(new=wide[variant], old=wide['native']))
                rows.append(dict(cohort=cohort, variant=variant) | rows_for(f'{variant} − native', pair, boats, metric, SCALE[metric], rng))
    paired = pd.DataFrame(rows)
    paired.to_csv(a.run / 'analysis/paired.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 300):
        print(json.dumps({k: v['variant'] for k, v in selected.items()}))
        print(summary.round(3).to_string())
        print(paired.drop(columns=['comparison']).round(4).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path); p.add_argument('data', type=Path); p.add_argument('results', type=Path)
    main(p.parse_args())

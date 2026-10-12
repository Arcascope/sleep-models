"""Somerset scored with the label shift fixed by the Somfit start-time seconds, not by fit.

Each cleaned Somerset PSG grid starts s seconds before Epoch 1 (s is the dropped seconds of the
Somfit Start Recording time). Label i then covers accelerometer time [30i + s, 30i + s + 30), so
the label that overlaps prediction epoch j most is epoch j - round(s/30): shift k = round(s/30).
Reads the per-night shift table and the source timing table; writes timing/corrected_*.csv.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from boat_floor.intervals import rows_for

METRICS = ['sleep_sensitivity', 'wake_specificity', 'still_wake_specificity', 'kappa_sleep_wake', 'kappa_4stage', 'auroc', 'descriptive_spec95']
SCALE = dict(sleep_sensitivity=100, wake_specificity=100, still_wake_specificity=100, descriptive_spec95=100, auroc=1, kappa_sleep_wake=1, kappa_4stage=1)


def main(a):
    nights = pd.read_csv(a.timing / 'shift_nights.csv')
    nights = nights[(nights.cohort == 'Somerset') & (nights['mode'] == a.mode)]
    source = pd.read_csv(a.timing / 'somerset_source_timing.csv')
    k = {f'Somerset_subject_{n.split("_", 1)[1]}': int(np.floor(s / 30 + 0.5)) for n, s in zip(source.night, source.psg_start_minus_anchor_s)}
    missing = set(nights.name) - set(k)
    assert not missing, missing
    nights = nights.assign(source_shift=nights.name.map(k))
    pick = nights[nights['shift'] == nights.source_shift].assign(variant='source')
    base = nights[nights['shift'] == 0].assign(variant='uncorrected')
    assert len(pick) == len(base) == len(k)
    both = pd.concat([base, pick])
    subjects = both.groupby(['variant', 'participant'])[METRICS].mean()
    summary = subjects.groupby('variant').mean()
    for m, s in SCALE.items():
        summary[m] *= s
    summary['participants'] = subjects.groupby('variant').size()
    rng = np.random.default_rng(31); rows = []
    for metric in METRICS:
        wide = subjects[metric].unstack('variant').rename(columns={'source': 'new', 'uncorrected': 'old'})
        rows.append(rows_for('source shift − uncorrected (Somerset)', wide, {}, metric, SCALE[metric], rng))
    paired = pd.DataFrame(rows)
    summary.round(4).to_csv(a.timing / f'corrected_summary_{a.mode}.csv')
    paired.to_csv(a.timing / f'corrected_paired_{a.mode}.csv', index=False)
    pick[['name', 'participant', 'source_shift']].to_csv(a.timing / 'corrected_shifts.csv', index=False)
    with pd.option_context('display.width', 250):
        print(pick.source_shift.value_counts().to_dict())
        print(summary.round(4).to_string())
        print(paired.drop(columns=['comparison']).round(4).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('timing', type=Path); p.add_argument('--mode', default='ggir')
    main(p.parse_args())

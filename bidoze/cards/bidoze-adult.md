---
license: mit
tags:
- sleep-staging
- actigraphy
- accelerometer
- time-series
---

# arcascope/arcasleep-bidoze-adult · v0.1.0

**Sleep stages from triaxial accelerometer recordings, for adults.**
4-class sleep stage prediction, metric reports, and hypnograms.

| **Predicts** | `wake`, `light`, `deep`, `rem` per 30-s epoch |
|:---|:---|
| **For** | Adults |
| **Input** | Timestamped triaxial accelerometer, 12 Hz or faster |
| **Score** | Cohen's κ **0.56** sleep/wake · **0.44** 4-stage (SleepAccel cross-validation, 28 adults) · **0.56** · **0.30** (SleepAccel-Clinical, 32 adults, not in training) |
| **License** | MIT |

## Quickstart

```bash
pip install 'arcasleep[bidoze,inference,hub]>=0.2.0'
hf download arcascope/arcasleep-bidoze-adult --local-dir arcasleep-bidoze-adult

# One CSV per recording: epoch, start_s, wake, light, deep, rem, stage, stage_raw, is_nap, is_sib, is_spt, is_gap
# plus <name>_summary.csv: subject_id, tst_minutes, missing_data_minutes, nap_minutes
arcasleep bidoze predict arcasleep-bidoze-adult recordings/*.csv -o predictions/ --hypnogram
```

Or from Python:

```python
from arcasleep.predict import load_predictor, read_accelerometer

predictor = load_predictor("arcasleep-bidoze-adult")   # load once, reuse
result = predictor.predict(read_accelerometer("night.csv"))  # (N, 4): t [s], x, y, z [g]

result.labels          # stage per 30-s epoch, "gap" where no data
result.probabilities   # (epochs, 4): wake, light, deep, rem
result.summary         # subject_id, tst_minutes, missing_data_minutes, nap_minutes
result.to_csv("night_stages.csv")
```

The GGIR-style sleep filter, `origin=`, `--no-ggir-sleep-filter` and the output columns work as
for [arcasleep-bidoze-pediatric](https://huggingface.co/arcascope/arcasleep-bidoze-pediatric).
For recordings of ≤ 9 hours, such as a single night, the filter changes nothing.

## How it scores a recording

The bundle holds three networks of one architecture, trained with different random seeds. They
run as one batched call and their outputs are averaged. Each 30-s epoch is then scored in two
steps:

1. **Sleep or wake:** the epoch is wake when wake is the most probable of wake, light, deep
   and REM in the averaged output; otherwise it is sleep.
2. **Light, deep or REM**, on sleep epochs only: the light, deep and REM probabilities are each
   divided by a fixed *stage divisor* before the most probable is chosen.

The divisor (in `composite.json`) corrects for how the model was trained. Training weights rare
stages up, and the training nights include children, who spend far more of the night in deep
sleep than adults. Without the divisor the model calls 27.5% of SleepAccel epochs deep sleep;
with it, 21.5% (PSG: 14.7%). The divisor is set from the stage proportions of 249 adult PSG
nights (BIDsleep) and does not use the evaluation cohorts.

The `wake` column of the CSV is the averaged probability of wake. The `light`, `deep` and `rem`
columns share the rest in proportion to their divided probabilities. The `stage` column comes
from the two steps above, so it is not always the largest of the four columns.

## Performance

Metrics compare the model's stages with PSG scoring, averaged over subjects, with 95%
confidence intervals of the mean. p: paired Wilcoxon signed-rank test over subjects, against
the pediatric release ([arcasleep-bidoze-pediatric](https://huggingface.co/arcascope/arcasleep-bidoze-pediatric)
v0.1.0) on the same subjects.

- **SleepAccel** (28 healthy adults) was part of the training data, so its column is
  cross-validated: SleepAccel was split into five folds by subject, and each subject is scored
  by models trained without its fold.
- **SleepAccel-Clinical** (32 adults with no diagnosis or mild, moderate or severe obstructive
  sleep apnea) was never used in training. It is scored by the released model.

| Metric | SleepAccel (cross-validation) | p | SleepAccel-Clinical | p |
|:---|---:|---:|---:|---:|
| Cohen's κ, sleep/wake | 0.56 (0.50 to 0.63) | 0.13 | 0.56 (0.48 to 0.63) | 0.90 |
| Cohen's κ, 4 stages | 0.44 (0.39 to 0.49) | 0.060 | 0.30 (0.24 to 0.36) | 0.011 |
| Sleep sensitivity (%) | 95.7 (94.7 to 96.7) | 0.055 | 94.1 (91.8 to 96.4) | 0.22 |
| Wake specificity (%) | 69.9 (62.2 to 77.7) | 0.68 | 63.1 (55.0 to 71.3) | 0.19 |
| Wake specificity at 95% sleep sensitivity (%) | 68.2 | | 53.9 | |
| TST error (min) | −4.2 (−11.4 to +3.0) | 0.14 | +19.0 (+3.6 to +34.4) | 0.065 |
| WASO error (min) | +2.4 (−3.6 to +8.4) | 0.50 | −12.3 (−23.9 to −0.8) | 0.037 |
| Deep sleep, % of epochs (PSG) | 21.5 (14.7) | < 0.001 | 21.8 (8.7) | < 0.001 |
| Light-sleep error (%) | 16.5 (12.3 to 20.7) | 0.013 | 22.8 (17.5 to 28.2) | < 0.001 |
| Deep-sleep error (%) | 107 (45 to 168) | < 0.001 | 244 (83 to 405) | < 0.001 |
| REM-sleep error (%) | 41.7 (27.3 to 56.0) | 0.20 | 115 (62 to 168) | 0.79 |

The same metrics for the pediatric release on these subjects:

| Metric | SleepAccel | SleepAccel-Clinical |
|:---|---:|---:|
| Cohen's κ, sleep/wake | 0.58 | 0.55 |
| Cohen's κ, 4 stages | 0.40 | 0.26 |
| Sleep sensitivity (%) | 96.3 | 95.2 |
| Wake specificity (%) | 69.3 | 59.2 |
| Wake specificity at 95% sleep sensitivity (%) | 69.1 | 54.0 |
| TST error (min) | −1.8 | +26.6 |
| Deep sleep, % of epochs | 29.6 | 33.2 |

- **Stages are more accurate than the pediatric release's**: 4-stage κ 0.30 vs 0.26 on
  SleepAccel-Clinical (p = 0.011) and 0.44 vs 0.40 on SleepAccel (p = 0.060).
- **Deep sleep is overestimated less**: 21.5% vs 29.6% of epochs on SleepAccel (PSG 14.7%), and
  21.8% vs 33.2% on SleepAccel-Clinical (PSG 8.7%).
- **Sleep/wake does not differ significantly from the pediatric release** on either cohort. At
  the same 95% sleep sensitivity, wake specificity is 68.2% vs 69.1% (SleepAccel) and 53.9% vs
  54.0% (SleepAccel-Clinical).
- "Wake specificity at 95% sleep sensitivity" sets the threshold on P(sleep) so that 95% of PSG
  sleep epochs, pooled over a cohort's subjects, are called sleep. It compares models at the
  same operating point; it has no paired test.

**What the κ gain means in a night.** Against the pediatric release on the same nights:

| | SleepAccel (median night 7.8 h) | SleepAccel-Clinical (median night 5.7 h) |
|:---|---:|---:|
| Epochs whose stage matches PSG | 64.6% vs 59.5% | 56.4% vs 48.9% |
| More time staged correctly, per night | 22 min | 27 min |
| Light sleep the pediatric release called deep, now scored light (net, per night) | 22 min | 23 min |
| Time the two models label differently, per night | 91 min | 93 min |
| … of which sleep vs wake | 13 min | 21 min |

Most of the gain is light sleep that the pediatric release scored as deep. The two models
disagree on more time than the net gain, because many epochs are borderline between stages.

**Whole recordings.** The table scores each night's PSG period. Scored from the whole raw
recording, which can run hours before and after it, the SleepAccel-Clinical results do not
differ significantly: 4-stage κ 0.30 (p = 0.71) and sleep/wake κ 0.54 (p = 0.43), on the same
PSG epochs.

## Intended use

Research estimates of sleep and wake, and of light, deep and REM sleep, from a wrist-worn
accelerometer in adults. You need no PSG to run it; PSG may be used to check its accuracy on
your own cohort.

Sleep/wake is the primary task. Stage estimates are less accurate and are best read as
cohort-level trends, not night-by-night measurements.

**Not a medical device.** Do not use it to diagnose sleep disorders (including obstructive
sleep apnea), to screen individuals, or to make clinical decisions.

## Limitations

- **Wake is harder to detect than sleep.** The model finds 95.7% of PSG sleep epochs but 69.9%
  of wake epochs in healthy adults, and 94.1% and 63.1% in adults with sleep apnea. Total sleep
  time is overestimated by about 19 minutes in adults with sleep apnea.
- **Deep sleep is still overestimated**, by about 7 percentage points of the night in healthy
  adults and 13 in adults with sleep apnea. The divisor is set from adults with 18.9% deep
  sleep; cohorts with less deep sleep, such as older adults or people with sleep apnea, are
  overestimated more.
- **Few adults have been evaluated:** 60, in two cohorts. Only SleepAccel-Clinical (32) was
  never used in training.
- **Device and placement.** Trained on wrist data from Apple Watch, Fitbit and ActiGraph
  devices, sampled at 12 Hz or faster. Hip, ankle and upper-arm placements have not been
  evaluated.
- **Recording length.** Up to 25 hours are scored in one pass; longer recordings are scored in
  consecutive passes.

## Training data

- **Weaver _et al_** ([PubMed 40676371](https://pubmed.ncbi.nlm.nih.gov/40676371/)): 202
  pediatric nights from 201 children (111 Fitbit, 70 Apple Watch and 21 ActiGraph recordings),
  a clinical sample spanning no, mild, moderate and severe obstructive sleep apnea.
- **[SleepAccel](https://pubmed.ncbi.nlm.nih.gov/31579900/)**: 28 healthy adults, Apple Watch.

Evaluation only: [SleepAccel-Clinical](https://doi.org/10.1093/sleepadvances/zpag051), 32 adults
with no diagnosis or mild, moderate or severe obstructive sleep apnea.

**Protocol.** Three seeds were trained with a class-balanced loss, early-stopping on 30% of the
training subjects held out by participant. The released networks trained on all of Weaver and
SleepAccel; the SleepAccel column comes from five sets of three, each trained without one
SleepAccel fold.

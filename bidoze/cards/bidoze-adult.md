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
| **License** | MIT |

| Test set | Cohen's κ, sleep/wake | Cohen's κ, 4 stages |
|:---|---:|---:|
| SleepAccel (28 adults, cross-validated) | **0.56** | **0.44** |
| SleepAccel-Clinical (32 adults, not in training) | **0.56** | **0.30** |

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

The GGIR-style sleep filter, `origin=`, `--no-ggir-sleep-filter` and the output columns are
the same as for [arcasleep-bidoze-pediatric](https://huggingface.co/arcascope/arcasleep-bidoze-pediatric).

- ≤ 9 hours, such as a single night: the filter changes nothing.

## How it scores a recording

The bundle holds three networks of one architecture, trained with different random seeds. They
run as one batched call; their outputs are averaged. Each 30-s epoch is then labeled in two
steps:

1. **Sleep or wake:** the epoch is wake when wake is the most probable of wake, light, deep and
   REM in the averaged output; otherwise it is sleep.
2. **Light, deep or REM**, on sleep epochs only: the light, deep and REM probabilities are each
   divided by a fixed *stage divisor*; the largest quotient gives the stage.

The divisor, stored in `composite.json`, corrects two properties of the training: the loss
weights rare stages up, and the training nights include children, who spend more of the night
in deep sleep than adults. On SleepAccel the model labels 27.5% of epochs deep sleep without the
divisor and 21.5% with it (PSG: 14.7%).

The divisor is computed from the stage proportions of 249 nights of BIDSleep [1], 47 healthy
adults recorded at home with expert-reviewed EEG-headband staging. BIDSleep is not used for
training or evaluation.

Output columns:

- `wake`: the averaged probability of wake.
- `light`, `deep`, `rem`: 1 − `wake`, split in proportion to the divided probabilities.
- `stage`: the label from steps 1 and 2. It is not always the largest of the four columns.

## Performance

Each metric compares the model's labels with PSG scoring per subject; the table gives the mean
over subjects and its 95% confidence interval. p is a paired Wilcoxon signed-rank test over
subjects against the pediatric release,
[arcasleep-bidoze-pediatric](https://huggingface.co/arcascope/arcasleep-bidoze-pediatric) v0.1.0.

- **SleepAccel** [3] (28 healthy adults) is in the training data, so it is cross-validated:
  the subjects are split into five folds; each subject is scored by models trained without its
  fold.
- **SleepAccel-Clinical** [4] (32 adults: no diagnosis, or mild, moderate or severe obstructive
  sleep apnea) is not in the training data. It is scored by the released model.

| Metric | SleepAccel (cross-validation) | p | SleepAccel-Clinical | p |
|:---|---:|---:|---:|---:|
| Cohen's κ, sleep/wake | 0.56 (0.50 to 0.63) | 0.13 | 0.56 (0.48 to 0.63) | 0.90 |
| Cohen's κ, 4 stages | 0.44 (0.39 to 0.49) | 0.060 | 0.30 (0.24 to 0.36) | 0.011 |
| 4-stage accuracy (%) | 64.6 (61.7 to 67.6) | 0.001 | 56.4 (52.5 to 60.3) | < 0.001 |
| Balanced 4-stage accuracy (%) | 67.0 (63.3 to 70.8) | 0.37 | 52.3 (45.9 to 58.7) | 0.057 |
| Sleep sensitivity (%) | 95.7 (94.7 to 96.7) | 0.055 | 94.1 (91.8 to 96.4) | 0.22 |
| Wake specificity (%) | 69.9 (62.2 to 77.7) | 0.68 | 63.1 (55.0 to 71.3) | 0.19 |
| Wake specificity at 95% sleep sensitivity (%) | 68.2 | | 53.9 | |
| TST error (min) | −4.2 (−11.4 to +3.0) | 0.14 | +19.0 (+3.6 to +34.4) | 0.065 |
| WASO error (min) | +2.4 (−3.6 to +8.4) | 0.50 | −12.3 (−23.9 to −0.8) | 0.037 |
| Deep sleep, % of epochs (PSG) | 21.5 (14.7) | < 0.001 | 21.8 (8.7) | < 0.001 |
| Light-sleep time, absolute error (%) | 16.5 (12.3 to 20.7) | 0.013 | 22.8 (17.5 to 28.2) | < 0.001 |
| Deep-sleep time, absolute error (%) | 107 (45 to 168) | < 0.001 | 244 (83 to 405) | < 0.001 |
| REM-sleep time, absolute error (%) | 41.7 (27.3 to 56.0) | 0.20 | 115 (62 to 168) | 0.79 |

The pediatric release on the same subjects:

| Metric | SleepAccel | SleepAccel-Clinical |
|:---|---:|---:|
| Cohen's κ, sleep/wake | 0.58 | 0.55 |
| Cohen's κ, 4 stages | 0.40 | 0.26 |
| 4-stage accuracy (%) | 59.5 | 48.9 |
| Balanced 4-stage accuracy (%) | 66.0 | 50.0 |
| Sleep sensitivity (%) | 96.3 | 95.2 |
| Wake specificity (%) | 69.3 | 59.2 |
| Wake specificity at 95% sleep sensitivity (%) | 69.1 | 54.0 |
| TST error (min) | −1.8 | +26.6 |
| Deep sleep, % of epochs | 29.6 | 33.2 |

Metric definitions:

- **4-stage accuracy:** the percentage of a night's PSG-scored epochs labeled with PSG's stage.
- **Balanced 4-stage accuracy:** the mean over wake, light, deep and REM of the percentage of
  that stage's PSG epochs labeled correctly, so each stage counts equally.
- **Stage-time absolute error:** |model − PSG| minutes in the stage, as a percentage of PSG's
  minutes.
- **Wake specificity at 95% sleep sensitivity:** the threshold on P(sleep) is set so that 95%
  of PSG sleep epochs, pooled over the cohort's subjects, are labeled sleep. This compares the
  two models at the same sleep sensitivity; it is one pooled value, so it has no paired test.

Compared with the pediatric release:

- **4-stage κ is higher:** 0.30 vs 0.26 on SleepAccel-Clinical (p = 0.011); 0.44 vs 0.40 on
  SleepAccel (p = 0.060, not significant).
- **Less deep sleep is labeled:** 21.5% vs 29.6% of epochs on SleepAccel (PSG 14.7%); 21.8% vs
  33.2% on SleepAccel-Clinical (PSG 8.7%).
- **Sleep/wake κ does not differ significantly** (p ≥ 0.13). At 95% sleep sensitivity, wake
  specificity is 68.2% vs 69.1% on SleepAccel and 53.9% vs 54.0% on SleepAccel-Clinical.
- **Balanced accuracy differs less than accuracy** (67.0% vs 66.0%, p = 0.37; 52.3% vs 50.0%,
  p = 0.057), because light-sleep recall rises while deep-sleep recall falls:
  - Light: 67.1% vs 55.0% on SleepAccel; 57.9% vs 44.8% on SleepAccel-Clinical.
  - Deep: 81.2% vs 89.1% on SleepAccel; 58.2% vs 69.2% on SleepAccel-Clinical.

  Light sleep is about half of the night and deep sleep 9–15%, so accuracy, which weights
  stages by their time, rises more than balanced accuracy, which weights them equally.

**Per night.** Against the pediatric release on the same nights:

| | SleepAccel (median night 7.8 h) | SleepAccel-Clinical (median night 5.7 h) |
|:---|---:|---:|
| 4-stage accuracy | 64.6% vs 59.5% | 56.4% vs 48.9% |
| Additional minutes labeled correctly | 22 | 27 |
| Light sleep labeled deep by the pediatric release and light by this model (net minutes) | 22 | 23 |
| Minutes the two models label differently | 91 | 93 |
| … of which one model labels sleep and the other wake | 13 | 21 |

Most of the additional correct minutes are light sleep that the pediatric release labels deep.
The two models label 91–93 minutes differently, more than the net gain of 22–27 minutes,
because some of those epochs move to PSG's stage and others move away from it.

**Whole recordings.** The tables score the PSG period of each night. Scoring the whole
recording, which can extend hours before and after the PSG period, and then comparing on the
same PSG epochs, SleepAccel-Clinical gives 4-stage κ 0.30 (p = 0.71) and sleep/wake κ 0.54
(p = 0.43); neither differs significantly from the PSG-period scores.

## Intended use

Research estimates of sleep and wake, and of light, deep and REM sleep, from a wrist-worn
accelerometer in adults. PSG is not required to run the model; it can be used to check the
model's accuracy on your own cohort.

Sleep/wake is the primary output. Stage estimates are less accurate; report them as cohort
means, not as per-night measurements.

**Not a medical device.** Do not use it to diagnose sleep disorders (including obstructive
sleep apnea), to screen individuals, or to make clinical decisions.

## Limitations

- **Wake is detected less often than sleep.** The model labels 95.7% of PSG sleep epochs sleep
  but 69.9% of PSG wake epochs wake on SleepAccel; 94.1% and 63.1% on SleepAccel-Clinical.
  Total sleep time is overestimated by 19.0 minutes on SleepAccel-Clinical.
- **Deep sleep is overestimated** by 6.8 percentage points of the night on SleepAccel and 13.1
  on SleepAccel-Clinical. The divisor assumes BIDSleep's 18.9% deep sleep; cohorts with less
  deep sleep, such as older adults or people with sleep apnea, are overestimated more.
- **60 adults have been evaluated,** in two cohorts. Only SleepAccel-Clinical (32 adults) is
  outside the training data.
- **Devices and placement.** Trained on wrist recordings from Apple Watch, Fitbit and ActiGraph
  devices at ≥ 12 Hz. Hip, ankle and upper-arm placements have not been evaluated.
- **Recording length.** Up to 25 hours are scored in one pass; longer recordings are scored in
  consecutive passes.

## Training data

- **Weaver _et al._** [2]: 202 nights from 201 children (111 Fitbit, 70 Apple Watch and 21
  ActiGraph recordings); a clinical sample with no, mild, moderate or severe obstructive sleep
  apnea.
- **SleepAccel** [3]: 28 healthy adults, Apple Watch.

Not trained on:

- **SleepAccel-Clinical** [4]: evaluation only.
- **BIDSleep** [1]: stage proportions only, to compute the stage divisor.

**Protocol.** Three seeds were trained with a class-balanced loss; each stopped early on 30% of
its training subjects, held out by participant. The released networks are trained on all of
Weaver and SleepAccel. The SleepAccel column comes from five sets of three networks, each set
trained without one SleepAccel fold.

## References

1. Song T, Zhang Y, Zhou Z, Dutta J. A Multi-Night Instantaneous Heart Rate and Accelerometry
   Dataset with EEG Sleep Stage Labels (version 1.0.1). PhysioNet; 2026.
   [doi:10.13026/rees-1092](https://doi.org/10.13026/rees-1092). Described in: Song T-A, Zhang
   Y, Zhou Z, Hou L, Malekzadeh M, Behzad A, Dutta J. AI-driven sleep staging using
   instantaneous heart rate and accelerometry: insights from an Apple Watch study. *IEEE Trans
   Biomed Eng*. 2025. [doi:10.1109/TBME.2025.3612158](https://doi.org/10.1109/TBME.2025.3612158).
2. Weaver RG, White JW, Finnegan O, et al. Predicting sleep and sleep stage in children using
   actigraphy and heartrate via a long short-term memory deep learning algorithm: a performance
   evaluation. *J Sleep Res*. 2026;35(1):e70149.
   [doi:10.1111/jsr.70149](https://doi.org/10.1111/jsr.70149).
3. Walch O, Huang Y, Forger D, Goldstein C. Sleep stage prediction with raw acceleration and
   photoplethysmography heart rate data derived from a consumer wearable device. *Sleep*. 2019;
   42(12):zsz180. [doi:10.1093/sleep/zsz180](https://doi.org/10.1093/sleep/zsz180).
4. Canton E, Tavella F, Drake C, Walch O, Cheng P. What matters beyond model choice for wearable
   sleep staging? How personalization, evaluation choices, and easy-to-classify wake impact
   performance. *SLEEP Adv*. 2026;7:zpag051.
   [doi:10.1093/sleepadvances/zpag051](https://doi.org/10.1093/sleepadvances/zpag051).

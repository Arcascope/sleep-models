## Intended use

Research estimates of sleep and wake, and of light, deep and REM sleep, from a
wrist-worn accelerometer in children and adolescents aged 5 to 18. You need no
polysomnography (PSG) to run it; PSG may be used to check its accuracy on
your own cohort.

Sleep/wake is the primary task. Stage estimates are less accurate and are best
read as cohort-level trends, not night-by-night measurements.

**Not a medical device.** Do not use it to diagnose sleep disorders (including
obstructive sleep apnea), to screen individuals, or to make clinical decisions.

## Limitations

- **Wake is harder to detect than sleep.** In the pediatric cross-validation
  it finds 94% of PSG sleep epochs but 82% of wake epochs; the wake it
  misses is usually quiet wakefulness. Total sleep time is underestimated by
  about 11 minutes on average.
- **Stages are harder than sleep/wake.** Light, deep and REM durations carry
  much larger errors than total sleep time; REM is the least reliable.
- **Adults are outside the intended population.** Wake specificity and κ are
  lower in both adult cohorts, and total sleep time is overestimated by about
  27 minutes in adults with sleep apnea. Deep sleep is overestimated: it is
  about twice the PSG amount in healthy adults and close to four times in
  adults with sleep apnea, and light sleep is underestimated by a similar
  amount. The training data is
  pediatric, and children spend far more of the night in deep sleep than
  adults do. Validate on your own data before using it with adults.
- **Device and placement.** It was trained on wrist data from Apple Watch and
  Fitbit devices, sampled at 12 Hz or faster. Hip, ankle and upper-arm
  placements have not been evaluated.
- **Each epoch's input extends 8 s into the next epoch.** The model reads
  10-s spectral windows that start every 2 s, at 0, 2, …, 28 s from the
  epoch start, so one epoch's input covers 38 s. A 1-s movement that starts
  1 s into an epoch appears in 4 windows of the previous epoch and in 1
  window of its own.
- **Vibration near half the sampling rate enters the jerk input.** The jerk
  channel is the norm of the difference between consecutive samples; this
  moves vibration close to half the sampling rate into the 0.1–6 Hz band the
  model reads. In a synthetic 50 Hz recording, a 10 mg vibration at 24 Hz
  raised the jerk input near 2 Hz by 28 dB; the same vibration at 10 Hz
  raised it by 1 dB. The magnitude input was unchanged. This applies to
  recordings made in vehicles or near machinery, and to devices with a high
  noise floor above 6 Hz.

## GGIR sleep filter

**For recordings of ≤ 9 hours, such as a single night, the filter changes
nothing:** the output is the model's own scoring. For longer recordings it
applies the rule in the Quickstart above.

**Why the model's scoring is unchanged inside the sleep period.** On the pediatric
cross-validation, also flipping sleep to wake inside the sleep period wherever
GGIR saw no sustained inactivity moved sensitivity from 94% to 88% and specificity from
82% to 88%, and lowered sleep/wake κ from 0.65 to 0.56. Lowering the model's
own sleep threshold to the same 88% sensitivity reached 91% specificity, so
the model detects wake better than the GGIR rule does.

**Not yet validated:** every PSG night available to us is ≤ 9 hours, so the
step outside the sleep period follows GGIR practice but has not been checked
against PSG.

## BiDoze or UNet

[arcasleep-unet-pediatric](https://huggingface.co/arcascope/arcasleep-unet-pediatric)
is a temporal U-Net trained on the same pediatric data, with the same folds
and the same class-balanced training scheme. At their default (argmax)
settings the two differ slightly: BiDoze finds 94.0% of sleep and 82.1% of
wake, and UNet finds 94.4% of sleep and 79.8% of wake. Calibrated to the same
93.3% sleep sensitivity across the five cross-validation folds, they perform
comparably in a paired, per-subject comparison (202 children):

| At 93.3% sleep sensitivity | BiDoze | UNet |
|:---|---:|---:|
| Wake specificity (%) | 83.9 | 82.2 |
| Cohen's κ, sleep/wake | 0.64 | 0.65 |
| Cohen's κ, 4 stages | 0.55 | 0.56 |
| TST error (min) | −14.0 | −12.3 |

The specificity and κ differences are under 2 points and not significant
(Wilcoxon signed-rank, p > 0.05). Choose on practical grounds:

- **Recording length:** BiDoze scores up to 25 hours in one pass and longer
  recordings in consecutive passes; UNet scores any recording whole.
- **Operating point:** BiDoze's default has slightly higher wake specificity
  (82.1% vs 79.8%) and slightly lower sleep sensitivity (94.0% vs 94.4%).

## Training data

Trained on the pediatric cohort of Weaver _et al_
([PubMed 40676371](https://pubmed.ncbi.nlm.nih.gov/40676371/)), a clinical sample spanning
no, mild, moderate and severe obstructive sleep apnea (OSA).

Two adult cohorts were additionally used for evaluating the trained model:
1. [SleepAccel](https://pubmed.ncbi.nlm.nih.gov/31579900/): healthy adults
2. [SleepAccel-Clinical](https://doi.org/10.1093/sleepadvances/zpag051): adults, with no diagnosis, plus mild, moderate, and severe OSA diagnoses.

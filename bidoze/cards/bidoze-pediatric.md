## Intended use

Research estimates of sleep and wake, and of light, deep and REM sleep, from a
wrist-worn accelerometer in children and adolescents aged 5 to 18. Running BiDoze does not
require polysomnography (PSG). PSG can be used to measure its accuracy on your
own cohort.

Sleep/wake is the primary task. 

**Not a medical device.** Do not use it to diagnose sleep disorders (including
obstructive sleep apnea), to screen individuals, or to make clinical decisions.

## Limitations

- **Wake specificity is lower than sleep sensitivity.** In the pediatric
  cross-validation, sleep sensitivity is 93.2% and wake specificity is 81.1%.
  Total sleep time is underestimated by 12.4 minutes on average.
- **Stage durations have larger errors than total sleep time.** In the
  pediatric cross-validation, the mean absolute percentage error is 7.6% for
  total sleep time, 27.3% for light sleep, 28.5% for deep sleep and 74.5% for
  REM sleep.
- **Adults are outside the intended population.**
  - Wake specificity is 73.4% in healthy adults (SleepAccel) and 63.8% in adults
    with sleep apnea (SleepAccel-Clinical), vs 81.1% in the pediatric
    cross-validation; sleep/wake κ is 0.58 and 0.52, vs 0.61.
  - Deep sleep is overestimated: the predicted amount is 2.2 times the PSG
    amount in healthy adults and 4.0 times in adults with sleep apnea. Light
    sleep is underestimated: 0.64 and 0.54 times the PSG amount.
  - Deep sleep is 33% of PSG sleep in the pediatric training data, vs 16% in
    healthy adults and 11% in adults with sleep apnea.
  - Validate on your own data before using the model with adults.
- **Device and placement.** The model was trained on wrist data from ActiGraph,
  Apple Watch, Fitbit and Garmin devices, sampled at 12 Hz or faster.

## GGIR sleep filter

**For recordings of ≤ 9 hours, such as a single night, the filter does not
change the output:** `stage` equals `stage_raw`. For longer recordings the
filter applies the rule in the Quickstart above.


**The model's scoring is unchanged inside the sleep period.** On the pediatric
cross-validation, also flipping sleep to wake inside the sleep period wherever
GGIR saw no sustained inactivity moved sensitivity from 94% to 88% and specificity from
82% to 88%, and lowered sleep/wake κ from 0.65 to 0.56. Lowering the model's
own sleep threshold to the same 88% sensitivity reached 91% specificity, so
the model detects wake better than the GGIR rule does.

## Training data

Trained on the pediatric cohort of Weaver _et al_
([PubMed 40676371](https://pubmed.ncbi.nlm.nih.gov/40676371/)), a clinical sample of
children with no, mild, moderate and severe obstructive sleep apnea (OSA): 634
recordings from 237 children, each night recorded by up to four wrist devices
(ActiGraph, Apple Watch, Fitbit and Garmin).

Two adult cohorts were used evaluate the trained model:
1. [SleepAccel](https://pubmed.ncbi.nlm.nih.gov/31579900/): healthy adults
2. [SleepAccel-Clinical](https://doi.org/10.1093/sleepadvances/zpag051): adults
   with no, mild, moderate and severe OSA

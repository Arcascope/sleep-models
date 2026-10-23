## Intended use

Research estimates of sleep and wake, and of light, deep and REM sleep, from a
wrist-worn accelerometer in adults. Running BiDoze does not require
polysomnography (PSG). PSG can be used to measure its accuracy on your own
cohort.

Sleep/wake is the primary task. Report stage estimates as cohort means, not as
per-night measurements.

**Not a medical device.** Do not use it to diagnose sleep disorders (including
obstructive sleep apnea), to screen individuals, or to make clinical decisions.

## How it scores a recording

The bundle holds three networks of one architecture, trained with different
random seeds. Their outputs are averaged, then each 30-s epoch is labeled in
two steps:

1. **Sleep or wake:** the epoch is wake when wake is the most probable of wake,
   light, deep and REM in the averaged output; otherwise it is sleep.
2. **Light, deep or REM**, on sleep epochs only: the light, deep and REM
   probabilities are each divided by a fixed stage divisor, stored in
   `composite.json`; the largest quotient gives the stage.

The divisor corrects for two properties of the training: the loss weights rare
stages up, and most training recordings are of children, who spend a larger
share of sleep in deep sleep than adults (33% of PSG sleep in the pediatric
training data). The adult stage proportions in the divisor are those of the
adult training recordings, SleepAccel and Wearanize+ (deep sleep 19.7% of PSG
sleep).

Output columns:

- `wake`: the averaged probability of wake.
- `light`, `deep`, `rem`: 1 − `wake`, split in proportion to the divided
  probabilities.
- `stage`: the label from steps 1 and 2. This is not always the largest of the
  four columns.

## Limitations

- **Wake specificity is lower than sleep sensitivity.**
  - SleepAccel cross-validation: sleep sensitivity 95.8%, wake specificity
    74.8%.
  - Wearanize+ cross-validation: sleep sensitivity 94.5%, wake specificity
    82.8%.
  - SleepAccel-Clinical: sleep sensitivity 94.5%, wake specificity 67.0%.
    Total sleep time is overestimated by 17.1 minutes on average.
- **Stage durations have larger errors than total sleep time.** On SleepAccel
  (cross-validation), the mean absolute percentage error is 3.8% for total
  sleep time, 17.9% for light sleep, 78.5% for deep sleep and 50.5% for REM
  sleep.
- **Deep sleep is overestimated in cohorts with little deep sleep.**
  - Deep sleep is 25.5% of predicted sleep vs 11.0% of PSG sleep on
    SleepAccel-Clinical, and 22.7% vs 16.2% on SleepAccel (cross-validation),
    as the mean over nights.
  - The divisor assumes the adult training recordings' 19.7% deep sleep.
    Cohorts with less deep sleep, such as older adults or people with sleep
    apnea, are overestimated more.
  - Do not use deep-sleep time as a per-night outcome.
- **Single nights differ more than cohort means.** 95% limits of agreement for
  total sleep time are −64 to +99 minutes on SleepAccel-Clinical.
- **Only the PSG night has been evaluated.** Daytime sleep, naps and non-wear
  periods in multi-day recordings have not been compared against a reference.
- **Device and placement.** The model was trained on wrist data from
  ActiGraph, Apple Watch, Empatica E4, Fitbit and Garmin devices, sampled at
  12 Hz or faster. Adult recordings in training and evaluation are from Apple
  Watch (SleepAccel, SleepAccel-Clinical) and Empatica E4 (Wearanize+).

## Training data

- **Weaver _et al_** ([PubMed 40676371](https://pubmed.ncbi.nlm.nih.gov/40676371/)):
  634 recordings from 237 children with no, mild, moderate and severe
  obstructive sleep apnea (OSA), each night recorded by up to four wrist
  devices (ActiGraph, Apple Watch, Fitbit and Garmin).
- **[SleepAccel](https://pubmed.ncbi.nlm.nih.gov/31579900/)**: 28 healthy
  adults, Apple Watch.
- **Wearanize+**: 92 nights from adults, Empatica E4, with manual PSG scoring.

Not trained on:

- **[SleepAccel-Clinical](https://doi.org/10.1093/sleepadvances/zpag051)**:
  32 adults with no, mild, moderate and severe OSA; evaluation only.

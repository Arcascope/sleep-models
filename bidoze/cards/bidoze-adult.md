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
random seeds. Each network scores 8-hour windows that start every hour
(`window_step` in `config.json`, in 30-s epochs). An epoch's probabilities are
the mean over the three networks and over the windows that cover it, with
windows weighted down toward their edges. Each 30-s epoch is then labeled in
two steps:

1. **Sleep or wake:** the epoch is sleep when light + deep + REM probability
   ≥ wake probability (summed sleep probability ≥ 0.5); otherwise it is wake.
   Pass `argmax=True` (or `--argmax`) to call sleep only when the most probable
   of the four is light, deep or REM.
2. **Light, deep or REM**, on sleep epochs only: the light, deep and REM
   probabilities are each divided by a fixed stage divisor, stored as
   `stage_divisor` in `config.json`; the largest quotient gives the stage.

The divisor corrects for two properties of the training: the loss weights rare
stages up, and most training recordings are of children, who spend a larger
share of sleep in deep sleep than adults (33% of PSG sleep in the pediatric
training data). The adult stage proportions in the divisor are those of the
adult training recordings, SleepAccel and Wearanize+ (deep sleep 19.7% of PSG
sleep).

The bundle also holds a small sleep/wake tree (`sleep_wake_tree.safetensors`,
the [`Arcascope/arcasleep-tree-adult-accel`](https://huggingface.co/Arcascope/arcasleep-tree-adult-accel)
model). On recordings longer than 9 hours, the sleep filter keeps model sleep
outside the main sleep period only where this tree also calls sleep
(`is_tree_sleep == 1`).

Output columns:

- `wake`: the averaged probability of wake.
- `light`, `deep`, `rem`: 1 − `wake`, split in proportion to the divided
  probabilities.
- `stage`: the label from steps 1 and 2 after the sleep filter. This is not
  always the largest of the four columns.

## Agreement with two human scorers

Wearanize+ nights were each scored by two independent human scorers. On the
90 nights with both scorings, scorer 2's sleep/wake κ against scorer 1 is
0.602, with wake specificity 95.2%. BiDoze cross-validation (each night scored
by models trained on the other Wearanize+ participants):

| Reference | Sleep/wake κ | Model − scorer 2 vs scorer 1 | Wake specificity |
|:---|--:|:---|--:|
| Scorer 1 | 0.636 | +0.034 (95% CI +0.003 to +0.065) | 79.8% |
| Scorer 2 | 0.609 | +0.008 (95% CI −0.021 to +0.036) | 56.9% |

- Against scorer 1 the model's κ is higher than the second scorer's (the
  interval excludes 0); against scorer 2 the difference is not significant.
- Wake specificity is far below the 95.2% of scorer 2 against scorer 1.
  Scorer 2 calls more wake than scorer 1, so wake specificity against scorer 2
  is lower for any model.
- Scorer 2's epochs are moved one epoch (30 s) to line up with scorer 1, the
  shift at which the two scorers agree most. Intervals are 95% participant
  bootstrap intervals of the per-night difference.

## Limitations

- **Wake specificity is lower than sleep sensitivity.**
  - SleepAccel cross-validation: sleep sensitivity 96.8%, wake specificity
    68.6%.
  - Wearanize+ cross-validation: sleep sensitivity 95.6%, wake specificity
    79.4%.
  - AAUWSS cross-validation: sleep sensitivity 95.7%, wake specificity 62.3%.
  - SleepAccel-Clinical: sleep sensitivity 96.2%, wake specificity 59.9%.
    Total sleep time is overestimated by 27.3 minutes on average.
- **Stage durations have larger errors than total sleep time.** On SleepAccel
  (cross-validation), the mean absolute percentage error is 18.5% for light
  sleep, 88.7% for deep sleep and 48.2% for REM sleep.
- **Deep sleep is overestimated in cohorts with little deep sleep.**
  - Deep sleep is 21.3% of predicted sleep vs 11.0% of PSG sleep on
    SleepAccel-Clinical, and 21.8% vs 15.9% on SleepAccel (cross-validation),
    as the mean over nights.
  - The divisor assumes the adult training recordings' 19.7% deep sleep.
    Cohorts with less deep sleep, such as older adults or people with sleep
    apnea, are overestimated more.
  - Do not use deep-sleep time as a per-night outcome.
- **Single nights differ more than cohort means.** 95% limits of agreement for
  total sleep time are −55 to +110 minutes on SleepAccel-Clinical.
- **Only the PSG night has been evaluated.** Daytime sleep, naps and non-wear
  periods in multi-day recordings have not been compared against a reference.
- **Device and placement.** The model was trained on wrist data from
  ActiGraph, Apple Watch, Empatica E4, Fitbit and Garmin devices, sampled at
  12 Hz or faster. Adult recordings in training and evaluation are from Apple
  Watch (SleepAccel, SleepAccel-Clinical) and Empatica E4 (Wearanize+,
  AAUWSS).

## Training data

- **Weaver _et al_** ([PubMed 40676371](https://pubmed.ncbi.nlm.nih.gov/40676371/)):
  609 recordings from 237 children with no, mild, moderate and severe
  obstructive sleep apnea (OSA), each night recorded by up to four wrist
  devices (ActiGraph, Apple Watch, Fitbit and Garmin). 25 Fitbit recordings
  whose timestamps do not line up with the PSG are left out.
- **[SleepAccel](https://pubmed.ncbi.nlm.nih.gov/31579900/)**: 28 healthy
  adults, Apple Watch.
- **Wearanize+**: 92 nights from adults, Empatica E4, with manual PSG scoring.
- **AAUWSS**: 13 healthy adults aged 22–71, one lab night each, Empatica E4.
  The 13 nights are weighted so that together they count as much as the
  Wearanize+ nights the model fits on.

Not trained on:

- **[SleepAccel-Clinical](https://doi.org/10.1093/sleepadvances/zpag051)**:
  32 adults with no, mild, moderate and severe OSA; evaluation only.

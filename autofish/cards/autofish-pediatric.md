## Intended use

Research estimates of sleep and wake, and of light, deep and REM sleep, from a
wrist-worn accelerometer in children and adolescents aged 5 to 18. You need no
polysomnography (PSG) to run it; PSG is only needed to check its accuracy on
your own cohort.

Sleep/wake is the primary task. Stage estimates are less accurate and are best
read as cohort-level trends, not night-by-night measurements.

**Not a medical device.** Do not use it to diagnose sleep disorders (including
obstructive sleep apnea), to screen individuals, or to make clinical decisions.

## Limitations

- **It tends to overestimate sleep.** Like other actigraphy methods, it
  identifies sleep more reliably than wake: still, quiet wakefulness is often
  scored as sleep. In cross-validation, total sleep time was overestimated on
  about three nights in four, by a median of 12.5 minutes on those nights. The
  median error across all nights was +8 minutes, 88% of nights were within
  30 minutes of PSG, and 2% were overestimated by more than an hour.
- **Stages are harder than sleep/wake.** Light, deep and REM durations carry
  much larger errors than total sleep time; REM is the least reliable.
- **Adults are outside the intended population.** Wake specificity and κ are
  much lower in the adult cohorts, and deep sleep is overestimated in healthy
  adults. Validate on your own data before using it with adults.
- **Device and placement.** It was trained on wrist data from Apple Watch and
  Fitbit devices, sampled at 12 Hz or faster. Hip, ankle and upper-arm
  placements have not been evaluated.
- **Excluded recordings.** Two SleepAccel nights sampled at 10 Hz fall below
  the model's minimum sampling rate, so they were not scored.

## Training data

Trained on the pediatric cohort of Weaver _et al_
([PubMed 40676371](https://pubmed.ncbi.nlm.nih.gov/40676371/)), a clinical sample spanning
no, mild, moderate and severe obstructive sleep apnea. SleepAccel (Walch _et al_, 2019)
healthy adults and the SleepAccel-Clinical adults with sleep apnea were held out
for external testing only.

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

- **It tends to overestimate sleep.** Like other actigraphy methods, it
  identifies sleep more reliably than wake: still, quiet wakefulness is often
  scored as sleep.
- **Stages are harder than sleep/wake.** Light, deep and REM durations carry
  much larger errors than total sleep time; REM is the least reliable.
- **Adults are outside the intended population.** Wake specificity and κ are
  lower in the adult cohorts, and deep sleep is overestimated in healthy
  adults. Validate on your own data before using it with adults.
- **Device and placement.** It was trained on wrist data from Apple Watch and
  Fitbit devices, sampled at 12 Hz or faster. Hip, ankle and upper-arm
  placements have not been evaluated.

## Training data

Trained on the pediatric cohort of Weaver _et al_
([PubMed 40676371](https://pubmed.ncbi.nlm.nih.gov/40676371/)), a clinical sample spanning
no, mild, moderate and severe obstructive sleep apnea. 

Two adult cohorts were additoinally used for evaluating the trained model:
1. SleepAccel: healthy adults ([Walch _et al_ 2019](https://pubmed.ncbi.nlm.nih.gov/31579900/))
2. SleepAccel-Clinical: adults, with no, mild, moderate, and severe OSA  ([Canton, Tavella, _et al_, 2026](https://doi.org/10.1093/sleepadvances/zpag051))

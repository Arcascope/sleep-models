# Heart rate as a light/deep/REM feature for the adult release

**Question.** Does heart rate from the SeqSmoother reproduction add light/deep/REM information
beyond `arcasleep-bidoze-adult` v0.1.0's own output?

**HR per 30-s PSG epoch**, each from the centred 60-s window:
- SeqSmoother (`hr_models.py`): retrained on the folds of `sleepaccel_5fold.json`, so a
  SleepAccel subject's HR comes from a model that never trained on its fold. SleepAccel-Clinical
  has no recorded HR; its HR comes from a model trained on all of SleepAccel
  (`clinical_prepare.py` builds the input caches).
- Jerk peak (`epoch_hr.py`): the peak of the jerk power in 0.9–1.3 Hz; jerk from senpy's
  `compute_jerk`, power from `compute_nufft_spectrogram`.
- Watch (SleepAccel only): the median of the Apple Watch's recorded HR.

**Diagnostic** (`readout.py`, `ablation.py`). On the epochs the release calls sleep, a
multinomial logistic readout picks light, deep or REM from the release's cross-validated
log-probabilities, with or without HR features. The release's sleep/wake calls are unchanged.
SleepAccel is cross-validated on the BiDoze folds; SleepAccel-Clinical uses the readout trained on
all of SleepAccel. No input describes time since recording start or position in the night.

**Running.** `setup/run.sh` in the results directory reproduces the run.
`release_divisors.py` needs the `adult-formulation` checkout on `PYTHONPATH`.

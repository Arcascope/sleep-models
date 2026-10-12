# Retained axes: readout and matched seed-0 learning

`readout.py` compares frozen ensemble score alone, score plus existing magnitude/jerk
band summaries, score plus raw trace/covariance summaries, and the combined inputs.
Fixed logistic regression C=1; vessel-held-out fits; scalers trained only on fit;
inner-group thresholds target 95% participant-macro sleep sensitivity for raw/GGIR calls.
Trace/covariance summaries preserve some rotation-invariant information. This is a
limited epoch-level probe, not an information ceiling for raw acceleration.

`prepare_trace.py` preserves the original cached inputs and labels bitwise and appends
one trace spectrum channel: sum linear per-axis PSDs per frame, log, then mean frames.
It audits class/clock alignment against existing aligned fused land exports and source
Clinical data; original gap fractions must agree. Research checkpoints use custom
cached inputs, not a packaged raw-data predictor.

`train.py` runs a matched scratch comparison on original and appended-trace inputs.
Use seed 0 for initialization, shuffle and dropout; fixed splits, batch 2, balanced CE
beta .999, initial LR .001, halving after 700 plateau steps, early stop 1750,
evaluation smoothing 350. Select the best unsmoothed inner stage kappa. Cap screening
at 100 epochs. Shape changes affect initialization despite matching seeds; promising
results need replication. `score_training.py` compares both learned arms and the frozen
released seed 0, with physical-gap decoding and recording-then-participant aggregation.
A linear-readout null does not veto matched neural retraining.

All cohort-bearing code is confidential-work only. Register outputs before S3 sync;
use only the confidential artifact destination specified by results-s3-sync.

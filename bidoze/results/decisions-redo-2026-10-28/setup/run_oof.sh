#!/bin/sh
# Task 1: 8 h window logits for the SleepAccel test nights, then out-of-fold predictions and Table A.
W=/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders
R=$W/bidoze/results/decisions-redo-2026-10-28
export PYTHONPATH=$W/core/src:$W/bidoze/src XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
PY=/home/eric/venvs/arcasleep/bin/python
cd $R
$PY setup/oof_tasks.py infer >> logs/infer.log 2>&1 && $PY setup/oof_tasks.py score >> logs/score.log 2>&1

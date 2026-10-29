#!/bin/sh
# Task 3: score the existing final 8 h checkpoints (no retraining) with probability-average blending. Usage: setup/run_final.sh [--smoke N]
W=/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders
R=$W/bidoze/results/decisions-redo-2026-10-28
export PYTHONPATH=$W/core/src:$W/bidoze/src XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
PY=/home/eric/venvs/arcasleep/bin/python
mkdir -p $R/final8h/logs; cd $R
if [ "$1" = "--smoke" ]; then EXTRA="--smoke ${2:-2} --no-shortcut"; TAG=smoke; else EXTRA=""; TAG=full; fi
$PY setup/score_final_redo.py score $EXTRA >> final8h/logs/score-$TAG.log 2>&1 && $PY setup/score_final_redo.py analyze $EXTRA >> final8h/logs/analyze-$TAG.log 2>&1 && echo "FINAL_DONE $TAG" >> final8h/logs/done-$TAG.log

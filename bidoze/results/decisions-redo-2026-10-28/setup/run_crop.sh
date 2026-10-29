#!/bin/sh
# Task 2: second pass with probability-average blending. Usage: setup/run_crop.sh [--smoke N]
W=/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders
R=$W/bidoze/results/decisions-redo-2026-10-28
export PYTHONPATH=$W/core/src:$W/bidoze/src XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
PY=/home/eric/venvs/arcasleep/bin/python
mkdir -p $R/crop/logs
if [ "$1" = "--smoke" ]; then EXTRA="--smoke ${2:-3} --folds 0 --no-shortcut"; TAG=smoke; else EXTRA=""; TAG=full; fi
cd $R
$PY setup/crop_redo.py score $EXTRA >> crop/logs/score-$TAG.log 2>&1 && \
$PY setup/crop_redo.py foundry ${EXTRA#--folds 0} >> crop/logs/foundry-$TAG.log 2>&1 ; \
$PY setup/crop_redo.py analyze $EXTRA >> crop/logs/analyze-$TAG.log 2>&1 && echo "CROP_DONE $TAG" >> crop/logs/done-$TAG.log

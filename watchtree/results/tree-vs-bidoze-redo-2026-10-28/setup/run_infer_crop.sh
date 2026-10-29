#!/bin/bash
W=/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders
R=/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree/results/tree-vs-bidoze-redo-2026-10-28
export PYTHONPATH=$W/core/src:$W/bidoze/src XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $R/setup
/home/eric/venvs/arcasleep/bin/python -B infer_8h_crop.py > $R/logs/infer_8h_crop.log 2>&1

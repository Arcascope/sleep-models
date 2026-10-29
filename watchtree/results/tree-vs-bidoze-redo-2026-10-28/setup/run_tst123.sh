#!/bin/bash
W=/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders
T=/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees
R=$T/watchtree/results/tree-vs-bidoze-redo-2026-10-28
export PYTHONPATH=$W/core/src:$W/bidoze/src:$T/watchtree/src CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $R/setup
/home/eric/venvs/arcasleep/bin/python -B "$1".py 2>&1 | grep -v -E "Jax plugin|An NVIDIA|RuntimeError" | tee $R/logs/"$1".log

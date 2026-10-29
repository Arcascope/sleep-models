#!/bin/bash
# Section 12 redo. Reuses the original run's BiDoze release-model probabilities for the other cohorts (read only).
set -euo pipefail
T=/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees
R=$T/watchtree/results/tree-vs-bidoze-redo-2026-10-28
export PYTHONPATH=$T/watchtree/src:$T/core/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 WATCHTREE_DATA_ROOT="/media/eric/Extreme SSD/pisces2-data"
mkdir -p $R/duration
cd $R/setup
/home/eric/venvs/arcasleep/bin/python -B duration_redo.py --out $R/duration --probs $T/watchtree/results/duration-exp3a-2026-10-28/bidoze_probs.parquet \
  --oof /home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders/bidoze/results/decisions-redo-2026-10-28/oof 2>&1 | tee $R/logs/duration.log

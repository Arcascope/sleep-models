#!/bin/bash
# Tree predictions of all arms (reads the ablation-exp3b models; at most 10 LightGBM threads).
set -euo pipefail
WT=/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees
export PYTHONPATH=$WT/watchtree/src:$WT/core/src OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
R=$WT/watchtree/results/tree-vs-bidoze-redo-2026-10-28
cd $R/setup
/home/eric/venvs/arcasleep/bin/python -B build_tree_preds.py 2>&1 | tee $R/logs/build_tree_preds.log

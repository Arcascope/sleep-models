#!/bin/bash
# Context ablation: WatchTree backward/forward reach arms on BiDoze v0.2.0 folds, 3 seeds, offsets decoding.
#   ./run.sh              build features, fit (resumable), score, figure
#   SMOKE=1 ./run.sh      fold0, current arm + (30,5), seed 0 (output in ../smoke)
# JOBS = parallel fit jobs, THREADS = LightGBM threads per job (4 threads total to start; <= 12 once load is low).
set -euo pipefail
WT=/home/eric/Engineering/Work/autopisces/sleep-models-tree-context
export PYTHONPATH=$WT/watchtree/src:$WT/core/src
PY=/home/eric/venvs/arcasleep/bin/python
R=$WT/watchtree/results/context-ablation-2026-10-28
JOBS=${JOBS:-1}; THREADS=${THREADS:-4}
cd $R/setup
export CTX_OUT=$R
mkdir -p $R/logs
$PY -B build_arm_features.py > $R/logs/build.log 2>&1
for f in fold0 fold1 fold2 fold3 fold4; do for a in b120_f45 b60_f45 b30_f45 b15_f45 b5_f45 b120_f15 b120_f5 b120_f0 b60_f15 b30_f15 b30_f5 b15_f0; do echo "$f $a"; done; done |
  xargs -P $JOBS -L 1 sh -c "$PY -B fit_context.py --phase \$0 --arms \$1 --seeds 0,1,2 --threads $THREADS >> $R/logs/fit-\$0-\$1.log 2>&1"
$PY -B score_context.py --draws 5000 | tee $R/logs/score.log
$PY -B costs_table.py | tee $R/logs/costs.log
$PY -B plot_context.py

#!/bin/bash
# Step 4: refit nohr_ahr and hr_bpm_ahr (3 seeds, 10 phases) on the ahr_* columns of the chosen estimator, and score paired against
# nohr / hr_bpm (their models and the BPM columns are the hr-offerings-2026-10-28 ones, linked, not refit). One variant per call:
#   ./run_step4.sh s4_ungated   |   ./run_step4.sh s4_gate50
set -euo pipefail
. "$(dirname "$0")/env.sh"
V=$1
export EXP4_OUT=$R/$V EXP4_SMOKE=0
mkdir -p $EXP4_OUT/models $EXP4_OUT/logs
ln -sfn $OLD/features $EXP4_OUT/features
for f in $OLD/models/*_nohr_s?.* $OLD/models/*_hr_bpm_s?.*; do ln -sf $f $EXP4_OUT/models/$(basename $f); done
cd $R/setup
for p in fold0 fold1 fold2 fold3 fold4 wv0 wv1 wv2 wv3 wv4; do echo $p; done |
  xargs -P 2 -L 1 sh -c "$PY -B $OLD/setup/fit_bpm.py --phase \$0 --arms nohr_ahr,hr_bpm_ahr --seeds 0,1,2 --threads 3 >> $EXP4_OUT/logs/fit-\$0.log 2>&1"
$PY -B $OLD/setup/score_bpm.py --arms nohr,nohr_ahr,hr_bpm,hr_bpm_ahr --seeds 0,1,2 --draws 5000 2>&1 | tee $EXP4_OUT/logs/score.log
date -Is > $EXP4_OUT/DONE

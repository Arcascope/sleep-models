#!/bin/bash
# HR offerings redo of the 2026-10-28 HR-feature work. CPU only; at most 12 LightGBM threads (4 jobs x 3).
#   ./run.sh            resumable; stops at once if DONE exists
# 1 build_bpm.py (BPM columns, SleepAccel at native rate)  2 build_accelhr.py (accelerometer-derived HR, 10/20/30 s windows)
# 3 validate_accelhr.py  4 build_ahr_features.py --window 30  5 fits (10 phases x 5 arms x 3 seeds)  6 score  7 figures
set -euo pipefail
. "$(dirname "$0")/env.sh"
if [ -e $R/DONE ]; then echo "already complete"; exit 0; fi
mkdir -p $R/logs $R/models $R/figures
cd $R/setup
$PY -B build_bpm.py --workers 4 2>&1 | tee -a $R/logs/build_bpm.log
$PY -B build_accelhr.py --workers 6 2>&1 | tee -a $R/logs/build_accelhr.log
$PY -B compare_old_new.py > $R/logs/compare_old_new.log 2>&1
$PY -B validate_accelhr.py > $R/logs/validate_accelhr.log 2>&1
$PY -B build_ahr_features.py --window 30 2>&1 | tee -a $R/logs/build_ahr_features.log
for p in fold0 fold1 fold2 fold3 fold4 wv0 wv1 wv2 wv3 wv4; do echo $p; done |
  xargs -P 4 -L 1 sh -c "$PY -B fit_bpm.py --phase \$0 --seeds 0,1,2 --threads 3 >> $R/logs/fit-\$0.log 2>&1"
$PY -B score_bpm.py --seeds 0,1,2 --draws 5000 2>&1 | tee $R/logs/score.log
$PY -B figures.py --window 30 2>&1 | tee $R/logs/figures.log
date -Is > $R/DONE

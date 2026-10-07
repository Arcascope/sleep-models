#!/usr/bin/env bash
# Train every arm and fold of the device ablation; one call per arm and fold trains seeds 0-2
# (setup/training.json: run_seed 0, seed_count 3). A fold whose results exist is skipped.
# Usage: run.sh <run dir> [arm ...]   (default: every arm, in the order below)
# The run dir holds setup/ from prepare.py; the code is this checkout.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$(cd "$HERE/../../.." && pwd)"
RUN="$(cd "$1" && pwd)"; shift
S=$RUN/setup
ARMS=("${@:-all apple-119 fitbit-119 garmin-119 actigraph-119 actigraph}")
export PYTHONPATH=$CODE/core/src:$CODE/bidoze/src PYTHONUNBUFFERED=1
cd "$CODE/bidoze"
echo "code $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- ../core ../bidoze/src || echo -dirty)"
for arm in ${ARMS[@]}; do
  for k in 0 1 2 3 4; do
    out=$RUN/$arm/fold$k
    if ls "$out"/cv_results_*_stages.csv > /dev/null 2>&1; then echo "skip $arm fold$k"; continue; fi
    mkdir -p "$out"
    echo "==> $arm fold$k $(date +%T)"
    FEATURE_CACHE_DIR=$S/cache/$arm arcasleep bidoze train --cache-only -pc config/inputs.json \
      -mc config/model.json -tc $S/training.json -mtc config/metrics.json --split-seed 0 \
      -cvc $S/cv/$arm/cv_fold$k.json -extn "$out" > "$out/training.log" 2>&1 || echo "FAILED $arm fold$k"
    grep "best eval kappa\|early stop" "$out/training.log"
  done
done
echo "done $(date +%T)"

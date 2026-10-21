#!/usr/bin/env bash
# Train every draw and arm of the joint-ensemble experiment, then score each joint fit's members
# by themselves. A draw and arm whose predictions are written is skipped.
# Usage: run.sh <run dir> [draw ...]   (default: the draws in setup/draws.json)
# The run dir holds setup/ from prepare.py; the code is this checkout.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$(cd "$HERE/../../.." && pwd)"
RUN="$(cd "$1" && pwd)"; shift
S=$RUN/setup
DRAWS=("${@:-$(python -c "import json,sys;print(' '.join(k[4:] for k in json.load(open(sys.argv[1]))))" "$S/draws.json")}")
ARMS=(${JOINT_ENSEMBLE_ARMS:-joint3 independent joint2})
export PYTHONPATH=$CODE/core/src:$CODE/bidoze/src PYTHONUNBUFFERED=1
cd "$CODE/bidoze"
echo "code $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- ../core ../bidoze/src || echo -dirty)"
for d in ${DRAWS[@]}; do
  for arm in ${ARMS[@]}; do
    out=$RUN/$arm/draw$d
    last=$(python -c "import json,sys;t=json.load(open(sys.argv[1]));print(t['run_seed']+t['seed_count']-1)" "$S/training-$arm.json")
    if [[ -d "$out/predictions/draw${d}_seed_$last" ]]; then echo "skip $arm draw$d"; continue; fi
    mkdir -p "$out"
    echo "==> $arm draw$d $(date +%T)"
    FEATURE_CACHE_DIR=$S/cache/draw$d arcasleep bidoze train --cache-only -pc config/inputs.json \
      -mc config/model.json -tc "$S/training-$arm.json" -mtc config/metrics.json --split-seed 0 \
      -cvc "$S/cv/draw$d.json" -extn "$out" > "$out/training.log" 2>&1 || { echo "FAILED $arm draw$d"; continue; }
    grep "best eval kappa\|early stop\|trained in" "$out/training.log"
    if [[ $arm == joint* ]]; then
      python "$HERE/member_predictions.py" "$RUN" "$arm" "$d" >> "$out/training.log" 2>&1 || echo "FAILED members $arm draw$d"
    fi
  done
done
echo "done $(date +%T)"

#!/usr/bin/env bash
# One phase of the device ablation in a training image: PHASE is <arm>/fold<k>
# (early-stopping split seed 0) or <arm>/split<s>/fold<k>.
# Called by scripts/harness.sh with BIDOZE_EXPERIMENT=experiments/adult_devices;
# FEATURE_CACHE_DIR is the image's cache root, which holds every cohort.
# ADULT_DEVICES_TRAINING overrides the recipe, e.g. config/training-smoke.json
# for a wiring check.
# Usage: task.sh PHASE OUT_DIR
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE=$1
OUT=$2
K=${PHASE##*/fold}
ARM=${PHASE%/fold*}
SPLIT=0
if [[ "$ARM" == */split* ]]; then
  SPLIT=${ARM##*/split}
  ARM=${ARM%/split*}
fi
[[ "$K" =~ ^[0-4]$ && "$SPLIT" =~ ^[0-9]+$ && "$ARM" != */* ]] \
  || { echo "phase must be <arm>/fold<0-4> or <arm>/split<s>/fold<0-4>: $PHASE" >&2; exit 2; }

WORK=${BIDOZE_WORK_DIR:-${TMPDIR:-/tmp}/adult-devices}
python "$HERE/prepare.py" "$WORK" "$FEATURE_CACHE_DIR" "$FEATURE_CACHE_DIR"
S=$WORK/setup
[[ -d "$S/cache/$ARM" ]] || { echo "unknown arm: $ARM" >&2; exit 2; }
TRAINING=${ADULT_DEVICES_TRAINING:-$S/training.json}
mkdir -p "$OUT/setup"
cp "$S/arms.json" "$TRAINING" "$S/cv/$ARM/cv_fold$K.json" "$OUT/setup/"

FEATURE_CACHE_DIR=$S/cache/$ARM arcasleep bidoze train --cache-only -pc config/inputs.json \
  -mc config/model.json -tc "$TRAINING" -mtc config/metrics.json --split-seed "$SPLIT" \
  -cvc "$S/cv/$ARM/cv_fold$K.json" -extn "$OUT"

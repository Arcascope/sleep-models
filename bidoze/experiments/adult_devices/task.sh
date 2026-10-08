#!/usr/bin/env bash
# One phase of the device ablation in a training image: PHASE is <arm>/fold<k>
# (early-stopping split seed 0) or <arm>/split<s>/fold<k>; <arm>/final trains the
# released stage source on all of SleepAccel. A trailing /seed<n> trains only that
# seed, so the seeds of one fold can run as separate tasks.
# Called by scripts/harness.sh with BIDOZE_EXPERIMENT=experiments/adult_devices;
# FEATURE_CACHE_DIR is the image's cache root, which holds every cohort.
# ADULT_DEVICES_TRAINING overrides the recipe, e.g. config/training-smoke.json
# for a wiring check. ADULT_DEVICES_PEDIATRIC is the pediatric release bundle,
# which the finetune arm starts from.
# Usage: task.sh PHASE OUT_DIR
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE=$1
OUT=$2
SEED=
if [[ "$PHASE" == */seed* ]]; then
  SEED=${PHASE##*/seed}
  PHASE=${PHASE%/seed*}
  [[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a number: $SEED" >&2; exit 2; }
fi
if [[ "$PHASE" == */final ]]; then
  K=final
  ARM=${PHASE%/final}
else
  K=${PHASE##*/fold}
  ARM=${PHASE%/fold*}
fi
SPLIT=0
if [[ "$ARM" == */split* ]]; then
  SPLIT=${ARM##*/split}
  ARM=${ARM%/split*}
fi
[[ "$K" =~ ^([0-4]|final)$ && "$SPLIT" =~ ^[0-9]+$ && "$ARM" != */* ]] \
  || { echo "phase must be <arm>/fold<0-4>, <arm>/split<s>/fold<0-4> or <arm>/final: $PHASE" >&2; exit 2; }

WORK=${BIDOZE_WORK_DIR:-${TMPDIR:-/tmp}/adult-devices}
python "$HERE/prepare.py" "$WORK" "$FEATURE_CACHE_DIR" "$FEATURE_CACHE_DIR" ${ADULT_DEVICES_PEDIATRIC:+"$ADULT_DEVICES_PEDIATRIC"}
S=$WORK/setup
[[ -d "$S/cache/$ARM" ]] || { echo "unknown arm: $ARM" >&2; exit 2; }
RECIPE=$S/training.json
[[ -f "$S/training-$ARM.json" ]] && RECIPE=$S/training-$ARM.json
TRAINING=${ADULT_DEVICES_TRAINING:-$RECIPE}
if [[ -n "$SEED" ]]; then
  mkdir -p "$OUT/setup"
  python -c "import json,sys;t=json.load(open(sys.argv[1]));t.update(run_seed=int(sys.argv[2]),seed_count=1);json.dump(t,open(sys.argv[3],'w'),indent=2)" \
    "$TRAINING" "$SEED" "$OUT/setup/training-seed$SEED.json"
  TRAINING=$OUT/setup/training-seed$SEED.json
fi
CV=$S/cv/$ARM/cv_$([[ $K == final ]] && echo final || echo fold$K).json
mkdir -p "$OUT/setup"
cp "$S/arms.json" "$TRAINING" "$CV" "$OUT/setup/"

FEATURE_CACHE_DIR=$S/cache/$ARM arcasleep bidoze train --cache-only -pc config/inputs.json \
  -mc config/model.json -tc "$TRAINING" -mtc config/metrics.json --split-seed "$SPLIT" \
  -cvc "$CV" -extn "$OUT"

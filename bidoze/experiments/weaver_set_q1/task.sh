#!/usr/bin/env bash
# One phase of the Weaver-set experiment in a training image: PHASE is <arm>/draw<d> (the recipe's
# 3 seeds, one after another) or <arm>/draw<d>/seed<n> (that seed only). Arms: matched, all,
# all-adult-share. Called by scripts/harness.sh with BIDOZE_EXPERIMENT=experiments/weaver_set_q1;
# FEATURE_CACHE_DIR is the image's cache root, which holds every cohort.
# WEAVER_SET_Q1_TRAINING overrides the recipe, e.g. config/training-smoke.json for a wiring check.
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
ARM=${PHASE%/draw*}
DRAW=${PHASE##*/draw}
[[ "$ARM" =~ ^(matched|all|all-adult-share)$ && "$DRAW" =~ ^[0-4]$ ]] \
  || { echo "phase must be <arm>/draw<0-4>[/seed<n>], arm matched, all or all-adult-share: $1" >&2; exit 2; }

WORK=${BIDOZE_WORK_DIR:-${TMPDIR:-/tmp}/weaver-set-q1}/$ARM/draw$DRAW
rm -rf "$WORK"
python "$HERE/prepare.py" "$WORK" "$FEATURE_CACHE_DIR" "$ARM" "$DRAW" ${WEAVER_SET_Q1_TRAINING:+"$WEAVER_SET_Q1_TRAINING"}
TRAINING=$WORK/training.json
mkdir -p "$OUT/setup"
if [[ -n "$SEED" ]]; then
  python -c "import json,sys;t=json.load(open(sys.argv[1]));t.update(run_seed=int(sys.argv[2]),seed_count=1);json.dump(t,open(sys.argv[3],'w'),indent=2)" \
    "$TRAINING" "$SEED" "$WORK/training-seed.json"
  TRAINING=$WORK/training-seed.json
fi
cp "$WORK/cv.json" "$WORK/recordings.json" "$TRAINING" "$OUT/setup/"

FEATURE_CACHE_DIR=$WORK/cache arcasleep bidoze train --cache-only -pc config/inputs.json \
  -mc config/model-gap-rule.json -tc "$TRAINING" -mtc config/metrics.json --split-seed 0 \
  -cvc "$WORK/cv.json" -extn "$OUT"

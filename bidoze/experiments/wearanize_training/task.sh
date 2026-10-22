#!/usr/bin/env bash
# One phase of the Wearanize+ training experiment in a training image: PHASE is <arm>/draw<d> (the
# recipe's 3 seeds, one after another) or <arm>/draw<d>/seed<n> (that seed only). Arms: g-base,
# g-wearanize, f2-base, f2-wearanize (prepare.py); g-* trains BiDoze-G, f2-* the strand-fusion network.
# Called by scripts/harness.sh with BIDOZE_EXPERIMENT=experiments/wearanize_training;
# FEATURE_CACHE_DIR is the image's cache root, which holds every cohort.
# WEARANIZE_TRAINING overrides the recipe, e.g. config/training-smoke.json for a wiring check.
# WEARANIZE_BASE_RUN is a finished Weaver-set Q1 run directory. g-base takes its checkpoints
# (<run>/training/all/draw<d>/model_outputs) instead of training: the fit loads them and only scores
# the draw's test sets, so g-base needs this variable and refuses to train without it.
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
[[ "$ARM" =~ ^(g|f2)-(base|wearanize)$ && "$DRAW" =~ ^[0-4]$ ]] \
  || { echo "phase must be <arm>/draw<0-4>[/seed<n>], arm g-base, g-wearanize, f2-base or f2-wearanize: $1" >&2; exit 2; }

WORK=${BIDOZE_WORK_DIR:-${TMPDIR:-/tmp}/wearanize-training}/$ARM/draw$DRAW
rm -rf "$WORK"
python "$HERE/prepare.py" "$WORK" "$FEATURE_CACHE_DIR" "$ARM" "$DRAW" ${WEARANIZE_TRAINING:+"$WEARANIZE_TRAINING"}
TRAINING=$WORK/training.json
MODEL=config/model-gap-rule.json
[[ "$ARM" == f2-* ]] && MODEL=config/model-f2.json
mkdir -p "$OUT/setup"
if [[ -n "$SEED" ]]; then
  python -c "import json,sys;t=json.load(open(sys.argv[1]));t.update(run_seed=int(sys.argv[2]),seed_count=1);json.dump(t,open(sys.argv[3],'w'),indent=2)" \
    "$TRAINING" "$SEED" "$WORK/training-seed.json"
  TRAINING=$WORK/training-seed.json
fi
if [[ "$ARM" == g-base ]]; then
  : "${WEARANIZE_BASE_RUN:?g-base scores the checkpoints of a finished Q1 run: set WEARANIZE_BASE_RUN}"
  CHECKPOINTS=$WEARANIZE_BASE_RUN/training/all/draw$DRAW/model_outputs
  compgen -G "$CHECKPOINTS/model_draw${DRAW}_seed_*.pisces" > /dev/null \
    || { echo "no checkpoints in $CHECKPOINTS" >&2; exit 2; }
  mkdir -p "$OUT/model_outputs"
  cp "$CHECKPOINTS"/model_draw"$DRAW"_seed_${SEED:-*}.pisces "$OUT/model_outputs/"
fi
cp "$WORK/cv.json" "$WORK/recordings.json" "$TRAINING" "$OUT/setup/"

FEATURE_CACHE_DIR=$WORK/cache arcasleep bidoze train --cache-only -pc config/inputs.json \
  -mc "$MODEL" -tc "$TRAINING" -mtc config/metrics.json --split-seed 0 \
  -cvc "$WORK/cv.json" -extn "$OUT"

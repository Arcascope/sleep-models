#!/usr/bin/env bash
# One phase of the free-living experiment in a training image: PHASE is <arm>/<phase>, with
# <phase> a held-out Foundry person (eric, olivia, doug) or final; see prepare.py.
# Called by scripts/harness.sh with BIDOZE_EXPERIMENT=experiments/free_living;
# FEATURE_CACHE_DIR is the image's cache root, which holds every cohort.
# FREE_LIVING_TRAINING overrides the recipe, e.g. config/training-smoke.json for a wiring
# check; the arm's segment_placement, shift_consistency_weight and crop keys are kept.
# FREE_LIVING_SEEDS=<first>:<count> trains fit seeds first..first+count-1 instead of 0..2,
# e.g. 3:1 to replace a stalled seed; the eval holdout (--split-seed 0) is unchanged.
# Usage: task.sh PHASE OUT_DIR
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE=$1
OUT=$2
ARM=${PHASE%/*}
K=${PHASE##*/}
[[ "$PHASE" == */* && "$ARM" != */* ]] || { echo "phase must be <arm>/<phase>: $PHASE" >&2; exit 2; }

WORK=${BIDOZE_WORK_DIR:-${TMPDIR:-/tmp}/free-living}
python "$HERE/prepare.py" "$WORK" "$FEATURE_CACHE_DIR"
S=$WORK/setup
CV=$S/cv/$ARM/cv_$K.json
[[ -f "$CV" ]] || { echo "unknown phase: $PHASE" >&2; exit 2; }
TRAINING=$S/training-$ARM.json
mkdir -p "$OUT/setup"
if [[ -n "${FREE_LIVING_TRAINING:-}" ]]; then
  python -c "import json,sys;t=json.load(open(sys.argv[1]));a=json.load(open(sys.argv[2]));t.update({k:a[k] for k in ('segment_placement','shift_consistency_weight','crop_sets','crop_epochs') if k in a});json.dump(t,open(sys.argv[3],'w'),indent=2)" \
    "$FREE_LIVING_TRAINING" "$TRAINING" "$OUT/setup/training-override.json"
  TRAINING=$OUT/setup/training-override.json
else
  cp "$TRAINING" "$OUT/setup/"
fi
if [[ -n "${FREE_LIVING_SEEDS:-}" ]]; then
  python -c "import json,sys;t=json.load(open(sys.argv[1]));s,n=sys.argv[2].split(':');t.update(run_seed=int(s),seed_count=int(n));json.dump(t,open(sys.argv[3],'w'),indent=2)" \
    "$TRAINING" "$FREE_LIVING_SEEDS" "$OUT/setup/training-seeds.json"
  TRAINING=$OUT/setup/training-seeds.json
fi
cp "$CV" "$OUT/setup/"

FEATURE_CACHE_DIR=$S/cache/$ARM arcasleep bidoze train --cache-only -pc config/inputs.json \
  -mc config/model.json -tc "$TRAINING" -mtc config/metrics.json --split-seed 0 \
  -cvc "$CV" -extn "$OUT"

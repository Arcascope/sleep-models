#!/usr/bin/env bash
# BiDoze-G fits for the ensemble-size study, in a training image or locally.
#
# Usage: task.sh PHASE OUT_DIR
#   PHASE  one or more fits joined by "+", each <arm>-d<draw>, e.g. bidoze_g_s3-d1+bidoze_g_bag3-d1.
#          Arms: bidoze_g_s{3..6}, bidoze_g_bag{3..6}; draws 0-4. Fits run one after another;
#          a failed fit does not stop the rest, and the exit status is non-zero if any failed.
#   OUT_DIR  the run root; each fit writes <OUT_DIR>/<arm>/draw<d>/ as burst.sh does:
#          predictions/, model_outputs/, train_predictions/, config/, training.log.
#
# Called by scripts/harness.sh (working directory bidoze/) with BIDOZE_EXPERIMENT=experiments/ensemble-size.
# FEATURE_ROOT is the directory holding 12Hz_log_PSD_aligned/ (default: the image's cache).
# TASK_MAX_TRAIN_SECONDS=<n> caps each fit's training time, for a wiring check.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE=$1
OUT=$2
FEATURE_ROOT=${FEATURE_ROOT:-/opt/bidoze/feature_cache}
CACHE_NAME=12Hz_log_PSD_aligned
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
failed=()

fit() {
  local spec=$1 arm d run training
  arm=${spec%-d*}
  d=${spec##*-d}
  [[ "$d" =~ ^[0-4]$ && -f "$HERE/model-$arm.json" && -f "$HERE/training-$arm.json" ]] \
    || { echo "bad fit: $spec" >&2; return 2; }
  run=$OUT/$arm/draw$d
  mkdir -p "$run" "$OUT/setup/cv" "$OUT/setup/cache/draw$d/$CACHE_NAME"
  python - "$HERE/cache_map.json" "$d" "$OUT/setup/cache/draw$d/$CACHE_NAME" "$FEATURE_ROOT/$CACHE_NAME" <<'PY' || return 1
import json, os, sys
mapping, draw, target, source = sys.argv[1:5]
for name, original in json.load(open(mapping))[draw].items():
    link = os.path.join(target, name)
    if os.path.islink(link):
        os.remove(link)
    os.symlink(os.path.join(source, original), link)
PY
  cp "$HERE/cv/draw$d.json" "$OUT/setup/cv/draw$d.json"
  training=$HERE/training-$arm.json
  if [[ -n ${TASK_MAX_TRAIN_SECONDS:-} ]]; then
    training=$run/training-capped.json
    python -c "import json,sys;t=json.load(open(sys.argv[1]));t['max_train_seconds']=int(sys.argv[3]);json.dump(t,open(sys.argv[2],'w'),indent=2)" \
      "$HERE/training-$arm.json" "$training" "$TASK_MAX_TRAIN_SECONDS"
  fi
  FEATURE_CACHE_DIR=$OUT/setup/cache/draw$d arcasleep bidoze train --cache-only -pc config/inputs.json \
    -mc "$HERE/model-$arm.json" -tc "$training" -mtc config/metrics.json --split-seed 0 \
    -cvc "$OUT/setup/cv/draw$d.json" -extn "$run" >> "$run/training.log" 2>&1 || return 1
  python experiments/tridoze/train_predictions.py "$OUT" "$arm" "$d" >> "$run/training.log" 2>&1
}

IFS='+' read -ra FITS <<< "$PHASE"
for spec in "${FITS[@]}"; do
  start=$(date +%s)
  echo "==> $spec $(date +%T)"
  if fit "$spec"; then status=ok; else status=FAILED; failed+=("$spec"); fi
  echo "<== $spec $status in $(( $(date +%s) - start )) s"
done
if (( ${#failed[@]} )); then echo "failed fits: ${failed[*]}" >&2; exit 1; fi

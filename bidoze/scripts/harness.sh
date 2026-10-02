#!/usr/bin/env bash
# BiDoze recipe: k-fold CV over the cohort, then a final model on all of it.
#
# The folds are not computed here. config/splits/*.json records the literal
# subject ids per fold; this script only materialises them as linked datasets
# and trains one model per generated cv config.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

RUN_ID="${BIDOZE_RUN_ID:-bidoze-training}"
# BIDOZE_RESULTS_DIR is what `arcasleep bidoze train` itself reads; harness.sh reads
# the same variable so the two cannot disagree. Its fallback matches the CLI's
# ("results", relative) because we cd to REPO_ROOT just above.
RESULTS_ROOT="${BIDOZE_RESULTS_DIR:-$REPO_ROOT/results}"
SPLIT_PLAN="${BIDOZE_SPLIT_PLAN:-config/splits/weaver_5fold.json}"
# Linked fold datasets and their cv configs are derived artifacts: they are
# rebuilt from the committed plan on every run, and live outside the results
# tree so they are not mistaken for outputs.
WORK_DIR="${BIDOZE_WORK_DIR:-${TMPDIR:-/tmp}/bidoze-folds}"
# config/training-balanced.json is the release recipe: the class-balanced fit,
# early-stopped on a held-out share of each fold's training subjects.
# config/training.json is the curriculum recipe (staged focal loss, deblip,
# SWA). For a quick wiring check, override BIDOZE_TRAINING_CONFIG with a
# smaller custom config.
TRAINING_CONFIG="${BIDOZE_TRAINING_CONFIG:-config/training-balanced.json}"
# The cv config follows the recipe unless BIDOZE_CV_CONFIG names one: the
# balanced recipe has no deblip phase, so it reads config/cv-balanced.json,
# which lists no deblip sets.
RECIPE=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('recipe','curriculum'))" "$TRAINING_CONFIG")
if [[ "$RECIPE" == "balanced" ]]; then
    CV_CONFIG="${BIDOZE_CV_CONFIG:-config/cv-balanced.json}"
else
    CV_CONFIG="${BIDOZE_CV_CONFIG:-config/cv.json}"
fi
# Where a finished run is shipped. Empty disables the upload, which is the
# default on a workstation; ECS sets it.
RESULTS_S3_PREFIX="${BIDOZE_RESULTS_S3_PREFIX:-}"
LINK_ROOT="$WORK_DIR/cache"
CV_DIR="$WORK_DIR/cv"

usage() {
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        *) usage ;;
    esac
    shift
done

BASE_ARGS=(
    --cache-only
    -pc config/inputs.json
    -mc config/model.json
    -tc "$TRAINING_CONFIG"
    -mtc config/metrics.json
    --split-seed 0
)

# One model per generated config: fold0..fold{k-1}, then final.
PHASES=()
N_SPLITS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['n_splits'])" "$CV_CONFIG")
for ((i = 0; i < N_SPLITS; i++)); do PHASES+=("fold$i"); done
PHASES+=("final")

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'Run directory: %s\n' "$RESULTS_ROOT/$RUN_ID"
    printf 'Training cfg:  %s (%s)\n' "$TRAINING_CONFIG" "$RECIPE"
    printf 'CV cfg:        %s\n' "$CV_CONFIG"
    printf 'Split plan:    %s\n' "$SPLIT_PLAN"
    printf 'Upload to:     %s\n' "${RESULTS_S3_PREFIX:-<disabled>}"
    printf 'Linked cache:  %s\n' "$LINK_ROOT"
    for phase in "${PHASES[@]}"; do
        printf '  '; printf ' %q' arcasleep bidoze train "${BASE_ARGS[@]}" \
            -cvc "$CV_DIR/cv_$phase.json" -extn "$RUN_ID/training/$phase"; printf '\n'
    done
    exit 0
fi

echo "==> materialising $SPLIT_PLAN"
arcasleep splits-apply \
    -pc config/inputs.json \
    -cvc "$CV_CONFIG" \
    --plan "$SPLIT_PLAN" \
    --link-dir "$LINK_ROOT/$(python -c "import json;print(json.load(open('config/inputs.json'))['cache_prefix'])")" \
    --cv-out "$CV_DIR"

# Training reads the linked cache, never the real one: the fold datasets exist
# only as symlink names.
export FEATURE_CACHE_DIR="$LINK_ROOT"

# On EXIT, not after the loop: a crashed run is exactly when the log and any
# partial checkpoints are worth having, and `set -e` would otherwise skip the
# upload entirely. The original exit status is preserved for ECS to report.
upload_results() {
    local status=$?
    if [[ -n "$RESULTS_S3_PREFIX" ]]; then
        echo
        echo "==> upload $RESULTS_ROOT/$RUN_ID -> $RESULTS_S3_PREFIX (exit $status)"
        arcasleep upload-results \
            --run-dir "$RESULTS_ROOT/$RUN_ID" \
            --destination "$RESULTS_S3_PREFIX" \
            --run-id "$RUN_ID" \
            ${BIDOZE_UPLOAD_OVERWRITE:+--overwrite} || echo "upload failed (exit $?)"
    fi
    return $status
}
trap upload_results EXIT

for phase in "${PHASES[@]}"; do
    BASE_DIR="$RESULTS_ROOT/$RUN_ID/training/$phase"
    mkdir -p "$BASE_DIR"
    echo
    echo "==> $phase"
    PYTHONUNBUFFERED=1 arcasleep bidoze train "${BASE_ARGS[@]}" \
        -cvc "$CV_DIR/cv_$phase.json" \
        -extn "$RUN_ID/training/$phase" 2>&1 | tee "$BASE_DIR/training.log"
done

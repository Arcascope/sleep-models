#!/usr/bin/env bash
# UNet recipe: k-fold CV over the cohort, then a final model on all of it.
#
#   scripts/harness.sh [--dry-run] [phase ...]
#
# Phases are fold0..fold{k-1} and final, all of them by default, in the order
# given. The folds are not computed here: config/splits/*.json records the
# literal subject ids per fold, and this script only materialises them as
# linked datasets and trains one model per generated cv config.
#
# FEATURE_CACHE_DIR is the feature-cache root (holding the cache_prefix from
# config/inputs.json), shared with every other model family. With
# UNET_RESULTS_S3_PREFIX set, as it is on ECS, each phase is uploaded there on
# exit.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

RUN_ID="${UNET_RUN_ID:-unet-training}"
RESULTS_ROOT="${UNET_RESULTS_DIR:-$REPO_ROOT/results}"
SPLIT_PLAN="${UNET_SPLIT_PLAN:-config/splits/weaver_5fold.json}"
# Linked fold datasets and their cv configs are derived artifacts: they are
# rebuilt from the committed plan on every run, and live outside the results
# tree so they are not mistaken for outputs.
WORK_DIR="${UNET_WORK_DIR:-${TMPDIR:-/tmp}/unet-folds}"
TRAINING_CONFIG="${UNET_TRAINING_CONFIG:-config/training.json}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_DIR:-$REPO_ROOT/feature_cache}"
# Empty disables the upload, which is the default on a workstation.
RESULTS_S3_PREFIX="${UNET_RESULTS_S3_PREFIX:-}"
LINK_ROOT="$WORK_DIR/cache"
CV_DIR="$WORK_DIR/cv"

usage() {
    echo "Usage: $0 [--dry-run] [phase ...]" >&2
    exit 2
}

DRY_RUN=0
PHASES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -*) usage ;;
        *) PHASES+=("$1") ;;
    esac
    shift
done

if [[ ${#PHASES[@]} -eq 0 ]]; then
    N_SPLITS=$(python -c "import json;print(json.load(open('config/cv.json'))['n_splits'])")
    for ((i = 0; i < N_SPLITS; i++)); do PHASES+=("fold$i"); done
    PHASES+=("final")
fi

BASE_ARGS=(
    --cache-only
    -pc config/inputs.json
    -mc config/model.json
    -tc "$TRAINING_CONFIG"
    -mtc config/metrics.json
    --split-seed 0
)

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'Run directory: %s\n' "$RESULTS_ROOT/$RUN_ID"
    printf 'Training cfg:  %s\n' "$TRAINING_CONFIG"
    printf 'Split plan:    %s\n' "$SPLIT_PLAN"
    printf 'Feature cache: %s\n' "$FEATURE_CACHE_ROOT"
    printf 'Linked cache:  %s\n' "$LINK_ROOT"
    printf 'Upload to:     %s\n' "${RESULTS_S3_PREFIX:-<disabled>}"
    for phase in "${PHASES[@]}"; do
        printf '  '; printf ' %q' arcasleep unet train "${BASE_ARGS[@]}" \
            -cvc "$CV_DIR/cv_$phase.json" -extn "$RESULTS_ROOT/$RUN_ID/training/$phase"
        printf '\n'
    done
    exit 0
fi

echo "==> materialising $SPLIT_PLAN"
CACHE_PREFIX=$(python -c "import json;print(json.load(open('config/inputs.json'))['cache_prefix'])")
FEATURE_CACHE_DIR="$FEATURE_CACHE_ROOT" arcasleep splits-apply \
    -pc config/inputs.json \
    -cvc config/cv.json \
    --plan "$SPLIT_PLAN" \
    --link-dir "$LINK_ROOT/$CACHE_PREFIX" \
    --cv-out "$CV_DIR"

# Training reads the linked cache, never the real one: the fold datasets exist
# only as symlink names.
export FEATURE_CACHE_DIR="$LINK_ROOT"

# On EXIT, not after the loop: a crashed run is exactly when the log and any
# partial checkpoints are worth having, and `set -e` would otherwise skip the
# upload entirely. The original exit status is preserved for ECS to report.
# Each phase is its own upload with its own manifest, so phases of one run
# can train as separate tasks without one refusing another's destination.
upload_results() {
    local status=$? phase
    if [[ -n "$RESULTS_S3_PREFIX" ]]; then
        for phase in "${PHASES[@]}"; do
            [[ -d "$RESULTS_ROOT/$RUN_ID/training/$phase" ]] || continue
            echo
            echo "==> upload $phase -> $RESULTS_S3_PREFIX/$RUN_ID/training/$phase (exit $status)"
            arcasleep upload-results \
                --run-dir "$RESULTS_ROOT/$RUN_ID/training/$phase" \
                --destination "$RESULTS_S3_PREFIX" \
                --run-id "$RUN_ID/training/$phase" \
                ${UNET_UPLOAD_OVERWRITE:+--overwrite} || echo "upload failed (exit $?)"
        done
    fi
    return $status
}
trap upload_results EXIT

for phase in "${PHASES[@]}"; do
    PHASE_DIR="$RESULTS_ROOT/$RUN_ID/training/$phase"
    mkdir -p "$PHASE_DIR"
    echo
    echo "==> $phase"
    # An absolute -extn is used as given, so the results land in RESULTS_ROOT.
    PYTHONUNBUFFERED=1 arcasleep unet train "${BASE_ARGS[@]}" \
        -cvc "$CV_DIR/cv_$phase.json" \
        -extn "$PHASE_DIR" 2>&1 | tee "$PHASE_DIR/training.log"
done

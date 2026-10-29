#!/usr/bin/env bash
# WatchTree recipe: feature tables per cohort, then the CV experiments, then the watch export.
#
#   scripts/harness.sh [--dry-run] [phase ...]
#
# Phases are features, cv, export and comparison, run in the order given. The default is
# features, cv and export. Each run writes to results/<run id>/:
#   features/   one parquet per cohort, plus the uncropped Wearanize+ table
#   cv/         per-night and summary tables of every CV experiment (arcasleep_watchtree.cv)
#   export/     the final model and trees.bin for the watch (arcasleep_watchtree.watch.export)
#   cv/bidoze_epochs.parquet
#               per-epoch calls of the out-of-fold adult BiDoze models on Wearanize+, which
#               arcasleep_watchtree.report compares with WatchTree (arcasleep_watchtree.comparison_calls)
#   config/     copies of the model and training configs the run used
#
#   WATCHTREE_RUN_ID           run directory name (default watchtree-training)
#   WATCHTREE_RESULTS_DIR      parent of the run directory (default results)
#   WATCHTREE_MODEL_CONFIG     LightGBM parameters (default config/model.json)
#   WATCHTREE_TRAINING_CONFIG  rounds and early stopping (default config/training.json)
#   WATCHTREE_FEATURES_DIR     an existing features/ directory to use instead of building one;
#                              the features phase is then skipped
#   WATCHTREE_CV_STAGES        stages of the cv phase (default cv,loco,leak)
#   WATCHTREE_WORKERS          processes for the features phase (default 6)
#   WATCHTREE_BIDOZE_TRAINING  training/ of the BiDoze run the comparison phase scores with
#                              (default ../bidoze/results/adult-f2-2026-10-23/training)
#   WATCHTREE_FEATURE_CACHE    the BiDoze feature cache root (default ../feature_cache)
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

RUN_ID="${WATCHTREE_RUN_ID:-watchtree-training}"
RESULTS_ROOT="${WATCHTREE_RESULTS_DIR:-$REPO_ROOT/results}"
RUN_DIR="$RESULTS_ROOT/$RUN_ID"
MODEL_CONFIG="${WATCHTREE_MODEL_CONFIG:-config/model.json}"
TRAINING_CONFIG="${WATCHTREE_TRAINING_CONFIG:-config/training.json}"
FEATURES_DIR="${WATCHTREE_FEATURES_DIR:-$RUN_DIR/features}"
CV_STAGES="${WATCHTREE_CV_STAGES:-cv,loco,leak}"
WORKERS="${WATCHTREE_WORKERS:-6}"
BIDOZE_TRAINING="${WATCHTREE_BIDOZE_TRAINING:-$REPO_ROOT/../bidoze/results/adult-f2-2026-10-23/training}"
FEATURE_CACHE="${WATCHTREE_FEATURE_CACHE:-$REPO_ROOT/../feature_cache}"
COHORTS=(wearanize aauwss sleepaccel weaver_garmin weaver_fitbit)

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
        features | cv | export | comparison) PHASES+=("$1") ;;
        *) usage ;;
    esac
    shift
done
if [[ ${#PHASES[@]} -eq 0 ]]; then
    PHASES=(cv export)
    [[ -n "${WATCHTREE_FEATURES_DIR:-}" ]] || PHASES=(features "${PHASES[@]}")
fi

CONFIG_ARGS=(--model-config "$MODEL_CONFIG" --training-config "$TRAINING_CONFIG")

# In a dry run each command is printed, not run.
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  '; printf ' %q' "$@"; printf '\n'
    else
        PYTHONUNBUFFERED=1 "$@"
    fi
}

run_phase() {
    case "$1" in
        features)
            for cohort in "${COHORTS[@]}"; do
                run python -m arcasleep_watchtree.data.build --cohort "$cohort" --out "$FEATURES_DIR" \
                    --workers "$WORKERS"
            done
            run python -m arcasleep_watchtree.data.build --cohort wearanize --uncropped --out "$FEATURES_DIR" \
                --workers "$WORKERS"
            ;;
        cv)
            run python -m arcasleep_watchtree.cv --features "$FEATURES_DIR" --uncropped "$FEATURES_DIR" \
                --out "$RUN_DIR/cv" --stages "$CV_STAGES" "${CONFIG_ARGS[@]}"
            ;;
        export)
            run python -m arcasleep_watchtree.watch.export --features "$FEATURES_DIR" --uncropped "$FEATURES_DIR" \
                --out "$RUN_DIR/export" "${CONFIG_ARGS[@]}"
            ;;
        comparison)
            run python -m arcasleep_watchtree.comparison_calls --training "$BIDOZE_TRAINING" \
                --cache-root "$FEATURE_CACHE" --out "$RUN_DIR/cv/bidoze_epochs.parquet"
            ;;
    esac
}

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'Run directory: %s\n' "$RUN_DIR"
    printf 'Model cfg:     %s\n' "$MODEL_CONFIG"
    printf 'Training cfg:  %s\n' "$TRAINING_CONFIG"
    printf 'Features:      %s\n' "$FEATURES_DIR"
    for phase in "${PHASES[@]}"; do run_phase "$phase"; done
    exit 0
fi

mkdir -p "$RUN_DIR/config" "$RUN_DIR/logs"
cp "$MODEL_CONFIG" "$RUN_DIR/config/model.json"
cp "$TRAINING_CONFIG" "$RUN_DIR/config/training.json"

for phase in "${PHASES[@]}"; do
    echo
    echo "==> $phase"
    run_phase "$phase" 2>&1 | tee "$RUN_DIR/logs/$phase.log"
done

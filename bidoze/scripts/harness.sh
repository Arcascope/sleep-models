#!/usr/bin/env bash
# BiDoze recipe: k-fold CV over the cohort, then a final model on all of it.
#
#   scripts/harness.sh [--dry-run] [phase ...]
#
# Phases are fold0..fold{k-1} and final, all of them by default, in the order
# given.
#
# With BIDOZE_RELEASE=adult or pediatric, the phases retrain that release with the gap rule
# (src/arcasleep_bidoze/release.py): fold0..fold4 and final by default, and the final phase
# also writes the release bundle to <final phase dir>/bundle.
#   BIDOZE_RELEASE_VERSION     the bundle's version (default v0.2.0)
#   BIDOZE_RELEASE_MODEL       g: BiDoze-G, config/model-gap-rule.json; f2: the strand-fusion
#                              network, config/model-f2.json. Default: f2 for adult, g for pediatric.
#                              The adult phases train on Weaver, SleepAccel and Wearanize+.
#   BIDOZE_WEAVER_SET          the Weaver recordings trained on: all (default) or matched.
#                              adult: matched is the 202 revised recordings of the earlier Weaver
#                              nights; pediatric: matched is the earlier weaver cache itself
#                              (config/splits/weaver_5fold.json). all is every revised-Weaver
#                              recording (the pediatric plan is
#                              config/splits/weaver_revised_pediatric_5fold.json).
#   BIDOZE_RELEASE_TRAINING    a training config in place of the release's own, e.g.
#                              config/training-smoke.json for a wiring check
# BIDOZE_TRAINING_CONFIG, BIDOZE_CV_CONFIG, BIDOZE_SPLIT_PLAN and BIDOZE_MODEL_CONFIG are
# ignored for a release.
#
# With BIDOZE_EXPERIMENT=experiments/<name>, each phase is instead one run of
# that experiment's task.sh, which receives the phase name and the phase's
# results directory and does its own data selection and training. Phases are
# then required and named by the experiment (adult_devices: <arm>/fold<k>). The folds are not computed here. config/splits/*.json records the literal
# subject ids per fold; this script only materialises them as linked datasets
# and trains one model per generated cv config.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ -n "${BIDOZE_ADULT_WEAVER_SET:-}" ]]; then
    echo "BIDOZE_ADULT_WEAVER_SET is replaced by BIDOZE_WEAVER_SET" >&2
    exit 2
fi

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
# config/model.json is the release architecture; BIDOZE_MODEL_CONFIG=config/model-gap-rule.json
# adds the gap rule (gap called only from the gap row, absent epochs masked).
MODEL_CONFIG="${BIDOZE_MODEL_CONFIG:-config/model.json}"
# The cv config follows the recipe unless BIDOZE_CV_CONFIG names one: the
# balanced recipe has no deblip phase, so it reads config/cv-balanced.json,
# which lists no deblip sets.
if [[ -n "${BIDOZE_RELEASE:-}" ]]; then
    RECIPE=balanced
else
    RECIPE=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('recipe','curriculum'))" "$TRAINING_CONFIG")
fi
if [[ "$RECIPE" == "balanced" ]]; then
    CV_CONFIG="${BIDOZE_CV_CONFIG:-config/cv-balanced.json}"
else
    CV_CONFIG="${BIDOZE_CV_CONFIG:-config/cv.json}"
fi
# Where a finished run is shipped. Empty disables the upload, which is the
# default on a workstation; ECS sets it.
RESULTS_S3_PREFIX="${BIDOZE_RESULTS_S3_PREFIX:-}"
EXPERIMENT="${BIDOZE_EXPERIMENT:-}"
RELEASE="${BIDOZE_RELEASE:-}"
RELEASE_VERSION="${BIDOZE_RELEASE_VERSION:-v0.2.0}"
if [[ -n "$RELEASE" ]]; then
    [[ -z "$EXPERIMENT" ]] || { echo "BIDOZE_RELEASE and BIDOZE_EXPERIMENT are exclusive" >&2; exit 2; }
    case "$RELEASE" in
        adult)     TRAINING_CONFIG=config/training-adult.json ;;
        pediatric) TRAINING_CONFIG=config/training-pediatric.json ;;
        *) echo "BIDOZE_RELEASE must be adult or pediatric, got $RELEASE" >&2; exit 2 ;;
    esac
    TRAINING_CONFIG="${BIDOZE_RELEASE_TRAINING:-$TRAINING_CONFIG}"
    RELEASE_MODEL="${BIDOZE_RELEASE_MODEL:-$([[ "$RELEASE" == adult ]] && echo f2 || echo g)}"
    case "$RELEASE_MODEL" in
        g)  MODEL_CONFIG=config/model-gap-rule.json ;;
        f2) MODEL_CONFIG=config/model-f2.json ;;
        *) echo "BIDOZE_RELEASE_MODEL must be g or f2, got $RELEASE_MODEL" >&2; exit 2 ;;
    esac
    RECIPE=balanced
    WEAVER_SET="${BIDOZE_WEAVER_SET:-all}"
    if [[ "$RELEASE" == pediatric ]]; then
        read -r SPLIT_PLAN CV_CONFIG < <(python -m arcasleep_bidoze.release pediatric-recipe --weaver-set "$WEAVER_SET")
    fi
fi
LINK_ROOT="$WORK_DIR/cache"
CV_DIR="$WORK_DIR/cv"
# The adult release's phases are materialised by arcasleep_bidoze.release, not from a split plan.
RELEASE_SETUP="$WORK_DIR/release-$RELEASE"
REAL_CACHE_DIR="${FEATURE_CACHE_DIR:-}"
if [[ "$RELEASE" == adult ]]; then
    CV_DIR="$RELEASE_SETUP/cv"
    TRAINING_CONFIG="${BIDOZE_RELEASE_TRAINING:-$RELEASE_SETUP/training.json}"
    LINK_ROOT="$RELEASE_SETUP/cache"
fi

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

BASE_ARGS=(
    --cache-only
    -pc config/inputs.json
    -mc "$MODEL_CONFIG"
    -tc "$TRAINING_CONFIG"
    -mtc config/metrics.json
    --split-seed 0
)

if [[ -n "$EXPERIMENT" ]]; then
    [[ -f "$EXPERIMENT/task.sh" ]] || { echo "no task.sh in $EXPERIMENT" >&2; exit 2; }
    [[ ${#PHASES[@]} -gt 0 ]] || { echo "$EXPERIMENT needs its phases named" >&2; exit 2; }
fi

# One model per generated config: fold0..fold{k-1}, then final.
if [[ ${#PHASES[@]} -eq 0 ]]; then
    N_SPLITS=$([[ "$RELEASE" == adult ]] && echo 5 || python -c "import json,sys;print(json.load(open(sys.argv[1]))['n_splits'])" "$CV_CONFIG")
    for ((i = 0; i < N_SPLITS; i++)); do PHASES+=("fold$i"); done
    PHASES+=("final")
fi

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'Run directory: %s\n' "$RESULTS_ROOT/$RUN_ID"
    if [[ -n "$EXPERIMENT" ]]; then
        printf 'Experiment:    %s\n' "$EXPERIMENT"
        printf 'Upload to:     %s\n' "${RESULTS_S3_PREFIX:-<disabled>}"
        for phase in "${PHASES[@]}"; do
            printf '   bash %q %q %q\n' "$EXPERIMENT/task.sh" "$phase" "$RESULTS_ROOT/$RUN_ID/training/$phase"
        done
        exit 0
    fi
    printf 'Model cfg:     %s\n' "$MODEL_CONFIG"
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

if [[ "$RELEASE" == adult ]]; then
    echo "==> materialising the adult release phases"
    python -m arcasleep_bidoze.release prepare-adult "$RELEASE_SETUP" "$REAL_CACHE_DIR" \
        --weaver-set "$WEAVER_SET"
    export FEATURE_CACHE_DIR="$LINK_ROOT"
elif [[ -z "$EXPERIMENT" ]]; then
    echo "==> materialising $SPLIT_PLAN"
    arcasleep splits-apply \
        -pc config/inputs.json \
        -cvc "$CV_CONFIG" \
        --plan "$SPLIT_PLAN" \
        --link-dir "$LINK_ROOT/$(python -c "import json;print(json.load(open('config/inputs.json'))['cache_prefix'])")" \
        --cv-out "$CV_DIR"

    # Training reads the linked cache, never the real one: the fold datasets
    # exist only as symlink names.
    export FEATURE_CACHE_DIR="$LINK_ROOT"
fi

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
                ${BIDOZE_UPLOAD_OVERWRITE:+--overwrite} || echo "upload failed (exit $?)"
        done
    fi
    return $status
}
trap upload_results EXIT

for phase in "${PHASES[@]}"; do
    BASE_DIR="$RESULTS_ROOT/$RUN_ID/training/$phase"
    mkdir -p "$BASE_DIR"
    echo
    echo "==> $phase"
    if [[ -n "$EXPERIMENT" ]]; then
        PYTHONUNBUFFERED=1 bash "$EXPERIMENT/task.sh" "$phase" "$BASE_DIR" 2>&1 | tee "$BASE_DIR/training.log"
    else
        PYTHONUNBUFFERED=1 arcasleep bidoze train "${BASE_ARGS[@]}" \
            -cvc "$CV_DIR/cv_$phase.json" \
            -extn "$RUN_ID/training/$phase" 2>&1 | tee "$BASE_DIR/training.log"
    fi
    if [[ -n "$RELEASE" && "$phase" == final ]]; then
        echo "==> release bundle $RELEASE $RELEASE_VERSION"
        python -m arcasleep_bidoze.release bundle "$RELEASE" "$BASE_DIR" "$BASE_DIR/bundle" "$RELEASE_VERSION" \
            ${REAL_CACHE_DIR:+--cache-root "$REAL_CACHE_DIR"} \
            $([[ "$RELEASE" == adult ]] && echo "--train-cache $RELEASE_SETUP/cache")
    fi
done

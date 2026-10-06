#!/usr/bin/env bash
# Build a feature cache on a GPU task: sync datasets from S3, process them,
# sync the cache back.
#
#   prep_harness.sh <inputs-config> <dataset> [dataset ...]
#
# <inputs-config> is a processing config inside the image, such as
# unet/config/inputs.json; its "cache_prefix" names the cache. Each <dataset>
# is a directory under PREP_DATA_S3 (data_set.json or pisces_lite_adapter.py
# at its root).
#
# The cache prefix already in S3 is synced down first, so subjects cached
# there are skipped and only new ones are processed. Nothing in S3 is
# deleted. The image's entrypoint; run_prep_task.sh launches it.
set -euo pipefail

[[ $# -ge 2 ]] || { echo "usage: $0 <inputs-config> <dataset> [dataset ...]" >&2; exit 2; }
INPUTS=$1
shift
DATASETS=("$@")

: "${PREP_DATA_S3:?set to the s3:// prefix holding the datasets}"
: "${PREP_CACHE_S3:?set to the s3:// prefix holding feature caches}"
WORK_DIR="${PREP_WORK_DIR:-${TMPDIR:-/tmp}/prep}"
CACHE_PREFIX=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['cache_prefix'])" "$INPUTS")
CACHE_URI="${PREP_CACHE_S3%/}/$CACHE_PREFIX"

echo "==> cache   $CACHE_PREFIX ($INPUTS)"
echo "    data    ${DATASETS[*]}"
echo "    to      $CACHE_URI/"

mkdir -p "$WORK_DIR/data" "$WORK_DIR/cache/$CACHE_PREFIX"
aws s3 sync --only-show-errors "$CACHE_URI/" "$WORK_DIR/cache/$CACHE_PREFIX/"
echo "    $(find "$WORK_DIR/cache/$CACHE_PREFIX" -name '*_features.npz' | wc -l) subjects already cached"

DATA_ARGS=()
for dataset in "${DATASETS[@]}"; do
    echo "==> sync $dataset"
    aws s3 sync --only-show-errors "${PREP_DATA_S3%/}/$dataset/" "$WORK_DIR/data/$dataset/"
    DATA_ARGS+=(-dd "$WORK_DIR/data/$dataset")
done

# On EXIT, so subjects finished before a crash or timeout are kept.
upload_cache() {
    local status=$?
    echo
    echo "==> upload $CACHE_URI/ (exit $status)"
    aws s3 sync --only-show-errors --exclude '*' --include '*_features.npz' \
        "$WORK_DIR/cache/$CACHE_PREFIX/" "$CACHE_URI/" || echo "upload failed (exit $?)"
    echo "    $(find "$WORK_DIR/cache/$CACHE_PREFIX" -name '*_features.npz' | wc -l) subjects in the cache"
    return $status
}
trap upload_cache EXIT

arcasleep process -pc "$INPUTS" -o "$WORK_DIR/cache" "${DATA_ARGS[@]}" ${PREP_PROCESS_ARGS:-}

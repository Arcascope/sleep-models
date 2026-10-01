#!/usr/bin/env bash
# Build the content-addressed data image from a feature cache directory.
#
#   scripts/build_data_image.sh feature_cache/12Hz_log_PSD
#   scripts/build_data_image.sh /path/to/12Hz_log_PSD --push <ecr-repo-uri>
#
# The argument is the cache *prefix* directory -- the one holding the NPZs --
# and its basename becomes the cache prefix, so it must match "cache_prefix" in
# config/inputs.json. The image is tagged by the manifest digest, not by hand:
# the same bytes always produce the same tag.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DOCKERFILE="$REPO_ROOT/docker/data.Dockerfile"
IMAGE_NAME="${BIDOZE_DATA_IMAGE_NAME:-bidoze-data}"

usage() {
    echo "Usage: $0 <cache-prefix-dir> [--push <repository-uri>]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
CACHE_DIR=$1
shift

PUSH_REPO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            [[ $# -ge 2 ]] || usage
            PUSH_REPO=$2
            shift 2
            ;;
        *) usage ;;
    esac
done

[[ -d "$CACHE_DIR" ]] || { echo "not a directory: $CACHE_DIR" >&2; exit 1; }
CACHE_DIR=$(cd -- "$CACHE_DIR" && pwd)
CACHE_PREFIX=$(basename -- "$CACHE_DIR")

# The manifest is a build product, so it lands in build/ rather than beside the
# data: build_local_manifest hashes every regular file under the cache root, and
# a manifest stored there would change the digest it is recording.
MANIFEST_DIR="$REPO_ROOT/build/manifest/$CACHE_PREFIX"
mkdir -p "$MANIFEST_DIR"

echo "==> manifest $CACHE_DIR"
MANIFEST_OUTPUT=$(bidoze manifest-local --root "$CACHE_DIR" --output "$MANIFEST_DIR/manifest.jsonl")
echo "$MANIFEST_OUTPUT"
MANIFEST_SHA256=${MANIFEST_OUTPUT##*manifest_sha256=}
[[ ${#MANIFEST_SHA256} -eq 64 ]] || { echo "could not read manifest digest" >&2; exit 1; }
SHORT_SHA=${MANIFEST_SHA256:0:12}


TAG="$IMAGE_NAME:$CACHE_PREFIX-$SHORT_SHA"
echo "==> build $TAG"
docker buildx build \
    --platform "${TARGET_PLATFORM:-linux/amd64}" \
    -f "$DOCKERFILE" \
    --build-context "manifest=$MANIFEST_DIR" \
    --build-arg "CACHE_PREFIX=$CACHE_PREFIX" \
    -t "$TAG" \
    --load \
    "$CACHE_DIR"

echo
echo "image      $TAG"
echo "manifest   $MANIFEST_SHA256"
echo "manifest   $MANIFEST_DIR/manifest.jsonl"

if [[ -n "$PUSH_REPO" ]]; then
    REMOTE="$PUSH_REPO:$CACHE_PREFIX-$SHORT_SHA"
    echo
    echo "==> push $REMOTE"
    docker tag "$TAG" "$REMOTE"
    # With ECR tag immutability on, a re-push of unchanged bytes fails here.
    # That is the intended signal: the data is already published, pin the
    # existing digest rather than making a second image of the same cache.
    docker push "$REMOTE"
    echo
    echo "pass this digest to scripts/build_train_image.sh:"
    docker inspect --format '{{index .RepoDigests 0}}' "$REMOTE"
fi

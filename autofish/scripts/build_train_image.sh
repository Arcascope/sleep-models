#!/usr/bin/env bash
# Build the training image on top of a data image.
#
#   scripts/build_train_image.sh <data-image-ref> [--push <repository-uri>]
#
# pisces-lite and arcascope-senpy are installed from PyPI, so no private
# checkout is needed. Override the pin with PISCES_LITE_VERSION when validating
# a new release.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DOCKERFILE="$REPO_ROOT/docker/train.Dockerfile"
IMAGE_NAME="${AUTOFISH_TRAIN_IMAGE_NAME:-autofish-train}"
PISCES_LITE_VERSION="${PISCES_LITE_VERSION:-4.0.0}"

usage() {
    echo "Usage: $0 <data-image-ref> [--push <repository-uri>]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
DATA_IMAGE=$1
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

# The tag records both halves: the data the model saw and the code that ran.
DATA_TAG=${DATA_IMAGE##*:}
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
# A tag naming a clean commit is a claim that the image holds that commit's
# tree. Mark it when it does not, so the run fingerprint cannot silently
# attribute a result to code that was never committed.
if [[ -n $(git -C "$REPO_ROOT" status --porcelain) ]]; then
    GIT_SHA="$GIT_SHA-dirty"
fi
TAG="$IMAGE_NAME:$DATA_TAG-$GIT_SHA"

echo "==> build $TAG"
echo "    data   $DATA_IMAGE"
echo "    pisces-lite $PISCES_LITE_VERSION (PyPI)"
docker buildx build \
    --platform "${TARGET_PLATFORM:-linux/amd64}" \
    -f "$DOCKERFILE" \
    --build-context "data=docker-image://$DATA_IMAGE" \
    --build-arg "GIT_COMMIT=$GIT_SHA" \
    --build-arg "PISCES_LITE_VERSION=$PISCES_LITE_VERSION" \
    -t "$TAG" \
    --load \
    "$REPO_ROOT"

echo
echo "image $TAG"

if [[ -n "$PUSH_REPO" ]]; then
    REMOTE="$PUSH_REPO:$DATA_TAG-$GIT_SHA"
    echo "==> push $REMOTE"
    docker tag "$TAG" "$REMOTE"
    docker push "$REMOTE"
    echo
    echo "pin this digest in the ECS task definition:"
    docker inspect --format '{{index .RepoDigests 0}}' "$REMOTE"
fi

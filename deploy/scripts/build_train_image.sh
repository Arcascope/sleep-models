#!/usr/bin/env bash
# Build a model family's training image on top of a data image.
#
#   deploy/scripts/build_train_image.sh <family> <data-image-ref> [--deps <deps-image-ref>] [--push <repository-uri>]
#
# The Dockerfile is <family>/docker/train.Dockerfile, built with the repository
# root as context, FROM the dependency image (build_deps_image.sh). --deps
# defaults to the local arcasleep-deps image with the current deps_tag.sh tag.
#
# The commit comes from git, or from GIT_COMMIT where there is no .git (the
# remote build's source archive).
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEPS_IMAGE="arcasleep-deps:$(bash "$REPO_ROOT/deploy/scripts/deps_tag.sh")"

usage() {
    echo "Usage: $0 <family> <data-image-ref> [--deps <deps-image-ref>] [--push <repository-uri>]" >&2
    exit 2
}

[[ $# -ge 2 ]] || usage
MODEL_FAMILY=$1
DATA_IMAGE=$2
shift 2
DOCKERFILE="$REPO_ROOT/$MODEL_FAMILY/docker/train.Dockerfile"
[[ -f "$DOCKERFILE" ]] || { echo "no training Dockerfile for $MODEL_FAMILY: $DOCKERFILE" >&2; exit 2; }
IMAGE_NAME="arcasleep-$MODEL_FAMILY"

PUSH_REPO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            [[ $# -ge 2 ]] || usage
            PUSH_REPO=$2
            shift 2
            ;;
        --deps)
            [[ $# -ge 2 ]] || usage
            DEPS_IMAGE=$2
            shift 2
            ;;
        *) usage ;;
    esac
done

# The tag records both halves: the data the model saw and the code that ran.
# A reference pinned as name:tag@sha256:... still names its tag before the @.
DATA_TAG=${DATA_IMAGE%@*}
DATA_TAG=${DATA_TAG##*:}
[[ "$DATA_TAG" != */* ]] || { echo "the data image reference needs a tag: $DATA_IMAGE" >&2; exit 2; }
if [[ -n ${GIT_COMMIT:-} ]]; then
    GIT_SHA=$GIT_COMMIT
else
    GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
    # A tag naming a clean commit is a claim that the image holds that
    # commit's tree. Mark it when it does not, so the run fingerprint cannot
    # silently attribute a result to code that was never committed.
    if [[ -n $(git -C "$REPO_ROOT" status --porcelain) ]]; then
        GIT_SHA="$GIT_SHA-dirty"
    fi
fi
TAG="$IMAGE_NAME:$DATA_TAG-$GIT_SHA"

echo "==> build $TAG"
echo "    deps   $DEPS_IMAGE"
echo "    data   $DATA_IMAGE"
docker buildx build \
    --platform "${TARGET_PLATFORM:-linux/amd64}" \
    -f "$DOCKERFILE" \
    --build-context "data=docker-image://$DATA_IMAGE" \
    --build-arg "GIT_COMMIT=$GIT_SHA" \
    --build-arg "DEPS_IMAGE=$DEPS_IMAGE" \
    -t "$TAG" \
    --load \
    "$REPO_ROOT"

echo
echo "image $TAG"

if [[ -n "$PUSH_REPO" ]]; then
    REMOTE="$PUSH_REPO:$DATA_TAG-$GIT_SHA"
    echo "==> push $REMOTE"
    docker tag "$TAG" "$REMOTE"
    # With ECR tag immutability on, re-pushing an existing tag fails here: that
    # commit and data are already published, so launch the existing image.
    docker push "$REMOTE"
    echo
    PINNED=$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$REMOTE" \
        | grep -m1 "^$PUSH_REPO@")
    echo "launch with deploy/scripts/run_task.sh $MODEL_FAMILY $REMOTE@${PINNED#*@}"
fi

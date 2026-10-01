#!/usr/bin/env bash
# Build the training image on top of a data image.
#
#   scripts/build_train_image.sh <data-image-ref> [--push <repository-uri>]
#
# pisces-lite and arcascope-senpy are installed from PyPI. The image installs
# exactly the minimum version pinned by the `pisces` extra in pyproject.toml;
# override it with PISCES_LITE_VERSION when validating a new release.
set -euo pipefail

MODEL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd -- "$MODEL_ROOT/.." && pwd)
DOCKERFILE="$MODEL_ROOT/docker/train.Dockerfile"
IMAGE_NAME="${BIDOZE_TRAIN_IMAGE_NAME:-bidoze-train}"
if [[ -z ${PISCES_LITE_VERSION:-} ]]; then
    PISCES_LITE_VERSION=$(python3 - "$REPO_ROOT/pyproject.toml" <<'EOF'
import re, sys, tomllib
with open(sys.argv[1], "rb") as f:
    (spec,) = tomllib.load(f)["project"]["optional-dependencies"]["pisces"]
print(re.fullmatch(r"pisces-lite>=([\w.]+)", spec).group(1))
EOF
)
fi

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

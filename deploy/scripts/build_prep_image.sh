#!/usr/bin/env bash
# Build the feature-cache build image (deploy/docker/prep.Dockerfile).
#
#   deploy/scripts/build_prep_image.sh [--deps <deps-image-ref>] [--push <repository-uri>]
#
# Tagged by the commit; --deps and GIT_COMMIT behave as in build_train_image.sh.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEPS_IMAGE="arcasleep-deps:$(bash "$REPO_ROOT/deploy/scripts/deps_tag.sh")"

usage() {
    echo "Usage: $0 [--deps <deps-image-ref>] [--push <repository-uri>]" >&2
    exit 2
}

PUSH_REPO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) [[ $# -ge 2 ]] || usage; PUSH_REPO=$2; shift 2 ;;
        --deps) [[ $# -ge 2 ]] || usage; DEPS_IMAGE=$2; shift 2 ;;
        *) usage ;;
    esac
done

if [[ -n ${GIT_COMMIT:-} ]]; then
    GIT_SHA=$GIT_COMMIT
else
    GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
    if [[ -n $(git -C "$REPO_ROOT" status --porcelain) ]]; then
        GIT_SHA="$GIT_SHA-dirty"
    fi
fi
TAG="arcasleep-prep:$GIT_SHA"

echo "==> build $TAG"
echo "    deps   $DEPS_IMAGE"
docker buildx build \
    --platform "${TARGET_PLATFORM:-linux/amd64}" \
    -f "$REPO_ROOT/deploy/docker/prep.Dockerfile" \
    --build-arg "GIT_COMMIT=$GIT_SHA" \
    --build-arg "DEPS_IMAGE=$DEPS_IMAGE" \
    -t "$TAG" \
    --load \
    "$REPO_ROOT"

if [[ -n "$PUSH_REPO" ]]; then
    echo "==> push $PUSH_REPO:$GIT_SHA"
    docker tag "$TAG" "$PUSH_REPO:$GIT_SHA"
    docker push "$PUSH_REPO:$GIT_SHA"
    PINNED=$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$PUSH_REPO:$GIT_SHA" \
        | grep -m1 "^$PUSH_REPO@")
    echo
    echo "launch with deploy/scripts/run_prep_task.sh $PUSH_REPO:$GIT_SHA@${PINNED#*@} <inputs-config> <dataset> ..."
fi

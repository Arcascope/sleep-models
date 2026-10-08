#!/usr/bin/env bash
# Build the dependency image every training and prep image starts FROM.
#
#   deploy/scripts/build_deps_image.sh [--push <deps-repository-uri>]
#
# Rebuild only when deps.Dockerfile, the jax pin (deps_tag.sh) or the
# pisces-lite pin (pyproject.toml) changes; the tag changes with them.
set -euo pipefail

SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPS_TAG=$(bash "$SCRIPTS/deps_tag.sh")
TAG="arcasleep-deps:$DEPS_TAG"

PUSH_REPO=""
if [[ $# -gt 0 ]]; then
    [[ $# -eq 2 && "$1" == --push ]] || { echo "Usage: $0 [--push <repository-uri>]" >&2; exit 2; }
    PUSH_REPO=$2
fi

echo "==> build $TAG"
# The Dockerfile copies nothing, so its own directory is the whole context.
docker buildx build \
    --platform "${TARGET_PLATFORM:-linux/amd64}" \
    -f "$SCRIPTS/../docker/deps.Dockerfile" \
    --build-arg "JAX_VERSION=$(bash "$SCRIPTS/deps_tag.sh" jax)" \
    --build-arg "PISCES_LITE_VERSION=$(bash "$SCRIPTS/deps_tag.sh" pisces)" \
    -t "$TAG" \
    --load \
    "$SCRIPTS/../docker"

if [[ -n "$PUSH_REPO" ]]; then
    echo "==> push $PUSH_REPO:$DEPS_TAG"
    docker tag "$TAG" "$PUSH_REPO:$DEPS_TAG"
    docker push "$PUSH_REPO:$DEPS_TAG"
fi

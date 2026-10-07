#!/usr/bin/env bash
# Print the dependency image's tag: the pinned versions and a digest of
# deps.Dockerfile, so the same inputs always name the same image.
#
#   deploy/scripts/deps_tag.sh      # e.g. jax0.11.2-pisces4.1.1-3f9c2a1b
#
# jax is pinned here, the one place: the jax[cuda12] extra brings the whole
# CUDA stack in as pinned wheels, so this version fixes the CUDA runtime too.
# pisces-lite is the minimum pinned by the `pisces` extra in pyproject.toml;
# PISCES_LITE_VERSION overrides it when validating a new release.
set -euo pipefail

JAX_VERSION=0.11.2
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -z ${PISCES_LITE_VERSION:-} ]]; then
    PISCES_LITE_VERSION=$(python3 - "$REPO_ROOT/pyproject.toml" <<'PY'
import re, sys, tomllib
with open(sys.argv[1], "rb") as f:
    (spec,) = tomllib.load(f)["project"]["optional-dependencies"]["pisces"]
print(re.fullmatch(r"pisces-lite>=([\w.]+)", spec).group(1))
PY
)
fi
DOCKERFILE_SHA=$(sha256sum "$REPO_ROOT/deploy/docker/deps.Dockerfile" | cut -c1-8)

case "${1:-tag}" in
    tag) echo "jax$JAX_VERSION-pisces$PISCES_LITE_VERSION-$DOCKERFILE_SHA" ;;
    jax) echo "$JAX_VERSION" ;;
    pisces) echo "$PISCES_LITE_VERSION" ;;
    *) echo "usage: $0 [tag|jax|pisces]" >&2; exit 2 ;;
esac

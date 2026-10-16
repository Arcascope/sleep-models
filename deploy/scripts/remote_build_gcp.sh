#!/usr/bin/env bash
# Build and push an image in Google Cloud Build instead of on this machine.
#
# The GCP counterpart of deploy/scripts/remote_build.sh.
#
#   deploy/scripts/remote_build_gcp.sh [--no-wait] KIND [ARGS]
#
#     deps                       the dependency image (jax, pisces-lite)
#     data CACHE_PREFIX          a data image from gs://.../feature_cache/CACHE_PREFIX/
#     train FAMILY DATA_IMAGE    FAMILY's training image on DATA_IMAGE
#
# The build uses HEAD as committed (git archive), so uncommitted changes are not
# in the image and every image names a commit. An image whose tag is already in
# Artifact Registry is not rebuilt. Waits for the build unless --no-wait.
#
# environment:
#   ARCASLEEP_PROJECT  project (default shiftapp-arcascope)
#   ARCASLEEP_REGION   region  (default us-central1)
#   ARCASLEEP_REPO     repository (default arcasleep-train)
#   ARCASLEEP_CACHE    feature-cache prefix
#                      (default gs://arcascope-sleep-models-data/feature_cache)
usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^#\( \|$\)//'
}
set -euo pipefail

PROJECT="${ARCASLEEP_PROJECT:-shiftapp-arcascope}"
REGION="${ARCASLEEP_REGION:-us-central1}"
REPO="${ARCASLEEP_REPO:-arcasleep-train}"
CACHE="${ARCASLEEP_CACHE:-gs://arcascope-sleep-models-data/feature_cache}"
PREFIX="$REGION-docker.pkg.dev/$PROJECT/$REPO"
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

WAIT=1
if [[ "${1:-}" == --no-wait ]]; then WAIT=0; shift; fi
[[ $# -ge 1 && "$1" != -h && "$1" != --help ]] || { usage; exit 2; }
KIND=$1
shift

SUBS=("_BUILD_KIND=$KIND" "_CACHE_GCS=$CACHE" "_DEPS_REPO=$PREFIX/deps"
      "_DATA_REPO=$PREFIX/data" "_UNET_REPO=$PREFIX/unet"
      "_BIDOZE_REPO=$PREFIX/bidoze")
case "$KIND" in
    deps) [[ $# -eq 0 ]] || { usage >&2; exit 2; } ;;
    data) [[ $# -eq 1 ]] || { usage >&2; exit 2; }
          SUBS+=("_CACHE_PREFIX=$1") ;;
    train) [[ $# -eq 2 ]] || { usage >&2; exit 2; }
           SUBS+=("_MODEL_FAMILY=$1" "_DATA_IMAGE=$2") ;;
    *) usage >&2; exit 2 ;;
esac

# deps_tag.sh reads this out of pyproject.toml; the build step's python is
# older than tomllib, so it is read here and travels as a substitution.
PISCES=$(bash "$REPO_ROOT/deploy/scripts/deps_tag.sh" pisces)
COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
if [[ -n $(git -C "$REPO_ROOT" status --porcelain --untracked-files=no) ]]; then
    echo "note: uncommitted changes are not in the build; it uses $COMMIT as committed" >&2
fi

# gcloud uploads a tarball of HEAD, so the build source matches the commit the
# image will name; a directory upload would include uncommitted work.
ARCHIVE=$(mktemp -t arcasleep-gcp.XXXXXX).tgz
trap 'rm -f "$ARCHIVE"' EXIT
git -C "$REPO_ROOT" archive --format=tar.gz -o "$ARCHIVE" HEAD

ARGS=(--project="$PROJECT" --region="$REGION"
      --config="$REPO_ROOT/deploy/gcp/cloudbuild.yaml"
      --substitutions="$(IFS=,; echo "${SUBS[*]}"),_GIT_COMMIT=$COMMIT,_PISCES_LITE_VERSION=$PISCES")
[[ $WAIT -eq 1 ]] || ARGS+=(--async)

echo "==> $KIND build (commit $COMMIT)"
echo "    logs: gcloud builds log --region=$REGION --stream <id>"
gcloud builds submit "${ARGS[@]}" "$ARCHIVE"

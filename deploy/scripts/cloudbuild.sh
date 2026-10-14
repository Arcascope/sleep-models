#!/usr/bin/env bash
# The build Cloud Build runs (deploy/gcp/cloudbuild.yaml). Not for workstations:
# start it with remote_build_gcp.sh.
#
# BUILD_KIND picks the image:
#   deps    the dependency image
#   data    a data image from the feature cache in $CACHE_GCS/$CACHE_PREFIX/
#   train   $MODEL_FAMILY's training image on $DATA_IMAGE
#
# The GCP counterpart of deploy/scripts/codebuild.sh: same tags, the same
# immutability rule, Artifact Registry in place of ECR. An image whose tag is
# already in Artifact Registry is not rebuilt -- the tag names its inputs, so
# the image already there is the one this build would have made.
set -euo pipefail

: "${BUILD_KIND:?}" "${GIT_COMMIT:?}" "${DEPS_REPO:?}" "${IMAGE_REF_FILE:?}"
SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export GIT_COMMIT
command -v gcloud >/dev/null
command -v docker >/dev/null

# The digest Artifact Registry holds under this tag, or nothing when the tag
# is not there yet.
ar_digest() {
    local uri=$1 tag=$2 digest error_file
    error_file=$(mktemp)
    if digest=$(gcloud artifacts docker images describe "$uri:$tag" \
        --format='value(image_summary.digest)' 2>"$error_file"); then
        rm -f "$error_file"
        [[ "$digest" == sha256:* ]] || { echo "no digest returned for $uri:$tag" >&2; return 1; }
        echo "$digest"
    elif grep -q 'NOT_FOUND' "$error_file"; then
        rm -f "$error_file"
    else
        cat "$error_file" >&2
        rm -f "$error_file"
        return 1
    fi
}

pinned() {
    local uri=$1 tag=$2 digest
    digest=$(ar_digest "$uri" "$tag")
    [[ -n "$digest" ]] || { echo "image missing after build: $uri:$tag" >&2; return 1; }
    echo "$uri:$tag@$digest"
}

DEPS_TAG=$(bash "$SCRIPTS/deps_tag.sh")
ensure_deps() {
    local digest
    digest=$(ar_digest "$DEPS_REPO" "$DEPS_TAG")
    if [[ -n "$digest" ]]; then
        echo "==> deps $DEPS_REPO:$DEPS_TAG already in Artifact Registry"
    else
        bash "$SCRIPTS/build_deps_image.sh" --push "$DEPS_REPO"
    fi
}

case "$BUILD_KIND" in
    deps)
        ensure_deps
        REF=$(pinned "$DEPS_REPO" "$DEPS_TAG")
        ;;
    data)
        : "${DATA_REPO:?}" "${CACHE_GCS:?}" "${CACHE_PREFIX:?}"
        CACHE_DIR="$PWD/cache/$CACHE_PREFIX"
        mkdir -p "$CACHE_DIR"
        echo "==> sync ${CACHE_GCS%/}/$CACHE_PREFIX/"
        gcloud storage cp --quiet \
            "${CACHE_GCS%/}/$CACHE_PREFIX/*_features.npz" "$CACHE_DIR/"
        echo "    $(find "$CACHE_DIR" -name '*_features.npz' | wc -l) subjects"
        # The tag is the manifest digest, known only after hashing the cache.
        OUT=$(bash "$SCRIPTS/build_data_image.sh" "$CACHE_DIR" | tee /dev/stderr)
        LOCAL_TAG=$(awk '$1 == "image" { print $2 }' <<<"$OUT")
        TAG=${LOCAL_TAG##*:}
        DIGEST=$(ar_digest "$DATA_REPO" "$TAG")
        if [[ -n "$DIGEST" ]]; then
            echo "==> $DATA_REPO:$TAG already in Artifact Registry"
        else
            docker tag "$LOCAL_TAG" "$DATA_REPO:$TAG"
            docker push "$DATA_REPO:$TAG"
        fi
        REF=$(pinned "$DATA_REPO" "$TAG")
        ;;
    train)
        : "${MODEL_FAMILY:?}" "${DATA_IMAGE:?}"
        REPO_VAR="$(tr '[:lower:]' '[:upper:]' <<<"$MODEL_FAMILY")_REPO"
        TRAIN_REPO=${!REPO_VAR:?no repository for $MODEL_FAMILY}
        DATA_TAG=${DATA_IMAGE%@*}
        TAG="${DATA_TAG##*:}-$GIT_COMMIT"
        DIGEST=$(ar_digest "$TRAIN_REPO" "$TAG")
        if [[ -n "$DIGEST" ]]; then
            echo "==> $TRAIN_REPO:$TAG already in Artifact Registry"
        else
            ensure_deps
            bash "$SCRIPTS/build_train_image.sh" "$MODEL_FAMILY" "$DATA_IMAGE" \
                --deps "$DEPS_REPO:$DEPS_TAG" --push "$TRAIN_REPO"
        fi
        REF=$(pinned "$TRAIN_REPO" "$TAG")
        ;;
    *) echo "unknown BUILD_KIND: $BUILD_KIND" >&2; exit 2 ;;
esac

echo
echo "image $REF"
echo "$REF" > "$IMAGE_REF_FILE"

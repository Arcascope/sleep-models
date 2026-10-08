#!/usr/bin/env bash
# The build CodeBuild runs (deploy/aws/buildspec.yml). Not for workstations:
# start it with remote_build.sh.
#
# BUILD_KIND picks the image:
#   deps    the dependency image
#   prep    the feature-cache build image
#   data    a data image from the cache in $CACHE_S3/$CACHE_PREFIX/
#   train   $MODEL_FAMILY's training image on $DATA_IMAGE
#
# An image whose tag is already in ECR is not rebuilt: tags are immutable and
# name their inputs, so the existing image is the one this build would make.
# A prep or train build needing a deps image that is not there builds it first.
# The pushed reference, pinned by digest, is written to $IMAGE_REF_FILE.
set -euo pipefail

: "${BUILD_KIND:?}" "${GIT_COMMIT:?}" "${DEPS_REPO:?}" "${IMAGE_REF_FILE:?}"
SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REGION="${AWS_REGION:-us-east-1}"
export GIT_COMMIT

# The tag's digest in ECR, or nothing when the tag is not there.
ecr_digest() {
    local uri=$1 tag=$2
    aws ecr describe-images --repository-name "${uri#*/}" --image-ids "imageTag=$tag" \
        --region "$REGION" --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true
}

pinned() {
    local uri=$1 tag=$2
    echo "$uri:$tag@$(ecr_digest "$uri" "$tag")"
}

DEPS_TAG=$(bash "$SCRIPTS/deps_tag.sh")
ensure_deps() {
    if [[ -n $(ecr_digest "$DEPS_REPO" "$DEPS_TAG") ]]; then
        echo "==> deps $DEPS_REPO:$DEPS_TAG already in ECR"
    else
        bash "$SCRIPTS/build_deps_image.sh" --push "$DEPS_REPO"
    fi
}

case "$BUILD_KIND" in
    deps)
        ensure_deps
        REF=$(pinned "$DEPS_REPO" "$DEPS_TAG")
        ;;
    prep)
        : "${PREP_REPO:?}"
        if [[ -n $(ecr_digest "$PREP_REPO" "$GIT_COMMIT") ]]; then
            echo "==> $PREP_REPO:$GIT_COMMIT already in ECR"
        else
            ensure_deps
            bash "$SCRIPTS/build_prep_image.sh" --deps "$DEPS_REPO:$DEPS_TAG" --push "$PREP_REPO"
        fi
        REF=$(pinned "$PREP_REPO" "$GIT_COMMIT")
        ;;
    data)
        : "${DATA_REPO:?}" "${CACHE_S3:?}" "${CACHE_PREFIX:?}"
        CACHE_DIR="$PWD/cache/$CACHE_PREFIX"
        echo "==> sync ${CACHE_S3%/}/$CACHE_PREFIX/"
        aws s3 sync --only-show-errors --exclude '*' --include '*_features.npz' \
            "${CACHE_S3%/}/$CACHE_PREFIX/" "$CACHE_DIR/"
        echo "    $(find "$CACHE_DIR" -name '*_features.npz' | wc -l) subjects"
        # The tag is the manifest digest, known only after hashing the cache.
        OUT=$(bash "$SCRIPTS/build_data_image.sh" "$CACHE_DIR" | tee /dev/stderr)
        LOCAL_TAG=$(awk '$1 == "image" { print $2 }' <<<"$OUT")
        TAG=${LOCAL_TAG##*:}
        if [[ -n $(ecr_digest "$DATA_REPO" "$TAG") ]]; then
            echo "==> $DATA_REPO:$TAG already in ECR"
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
        if [[ -n $(ecr_digest "$TRAIN_REPO" "$TAG") ]]; then
            echo "==> $TRAIN_REPO:$TAG already in ECR"
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

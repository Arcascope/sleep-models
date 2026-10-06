#!/usr/bin/env bash
# Build and push an image in AWS CodeBuild instead of on this machine.
usage() {
    cat <<'USAGE'
usage: deploy/scripts/remote_build.sh [--no-wait] KIND [ARGS]

  deps                          the dependency image (jax, pisces-lite)
  prep                          the feature-cache build image
  data CACHE_PREFIX             a data image from s3://.../feature_cache/CACHE_PREFIX/
  train FAMILY DATA_IMAGE       FAMILY's training image on DATA_IMAGE

The build uses HEAD as committed (git archive), so uncommitted changes are not
in the image and every image names a commit. An image whose tag is already in
ECR is not rebuilt. Waits for the build and prints the pushed image pinned by
digest, unless --no-wait.

environment:
  ARCASLEEP_STACK   stack name (default arcasleep-train)
  AWS_REGION        region     (default us-east-1)
USAGE
}
set -euo pipefail

STACK="${ARCASLEEP_STACK:-arcasleep-train}"
REGION="${AWS_REGION:-us-east-1}"
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

WAIT=1
if [[ "${1:-}" == --no-wait ]]; then WAIT=0; shift; fi
[[ $# -ge 1 && "$1" != -h && "$1" != --help ]] || { usage; exit 2; }
KIND=$1
shift

ENV_VARS=()
case "$KIND" in
    deps|prep) [[ $# -eq 0 ]] || { usage >&2; exit 2; } ;;
    data) [[ $# -eq 1 ]] || { usage >&2; exit 2; }
          ENV_VARS+=("name=CACHE_PREFIX,value=$1,type=PLAINTEXT") ;;
    train) [[ $# -eq 2 ]] || { usage >&2; exit 2; }
           ENV_VARS+=("name=MODEL_FAMILY,value=$1,type=PLAINTEXT" "name=DATA_IMAGE,value=$2,type=PLAINTEXT") ;;
    *) usage >&2; exit 2 ;;
esac

OUTPUTS=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output text)
output() { awk -v k="$1" '$1 == k { print $2 }' <<<"$OUTPUTS"; }
PROJECT=$(output BuildProjectName)
SOURCE_URI=$(output BuildSourceUri)

COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
if [[ -n $(git -C "$REPO_ROOT" status --porcelain --untracked-files=no) ]]; then
    echo "note: uncommitted changes are not in the build; it uses $COMMIT as committed" >&2
fi

ARCHIVE=$(mktemp --suffix .zip)
trap 'rm -f "$ARCHIVE"' EXIT
git -C "$REPO_ROOT" archive --format=zip -o "$ARCHIVE" HEAD
KEY="${SOURCE_URI#s3://}/$COMMIT.zip"
aws s3 cp --only-show-errors "$ARCHIVE" "s3://$KEY" --region "$REGION"

ENV_VARS+=("name=BUILD_KIND,value=$KIND,type=PLAINTEXT" "name=GIT_COMMIT,value=$COMMIT,type=PLAINTEXT")
BUILD_ID=$(aws codebuild start-build --project-name "$PROJECT" --region "$REGION" \
    --source-type-override S3 --source-location-override "$KEY" \
    --buildspec-override deploy/aws/buildspec.yml \
    --environment-variables-override "${ENV_VARS[@]}" \
    --query 'build.id' --output text)
echo "==> $KIND build $BUILD_ID (commit $COMMIT)"
echo "    logs: aws logs tail /aws/codebuild/$PROJECT --follow --region $REGION"

[[ $WAIT -eq 1 ]] || exit 0

LAST=""
while true; do
    read -r STATUS PHASE < <(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
        --query 'builds[0].[buildStatus,currentPhase]' --output text)
    [[ "$PHASE" == "$LAST" ]] || { echo "    $PHASE"; LAST=$PHASE; }
    [[ "$STATUS" == IN_PROGRESS ]] || break
    sleep 20
done

if [[ "$STATUS" != SUCCEEDED ]]; then
    echo "build $STATUS; see the logs above" >&2
    exit 1
fi
aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
    --query "builds[0].exportedEnvironmentVariables[?name=='IMAGE_REF'].value" --output text

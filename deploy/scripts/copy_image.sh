#!/usr/bin/env bash
# Copy one image between ECR and Artifact Registry on a temporary GCP VM.
#
#   deploy/scripts/copy_image.sh [--dry-run] SOURCE DESTINATION
#
# SOURCE may be tagged or pinned; it is resolved and pinned before launch.
# DESTINATION must be <repo>:<tag>. Existing tags are never overwritten.
# The VM exchanges its Google identity for a restricted AWS role; no registry
# password is supplied by the launcher. See deploy/gcp/image-copy-auth.md.
#
# Required environment:
#   ARCASLEEP_PROJECT      GCP project for the VM
#   COPY_SERVICE_ACCOUNT   dedicated GCP service account for this direction
#   COPY_AWS_ROLE_ARN      corresponding ECR pull or push role
# Optional:
#   COPY_AWS_AUDIENCE      OIDC audience (default arcasleep-image-copy)
#   COPY_ZONE              default us-east4-a
#   COPY_MACHINE           default e2-standard-4
set -euo pipefail

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^#\( \|$\)//'; }
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 2 ]] || { usage >&2; exit 2; }
SRC=$1
DST=$2
: "${ARCASLEEP_PROJECT:?set the GCP project explicitly}"
: "${COPY_SERVICE_ACCOUNT:?set the dedicated image-copy service account}"
: "${COPY_AWS_ROLE_ARN:?set the restricted image-copy AWS role}"
PROJECT=$ARCASLEEP_PROJECT
ZONE=${COPY_ZONE:-us-east4-a}
MACHINE=${COPY_MACHINE:-e2-standard-4}
AUDIENCE=${COPY_AWS_AUDIENCE:-arcasleep-image-copy}
SKOPEO=quay.io/skopeo/stable:v1.16.1
# Official AWS CLI 2.38.1 image, pinned to its manifest digest.
AWSCLI=public.ecr.aws/aws-cli/aws-cli@sha256:733d5be1015825e4038c6016f3fa9954868ddee1f611b73be795a9cc2e706bfd
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

fail() { echo "$*" >&2; exit 2; }
[[ "$PROJECT" =~ ^[a-z][a-z0-9-]+[a-z0-9]$ ]] || fail "invalid GCP project"
[[ "$COPY_SERVICE_ACCOUNT" =~ ^[a-z][a-z0-9-]+@${PROJECT}\.iam\.gserviceaccount\.com$ ]] \
    || fail "COPY_SERVICE_ACCOUNT must be a service account in $PROJECT"
for value in "$SRC" "$DST"; do
    [[ "$value" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/@-]+$ ]] || fail "invalid image reference"
done
[[ "$DST" == *:* && "$DST" != *@* ]] || fail "destination must be <repo>:<tag>"
[[ "$AUDIENCE" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]+$ ]] || fail "invalid OIDC audience"

registry() { echo "${1%%/*}"; }
kind() {
    case "$(registry "$1")" in
        *.dkr.ecr.*.amazonaws.com) echo ecr ;;
        *-docker.pkg.dev) echo ar ;;
        *) fail "not an ECR or Artifact Registry reference: $1" ;;
    esac
}
SRC_KIND=$(kind "$SRC")
DST_KIND=$(kind "$DST")
[[ "$SRC_KIND" != "$DST_KIND" ]] || fail "source and destination are both $SRC_KIND"
if [[ "$SRC_KIND" == ecr ]]; then ECR_REF=$SRC; else ECR_REF=$DST; fi
[[ "$(registry "$ECR_REF")" =~ ^([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$ ]] \
    || fail "invalid ECR registry"
ECR_ACCOUNT=${BASH_REMATCH[1]}
ECR_REGION=${BASH_REMATCH[2]}
[[ "$COPY_AWS_ROLE_ARN" =~ ^arn:aws:iam::${ECR_ACCOUNT}:role/[a-zA-Z0-9+=,.@_/-]+$ ]] \
    || fail "COPY_AWS_ROLE_ARN must be a role in the ECR registry's account"
# Commas delimit gcloud metadata; forbid them even though IAM role names allow them.
[[ "$COPY_AWS_ROLE_ARN" != *,* ]] || fail "role names containing commas are unsupported"

WORK=$(mktemp -d)
VM=""
cleanup_vm() {
    local attempt
    for attempt in 1 2 3; do
        # A delete request can race with automatic deletion. Confirm absence
        # separately, even when the request itself reported success.
        gcloud compute instances delete "$VM" --project="$PROJECT" --zone="$ZONE" --quiet \
            >"$WORK/delete-output" 2>"$WORK/delete-error" || true
        if gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" \
            --format='value(status)' >"$WORK/vm-status" 2>"$WORK/status-error"; then
            echo "VM still exists after deletion attempt $attempt: $VM in $ZONE" >&2
        elif grep -Eq 'was not found|NOT_FOUND|could not be found' "$WORK/status-error"; then
            return 0
        else
            echo "Could not verify deletion of $VM in $ZONE (attempt $attempt)" >&2
            cat "$WORK/status-error" >&2
        fi
        if [[ $attempt -lt 3 ]]; then sleep "$((attempt * 2))"; fi
    done
    cat "$WORK/delete-error" >&2
    return 1
}

cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "$VM" ]] && ! cleanup_vm; then
        echo "VM cleanup failed after 3 attempts: $VM in $ZONE; its 2 h lifetime is the backstop" >&2
        [[ $status -ne 0 ]] || status=1
    fi
    rm -rf "$WORK"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Absence is allowed; permission, connectivity and unexpected errors fail closed.
digest_of() {
    local ref=$1 repo name tag digest
    repo=${ref%@*}
    repo=${repo%:*}
    if [[ $(kind "$ref") == ecr ]]; then
        name=${repo#*/}
        if [[ "$ref" == *@* ]]; then tag="imageDigest=${ref#*@}"; else tag="imageTag=${ref##*:}"; fi
        if digest=$(aws ecr describe-images --region "$ECR_REGION" --registry-id "$ECR_ACCOUNT" \
            --repository-name "$name" --image-ids "$tag" \
            --query 'imageDetails[0].imageDigest' --output text 2>"$WORK/lookup-error"); then
            :
        elif grep -q 'ImageNotFoundException' "$WORK/lookup-error"; then return 0
        else cat "$WORK/lookup-error" >&2; return 1; fi
    else
        local lookup=$ref
        [[ "$ref" != *@* ]] || lookup="$repo@${ref#*@}"
        if digest=$(gcloud artifacts docker images describe "$lookup" --project="$PROJECT" \
            --format='value(image_summary.digest)' 2>"$WORK/lookup-error"); then
            :
        elif grep -Eq 'NOT_FOUND|Image not found' "$WORK/lookup-error"; then return 0
        else cat "$WORK/lookup-error" >&2; return 1; fi
    fi
    [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "invalid digest for $ref" >&2; return 1; }
    echo "$digest"
}

SRC_DIGEST=$(digest_of "$SRC")
[[ -n "$SRC_DIGEST" ]] || { echo "source not found: $SRC" >&2; exit 1; }
EXISTING=$(digest_of "$DST")
if [[ -n "$EXISTING" ]]; then
    [[ "$EXISTING" == "$SRC_DIGEST" ]] || { echo "destination exists with a different digest" >&2; exit 1; }
    echo "==> $DST already exists ($EXISTING)"
    exit 0
fi
# Resolve a mutable source tag only once, so it cannot change during provisioning.
SRC_REPO=${SRC%@*}
SRC_REPO=${SRC_REPO%:*}
SRC="$SRC_REPO@$SRC_DIGEST"

echo "==> $SRC -> $DST"
echo "    $MACHINE in $ZONE; Google identity $COPY_SERVICE_ACCOUNT"
echo "    AWS role $COPY_AWS_ROLE_ARN; audience $AUDIENCE"
[[ $DRY_RUN -eq 0 ]] || exit 0

VM="copy-image-$(date -u +%Y%m%d-%H%M%S)-$RANDOM"
# Set VM before create so even a partially completed API request gets cleanup.
gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
    --image-project=cos-cloud --image-family=cos-stable --boot-disk-size=50GB \
    --service-account="$COPY_SERVICE_ACCOUNT" \
    --scopes=https://www.googleapis.com/auth/cloud-platform --labels=purpose=image-copy \
    --max-run-duration=7200s --instance-termination-action=DELETE \
    --metadata="src=$SRC,dst=$DST,src-kind=$SRC_KIND,skopeo=$SKOPEO,awscli=$AWSCLI,aws-role=$COPY_AWS_ROLE_ARN,aws-audience=$AUDIENCE,ecr-region=$ECR_REGION" \
    --metadata-from-file="startup-script=$ROOT/deploy/gcp/copy-image-startup.sh" >/dev/null

START=$SECONDS
STATUS=""
while (( SECONDS - START < 7500 )); do
    SERIAL=$(gcloud compute instances get-serial-port-output "$VM" --project="$PROJECT" --zone="$ZONE")
    STATUS=$(sed -nE 's/.*copy-image: exit ([0-9]+).*/\1/p' <<<"$SERIAL" | tail -1)
    [[ -z "$STATUS" ]] || break
    sleep 30
done
[[ -n "$STATUS" ]] || { echo "copy did not finish within its VM lifetime" >&2; exit 1; }
grep 'copy-image:' <<<"$SERIAL"
[[ "$STATUS" == 0 ]] || { echo "copy failed (exit $STATUS)" >&2; exit 1; }
COPIED=$(digest_of "$DST")
[[ "$COPIED" == "$SRC_DIGEST" ]] || { echo "destination digest differs from source" >&2; exit 1; }
echo "$DST@$COPIED"

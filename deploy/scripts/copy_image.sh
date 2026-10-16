#!/usr/bin/env bash
# Copy one image between ECR and Artifact Registry, in either direction, on a
# small GCP VM, so the bytes cross the cloud backbone instead of this machine.
#
#   deploy/scripts/copy_image.sh [--dry-run] SOURCE DESTINATION
#
# SOURCE is pinned by digest (<repo>@sha256:...) or tag; DESTINATION is
# <repo>:<tag>. Each is an ECR (*.dkr.ecr.*.amazonaws.com) or Artifact Registry
# (*-docker.pkg.dev) reference, one of each. The VM runs `skopeo copy
# --preserve-digests`, so the destination has the source's manifest digest;
# the script then checks that digest in the destination registry and deletes
# the VM. A tag already in the destination is not copied again.
#
# The ECR password (valid 12 h) reaches the VM as instance metadata; the VM is
# deleted when the copy ends, and GCE deletes it after 2 h in any case.
#
# environment:
#   ARCASLEEP_PROJECT  GCP project (default shiftapp-arcascope)
#   COPY_ZONE          VM zone (default us-east4-a, near us-east-1 and us-central1)
#   COPY_MACHINE       VM machine type (default e2-standard-4)
set -euo pipefail

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^#\( \|$\)//'; }
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 2 ]] || { usage >&2; exit 2; }
SRC=$1
DST=$2
PROJECT="${ARCASLEEP_PROJECT:-shiftapp-arcascope}"
ZONE="${COPY_ZONE:-us-east4-a}"
MACHINE="${COPY_MACHINE:-e2-standard-4}"
SKOPEO=quay.io/skopeo/stable:v1.16.1

registry() { echo "${1%%/*}"; }
kind() {
    case "$(registry "$1")" in
        *.dkr.ecr.*.amazonaws.com) echo ecr ;;
        *-docker.pkg.dev) echo ar ;;
        *) echo "not an ECR or Artifact Registry reference: $1" >&2; exit 2 ;;
    esac
}
[[ "$DST" == *:* && "$DST" != *@* ]] || { echo "destination must be <repo>:<tag>: $DST" >&2; exit 2; }
SRC_KIND=$(kind "$SRC")
DST_KIND=$(kind "$DST")
[[ "$SRC_KIND" != "$DST_KIND" ]] || { echo "source and destination are both $SRC_KIND" >&2; exit 2; }

ECR_REF=$([[ $SRC_KIND == ecr ]] && echo "$SRC" || echo "$DST")
ECR_REGION=$(registry "$ECR_REF" | cut -d. -f4)

# Digest of <repo>:<tag> or <repo>@<digest> in its registry; empty if absent.
digest_of() {
    local ref=$1 repo name tag
    if [[ "$ref" == *@* ]]; then repo=${ref%@*}; else repo=${ref%:*}; fi
    if [[ $(kind "$ref") == ecr ]]; then
        name=${repo#*/}
        if [[ "$ref" == *@* ]]; then tag="imageDigest=${ref#*@}"; else tag="imageTag=${ref##*:}"; fi
        aws ecr describe-images --region "$ECR_REGION" --repository-name "$name" --image-ids "$tag" \
            --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true
    else
        gcloud artifacts docker images describe "$ref" --project="$PROJECT" \
            --format='value(image_summary.digest)' 2>/dev/null || true
    fi
}

SRC_DIGEST=$(digest_of "$SRC")
[[ "$SRC_DIGEST" == sha256:* ]] || { echo "source not found: $SRC" >&2; exit 1; }
EXISTING=$(digest_of "$DST")
if [[ "$EXISTING" == sha256:* ]]; then
    echo "==> $DST already exists ($EXISTING)"
    [[ "$EXISTING" == "$SRC_DIGEST" ]] || { echo "    and its digest differs from the source's $SRC_DIGEST" >&2; exit 1; }
    exit 0
fi

VM="copy-image-$(date -u +%Y%m%d-%H%M%S)"
echo "==> $SRC ($SRC_DIGEST)"
echo "    -> $DST"
echo "    on $VM ($MACHINE, $ZONE)"
[[ $DRY_RUN -eq 0 ]] || exit 0

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
aws ecr get-login-password --region "$ECR_REGION" > "$WORK/ecr-password"
cat > "$WORK/startup.sh" <<'EOF'
#!/bin/bash
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
log() { echo "copy-image: $*" > /dev/ttyS0; }
SRC=$(md src); DST=$(md dst); SRC_KIND=$(md src-kind); ECR=$(md ecr-password)
AR=$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | sed -E 's/.*"access_token":"([^"]+)".*/\1/')
if [[ $SRC_KIND == ecr ]]; then SRC_CREDS="AWS:$ECR"; DST_CREDS="oauth2accesstoken:$AR"
else SRC_CREDS="oauth2accesstoken:$AR"; DST_CREDS="AWS:$ECR"; fi
log "start"
docker run --rm "$(md skopeo)" copy --retry-times 3 --preserve-digests \
  --src-creds "$SRC_CREDS" --dest-creds "$DST_CREDS" "docker://$SRC" "docker://$DST" > /tmp/copy.log 2>&1
status=$?
tail -5 /tmp/copy.log | while read -r line; do log "$line"; done
log "exit $status"
EOF

gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
    --image-project=cos-cloud --image-family=cos-stable --boot-disk-size=50GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform --labels=purpose=image-copy \
    --max-run-duration=7200s --instance-termination-action=DELETE \
    --metadata="src=$SRC,dst=$DST,src-kind=$SRC_KIND,skopeo=$SKOPEO" \
    --metadata-from-file="startup-script=$WORK/startup.sh,ecr-password=$WORK/ecr-password" >/dev/null
cleanup() { gcloud compute instances delete "$VM" --project="$PROJECT" --zone="$ZONE" --quiet >/dev/null 2>&1 || true; }
trap 'cleanup; rm -rf "$WORK"' EXIT

START=$(date +%s)
until gcloud compute instances get-serial-port-output "$VM" --project="$PROJECT" --zone="$ZONE" 2>/dev/null \
        | grep -q 'copy-image: exit'; do
    sleep 30
done
gcloud compute instances get-serial-port-output "$VM" --project="$PROJECT" --zone="$ZONE" | grep 'copy-image:'
echo "    $(( ($(date +%s) - START) / 60 )) min"
STATUS=$(gcloud compute instances get-serial-port-output "$VM" --project="$PROJECT" --zone="$ZONE" \
    | sed -nE 's/.*copy-image: exit ([0-9]+).*/\1/p' | tail -1)
[[ "$STATUS" == 0 ]] || { echo "copy failed (exit $STATUS)" >&2; exit 1; }

COPIED=$(digest_of "$DST")
[[ "$COPIED" == "$SRC_DIGEST" ]] || { echo "destination digest $COPIED differs from $SRC_DIGEST" >&2; exit 1; }
echo "$DST@$COPIED"

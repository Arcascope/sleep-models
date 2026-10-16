#!/bin/bash
# GCE startup script. Metadata contains configuration only, never passwords.
set +x
set -euo pipefail
umask 077
AUTH=""
log() { printf 'copy-image: %s\n' "$*" > "${COPY_SERIAL_DEVICE:-/dev/ttyS0}"; }
finish() {
    local status=$?
    trap - EXIT
    [[ -z "$AUTH" ]] || rm -rf "$AUTH"
    log "exit $status"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
md() {
    curl --fail --silent --show-error --connect-timeout 5 --max-time 15 \
        -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}
SRC=$(md src)
DST=$(md dst)
SRC_KIND=$(md src-kind)
ROLE=$(md aws-role)
AUDIENCE=$(md aws-audience)
REGION=$(md ecr-region)
AWSCLI=$(md awscli)
SKOPEO=$(md skopeo)
[[ "$SRC_KIND" == ecr || "$SRC_KIND" == ar ]]
AUTH=$(mktemp -d "${COPY_RUNTIME_DIR:-/run}/arcasleep-copy.XXXXXX")
chmod 700 "$AUTH"
log "authenticating"
# Google signs an identity token for this VM's dedicated service account.
# The JWT is read from a file by AWS CLI, keeping it out of process arguments.
curl --fail --silent --show-error --connect-timeout 5 --max-time 15 \
    -H 'Metadata-Flavor: Google' --get --data-urlencode "audience=$AUDIENCE" \
    --data-urlencode 'format=full' \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity \
    > "$AUTH/google-identity.jwt"
docker run --rm --log-driver=none --network=host --volume "$AUTH:/auth:ro" "$AWSCLI" \
    sts assume-role-with-web-identity --role-arn "$ROLE" \
    --role-session-name image-copy --duration-seconds 900 --region "$REGION" \
    --web-identity-token file:///auth/google-identity.jwt \
    --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text \
    > "$AUTH/sts.tsv"
read -r ACCESS_KEY SECRET_KEY SESSION_TOKEN < "$AUTH/sts.tsv"
[[ -n "$ACCESS_KEY" && -n "$SECRET_KEY" && -n "$SESSION_TOKEN" ]]
printf '[default]\naws_access_key_id = %s\naws_secret_access_key = %s\naws_session_token = %s\n' \
    "$ACCESS_KEY" "$SECRET_KEY" "$SESSION_TOKEN" > "$AUTH/aws-credentials"
unset ACCESS_KEY SECRET_KEY SESSION_TOKEN
docker run --rm --log-driver=none --network=host --volume "$AUTH:/auth:ro" \
    --env AWS_SHARED_CREDENTIALS_FILE=/auth/aws-credentials \
    --env AWS_EC2_METADATA_DISABLED=true --env AWS_PAGER= "$AWSCLI" \
    ecr get-login-password --region "$REGION" > "$AUTH/ecr-password"
# The GCP access token has only the repository permissions assigned to this SA.
AR_TOKEN=$(curl --fail --silent --show-error --connect-timeout 5 --max-time 15 \
    -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
    | sed -nE 's/.*"access_token"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p')
ECR_TOKEN=$(cat "$AUTH/ecr-password")
[[ -n "$AR_TOKEN" && -n "$ECR_TOKEN" ]]
if [[ "$SRC_KIND" == ecr ]]; then ECR=$SRC; AR=$DST; else ECR=$DST; AR=$SRC; fi
ECR_AUTH=$(printf 'AWS:%s' "$ECR_TOKEN" | base64 | tr -d '\n')
AR_AUTH=$(printf 'oauth2accesstoken:%s' "$AR_TOKEN" | base64 | tr -d '\n')
# Registry names have already been validated by the launcher; no credentials
# appear in docker arguments, environment variables, metadata or serial output.
printf '{"auths":{"%s":{"auth":"%s"},"%s":{"auth":"%s"}}}\n' \
    "${ECR%%/*}" "$ECR_AUTH" "${AR%%/*}" "$AR_AUTH" > "$AUTH/registry-auth.json"
unset ECR_TOKEN AR_TOKEN ECR_AUTH AR_AUTH
rm -f "$AUTH/google-identity.jwt" "$AUTH/sts.tsv" "$AUTH/aws-credentials" "$AUTH/ecr-password"
log "start"
docker run --rm --log-driver=none --volume "$AUTH:/auth:ro" "$SKOPEO" copy \
    --authfile /auth/registry-auth.json --retry-times 3 --preserve-digests \
    "docker://$SRC" "docker://$DST"

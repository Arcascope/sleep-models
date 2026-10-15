#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/arcasleep-train.log /dev/ttyS0) 2>&1
# Filled with shell-quoted values by run_task_gcp.py.
RUN_ID=__RUN_ID__
PHASE=__PHASE__
IMAGE=__IMAGE_REF__
RESULTS_BUCKET=__RESULTS_BUCKET__
ENV_PREFIX=__ENV_PREFIX__
TRAINING_CONFIG=__TRAINING_CONFIG__
TIMEOUT_SECONDS=__TIMEOUT_SECONDS__
HARNESS_ARGS=(__HARNESS_ARGS__)
EXTRA_ENV=(__EXTRA_ENV__)
# /mnt is a small tmpfs on COS; results belong on the stateful disk.
RESULTS=/var/lib/arcasleep-results
SDK=gcr.io/google.com/cloudsdktool/google-cloud-cli:stable
export DOCKER_CONFIG=/var/lib/arcasleep-docker
mkdir -p "$DOCKER_CONFIG"
mkdir -p "$RESULTS/$RUN_ID/gcp/$PHASE"
status=1
finish() {
    status=$?
    trap - EXIT
    set +e
    docker stop --time=30 arcasleep-training >/dev/null 2>&1
    printf '%s\n' "$status" > "$RESULTS/$RUN_ID/gcp/$PHASE/exit-code.txt"
    cp /var/log/arcasleep-train.log "$RESULTS/$RUN_ID/gcp/$PHASE/startup.log"
    docker run --rm --network=host --volume "$RESULTS/$RUN_ID:/results:ro" \
      "$SDK" gcloud storage rsync --recursive /results \
      "gs://$RESULTS_BUCKET/training_outputs/$RUN_ID/"
    upload=$?
    echo "ARCASLEEP_FINISHED run=$RUN_ID phase=$PHASE training_exit=$status upload_exit=$upload"
    exit "$status"
}
trap finish EXIT
df -h "$RESULTS"
available_kb=$(df -Pk "$RESULTS" | awk 'NR == 2 { print $4 }')
[[ "$available_kb" -ge 1048576 ]] || { echo "Results need at least 1 GiB of free disk"; exit 1; }
echo "==> installing GPU driver"
cos-extensions install gpu -- -version=latest
mount --bind /var/lib/nvidia /var/lib/nvidia
mount -o remount,exec /var/lib/nvidia
/var/lib/nvidia/bin/nvidia-smi
echo "==> authenticating Artifact Registry"
TOKEN=$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | sed -n 's/.*"access_token": *"\([^"]*\)".*/\1/p')
[[ -n "$TOKEN" ]]
printf '%s' "$TOKEN" | docker login --username oauth2accesstoken --password-stdin us-central1-docker.pkg.dev
unset TOKEN
echo "==> pulling training image"
docker pull "$IMAGE"
echo "==> training ${HARNESS_ARGS[*]:-all phases} with $TRAINING_CONFIG on CUDA"
# The harness's own S3 upload stays disabled (empty prefix); finish() uploads
# the results directory to GCS, including partial outputs after a failure.
timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" docker run --rm --init --name arcasleep-training \
  --shm-size=2g \
  --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64:ro \
  --volume /var/lib/nvidia/bin:/usr/local/nvidia/bin:ro \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --volume "$RESULTS:/results" \
  --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --env JAX_PLATFORMS=cuda --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env PYTHONUNBUFFERED=1 \
  --env "${ENV_PREFIX}_TRAINING_CONFIG=$TRAINING_CONFIG" \
  --env "${ENV_PREFIX}_RESULTS_DIR=/results" --env "${ENV_PREFIX}_RUN_ID=$RUN_ID" \
  --env "${ENV_PREFIX}_RESULTS_S3_PREFIX=" \
  "${EXTRA_ENV[@]}" --entrypoint /bin/bash \
  "$IMAGE" -c 'set -euo pipefail; /usr/local/nvidia/bin/nvidia-smi; python -c '\''import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'\''; exec bash scripts/harness.sh "$@"' harness "${HARNESS_ARGS[@]}"

#!/usr/bin/env bash
# Run UNet fold0 with the committed two-epoch smoke recipe on Cloud Batch.
#
#   bash deploy/scripts/run_smoke_gcp.sh IMAGE [REGION]
#
# Resolves the image to a digest before submitting. Results and partial
# outputs go to gs://arcascope-sleep-models-data/training_outputs/RUN_ID/.
# The job uses one on-demand L4; its task stops after one hour, with no retry.
# ARCASLEEP_PROJECT defaults to shiftapp-arcascope.
set -euo pipefail
[[ $# -ge 1 && $# -le 2 ]] || {
    echo "Usage: $0 IMAGE [REGION]" >&2
    exit 2
}
PROJECT="${ARCASLEEP_PROJECT:-shiftapp-arcascope}"
REGION="${2:-us-central1}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=$1
DIGEST=$(gcloud artifacts docker images describe "$IMAGE" \
    --project="$PROJECT" --format='value(image_summary.digest)')
[[ "$DIGEST" == sha256:* ]] || { echo "missing image digest: $IMAGE" >&2; exit 1; }
PINNED="${IMAGE%@*}@$DIGEST"
RUN_ID="unet-gcp-smoke-$(date -u +%Y%m%d-%H%M%S)"
CONFIG=$(mktemp -t arcasleep-smoke)
trap 'rm -f "$CONFIG"' EXIT
python3 - "$ROOT/deploy/gcp/unet-smoke.json" "$CONFIG" "$PINNED" "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

source, destination, image, run_id = sys.argv[1:]
config = json.loads(Path(source).read_text())
training = config["taskGroups"][0]["taskSpec"]["runnables"][0]
training["container"]["imageUri"] = image
training["environment"]["variables"]["UNET_RUN_ID"] = run_id
Path(destination).write_text(json.dumps(config, indent=2) + "\n")
PY
echo "==> $RUN_ID ($REGION)"
echo "    image: $PINNED"
echo "    results: gs://arcascope-sleep-models-data/training_outputs/$RUN_ID/"
gcloud batch jobs submit "$RUN_ID" --project="$PROJECT" \
    --location="$REGION" --config="$CONFIG"

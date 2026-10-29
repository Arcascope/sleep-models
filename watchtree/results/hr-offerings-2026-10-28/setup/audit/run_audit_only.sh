#!/bin/bash
# Per-night audit of every cohort (no aggregation); see run.sh for the full pipeline.
set -euo pipefail
WT=/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees
RUN=$WT/watchtree/results/bpm-audit-exp4-2026-10-28
export PYTHONPATH=$WT/watchtree/src:$WT/core/src
export WATCHTREE_DATA_ROOT="/media/eric/Extreme SSD/pisces2-data"
cd "$WT/watchtree"
for cohort in sleepaccel aauwss wearanize weaver_garmin weaver_fitbit; do
    /home/eric/venvs/arcasleep/bin/python "$RUN/setup/audit_night.py" --cohort "$cohort" --out "$RUN/local-v2" --workers 6 2>&1 | tee "$RUN/local-v2/audit-$cohort.log"
done

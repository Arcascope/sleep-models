#!/bin/bash
# BPM stream audit (Experiment 4, step 1): per-night audit for every cohort with heart rate, then aggregation.
# Run from the sleep-models-instinct-trees worktree; raw cohorts from the SSD; <= 6 workers, 1 thread each.
set -euo pipefail
WT=/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees
RUN=$WT/watchtree/results/bpm-audit-exp4-2026-10-28
export PYTHONPATH=$WT/watchtree/src:$WT/core/src
export WATCHTREE_DATA_ROOT="/media/eric/Extreme SSD/pisces2-data"
cd "$WT/watchtree"
PY=/home/eric/venvs/arcasleep/bin/python
LIMIT=${LIMIT:-}
for cohort in sleepaccel aauwss wearanize weaver_garmin weaver_fitbit; do
    $PY "$RUN/setup/audit_night.py" --cohort "$cohort" --out "$RUN/local-v2" --workers 6 ${LIMIT:+--limit $LIMIT} 2>&1 | tee "$RUN/local-v2/audit-$cohort.log"
done
$PY "$RUN/setup/aggregate.py" --run "$RUN" 2>&1 | tee "$RUN/aggregate.log"

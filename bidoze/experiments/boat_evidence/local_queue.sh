#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PY=${BOAT_EVIDENCE_PYTHON:-/home/eric/venvs/.bidoze/bin/python}
RUN=${BOAT_EVIDENCE_RUN:-$ROOT/bidoze/results/boat-evidence-2026-10-11}
export PYTHONPATH=$ROOT/core/src:$ROOT/bidoze/src:$ROOT/unet/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2
export GIT_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
for PHASE in ${BOAT_EVIDENCE_LOCAL_PHASES:-E/seed0 D/seed0}; do
    FOLD=${PHASE%/*}
    SEED=${PHASE##*seed}
    OUT=$RUN/local/full/$FOLD/seed$SEED
    [[ -f "$OUT/complete.json" ]] && continue
    mkdir -p "$OUT"
    "$PY" -u "$HERE/train.py" "$RUN/setup/data" "$OUT" --arm full --fold "$FOLD" --seed "$SEED" --max-epochs 100 > "$OUT/training.log" 2>&1
done

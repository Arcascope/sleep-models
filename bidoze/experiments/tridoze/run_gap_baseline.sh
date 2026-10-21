#!/usr/bin/env bash
# Gap checks of the 2026-10-15 independent ×3 baseline (seeds 0-2) on pilot draws 0 and 1.
# Usage: run_gap_baseline.sh [draw ...]   (CPU unless JAX_PLATFORMS is set otherwise; existing outputs are skipped)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$(cd "$HERE/../../.." && pwd)"
BASE=/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} PYTHONPATH=$CODE/core/src:$CODE/bidoze/src PYTHONUNBUFFERED=1
for d in "${@:-0 1}"; do
  for d in $d; do
    ck=$(for s in 0 1 2; do printf '%s,' "$BASE/independent/draw$d/model_outputs/model_draw${d}_seed_$s.pisces"; done)
    /home/eric/venvs/arcasleep/bin/python "$HERE/gap_checks.py" --draw "$d" --out "$CODE/bidoze/results/tridoze-2026-10-16/gap_checks" \
      --models "independent_x3=${ck%,}" $(for s in 0 1 2; do printf 'independent_seed%s=%s ' $s "$BASE/independent/draw$d/model_outputs/model_draw${d}_seed_$s.pisces"; done)
  done
done

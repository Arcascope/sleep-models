#!/usr/bin/env bash
set -euo pipefail
: "${SLEEP_ANALYSIS_DATA:?Set SLEEP_ANALYSIS_DATA to the cleaned cohort root}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
COMB_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMB_OUTPUT_DIR=bidoze/results/comb-floor-2026-10-20
mkdir -p "$COMB_OUTPUT_DIR/setup" "$COMB_OUTPUT_DIR/features" "$COMB_OUTPUT_DIR/figures"
for stage in extract models raw figures stage_highres representation audit; do
  python "$COMB_SOURCE_DIR/$stage.py" > "$COMB_OUTPUT_DIR/$stage.log" 2>&1
done
cp "$COMB_SOURCE_DIR/"*.py "$COMB_SOURCE_DIR/run.sh" "$COMB_SOURCE_DIR/README.md" "$COMB_OUTPUT_DIR/setup/"

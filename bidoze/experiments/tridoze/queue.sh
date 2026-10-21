#!/usr/bin/env bash
# Run TriDoze fits one at a time, each under its own hold of the shared GPU lock, so that other jobs can take
# the lock between fits. Each fit runs through burst.sh with a budget of BUDGET seconds (default 4000), so a
# fit is capped at BUDGET − 150 s (− 200 s for tridoze_j*) and resumable from its saved state.
# Usage: queue.sh <run root> <arm>:<draw> [<arm>:<draw> ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN=$1; shift
for spec in "$@"; do
  flock ~/.cache/arcasleep-gpu.lock bash "$HERE/burst.sh" "$RUN" "${BUDGET:-4000}" "$spec"
done

#!/usr/bin/env bash
# Score every listed fit on its draw's training recordings (train_predictions.py, CPU), waiting for
# each fit to finish (its checkpoint present and no saved fit state). Items: <arm>:<draw>[:<source arm dir>];
# arms named tridoze_j* are scored member by member.
# Usage: train_preds_all.sh <run root> <item> [<item> ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$(cd "$HERE/../../.." && pwd)"
RUN="$(cd "$1" && pwd)"; shift
export PYTHONPATH=$CODE/core/src:$CODE/bidoze/src PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
PY=${PY:-/home/eric/venvs/arcasleep/bin/python}
cd "$CODE/bidoze"
for item in "$@"; do
  IFS=: read -r arm d src <<< "$item"
  dir=${src:-$RUN/$arm}/draw$d
  glob="model_draw${d}_seed_*.pisces"; [[ $arm == tridoze_j* ]] && glob="model_draw${d}_seed_0_member*.pisces"
  until compgen -G "$dir/model_outputs/$glob" > /dev/null && ! compgen -G "$dir/fit_state_seed*.pkl" > /dev/null \
        && [[ -n $(ls "$dir/predictions" 2>/dev/null) ]]; do sleep 120; done
  $PY "$HERE/train_predictions.py" "$RUN" "$arm" "$d" "$glob" ${src:+"$src"} || echo "FAILED $item"
done
echo "train predictions done: $*"

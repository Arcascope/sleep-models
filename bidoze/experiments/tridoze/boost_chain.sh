#!/usr/bin/env bash
# Train the boosted members of one draw in sequence: member 0 is bidoze_g (seed 0); member k (1, 2)
# is bidoze_g_boost<k>, trained with the SAMME epoch weights (boost_weights.py) of the ensemble of
# members 0..k−1 on the draw's fit recordings. Scoring on the training recordings runs on the CPU;
# each fit runs through burst.sh under its own hold of the GPU lock. Waits for bidoze_g's fit.
# Usage: boost_chain.sh <run root> <draw>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$(cd "$HERE/../../.." && pwd)"
RUN="$(cd "$1" && pwd)"; d=$2
export PYTHONPATH=$CODE/core/src:$CODE/bidoze/src PYTHONUNBUFFERED=1
PY=${PY:-/home/eric/venvs/arcasleep/bin/python}
cd "$CODE/bidoze"
until [[ -d $RUN/bidoze_g/draw$d/predictions/draw${d}_seed_0 && ! -f $RUN/bidoze_g/draw$d/fit_state_seed0.pkl ]]; do sleep 120; done
members=("bidoze_g:0")
for k in 1 2; do
  npzs=()
  for m in "${members[@]}"; do
    arm=${m%%:*}; s=${m##*:}
    JAX_PLATFORMS=cpu $PY "$HERE/train_predictions.py" "$RUN" "$arm" "$d" "model_draw${d}_seed_$s.pisces" || exit 1
    npzs+=("$RUN/$arm/draw$d/train_predictions/draw${d}_seed_$s.npz")
  done
  out=$RUN/bidoze_g_boost$k/draw$d
  [[ -f $out/epoch_weights.npz ]] || $PY "$HERE/boost_weights.py" "$out/epoch_weights" "${npzs[@]}" > /dev/null || exit 1
  flock ~/.cache/arcasleep-gpu.lock bash "$HERE/burst.sh" "$RUN" "${BUDGET:-4000}" "bidoze_g_boost$k:$d"
  [[ -d $out/predictions/draw${d}_seed_$k && ! -f $out/fit_state_seed$k.pkl ]] || { echo "boost$k draw$d did not finish"; exit 1; }
  members+=("bidoze_g_boost$k:$k")
done
JAX_PLATFORMS=cpu $PY "$HERE/train_predictions.py" "$RUN" bidoze_g_boost2 "$d" "model_draw${d}_seed_2.pisces"
echo "boost chain draw$d done"

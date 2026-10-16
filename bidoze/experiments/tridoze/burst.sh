#!/usr/bin/env bash
# Run listed TriDoze fits one after another inside one GPU burst.
# Usage: burst.sh [--dry-run] <run root> <seconds budget> <arm>:<draw> [<arm>:<draw> ...]
# Launched by the lead as:
#   timeout 15m flock ~/.cache/arcasleep-gpu.lock bash burst.sh <run root> 840 <arm>:<draw> ...
# Before each fit: remaining = budget - elapsed; max_train_seconds = remaining - 150 (200 for tridoze_j, whose members are then scored); the fit is
# skipped if remaining < 240 s, or if its predictions exist and it has no saved fit state. A capped fit saves its
# state to <arm>/draw<d>/fit_state_seed<s>.pkl, and the next burst listing the same fit resumes it. A fit writes <arm>/draw<d>/training.log;
# the per-fit training json is <arm>/draw<d>/config-burst/training.json. After a tridoze_j fit,
# member_predictions.py scores the members. --dry-run prints the commands and writes the jsons only.
set -uo pipefail
DRY=0; [[ ${1:-} == --dry-run ]] && { DRY=1; shift; }
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$(cd "$HERE/../../.." && pwd)"
RUN="$(cd "$1" && pwd)"; BUDGET=$2; shift 2
S=$RUN/setup
MARGIN=150; MIN_REMAINING=240
export PYTHONPATH=$CODE/core/src:$CODE/bidoze/src PYTHONUNBUFFERED=1
PY=${PY:-/home/eric/venvs/arcasleep/bin/python}
ARCASLEEP=${ARCASLEEP:-/home/eric/venvs/arcasleep/bin/arcasleep}
cd "$CODE/bidoze"
START=$(date +%s)
LOG=$RUN/bursts.log
N=$(( $(grep -c '^burst [0-9]* start' "$LOG" 2>/dev/null || true) + 1 ))
emit() { echo "$*"; [[ $DRY == 1 ]] || echo "$*" >> "$LOG"; }
emit "burst $N start $(date '+%F %T') budget ${BUDGET}s fits: $*$([[ $DRY == 1 ]] && echo ' (dry run)')"
for spec in "$@"; do
  arm=${spec%%:*}; d=${spec##*:}
  out=$RUN/$arm/draw$d
  remaining=$(( BUDGET - ($(date +%s) - START) ))
  # BURST_DEADLINE (epoch seconds, set before `timeout` and `flock`) also counts the lock wait.
  if [[ -n ${BURST_DEADLINE:-} ]] && (( BURST_DEADLINE - $(date +%s) < remaining )); then
    remaining=$(( BURST_DEADLINE - $(date +%s) ))
  fi
  last=$($PY -c "import json,sys;t=json.load(open(sys.argv[1]));print(t['run_seed']+t.get('seed_count',1)-1)" "$S/training-$arm.json")
  if [[ -d "$out/predictions/draw${d}_seed_$last" && ! -f "$out/fit_state_seed$last.pkl" ]]; then emit "skip $arm draw$d: predictions exist"; continue; fi
  if (( remaining < MIN_REMAINING )); then emit "skip $arm draw$d: ${remaining}s remain"; continue; fi
  margin=$MARGIN; [[ $arm == tridoze_j* ]] && margin=200
  cap=$(( remaining - margin ))
  mkdir -p "$out/config-burst"
  $PY -c "import json,sys;t=json.load(open(sys.argv[1]));t['max_train_seconds']=int(sys.argv[3]);t['fit_state_path']=sys.argv[4];json.dump(t,open(sys.argv[2],'w'),indent=2)" \
    "$S/training-$arm.json" "$out/config-burst/training.json" "$cap" "$out/fit_state.pkl"
  cmd=(env FEATURE_CACHE_DIR=$S/cache/draw$d $ARCASLEEP bidoze train --cache-only -pc config/inputs.json
       -mc "$S/model-$arm.json" -tc "$out/config-burst/training.json" -mtc config/metrics.json --split-seed 0
       -cvc "$S/cv/draw$d.json" -extn "$out")
  if [[ $DRY == 1 ]]; then
    echo "[dry] $arm draw$d remaining ${remaining}s cap ${cap}s"
    echo "[dry]   ${cmd[*]} > $out/training.log 2>&1"
    [[ $arm == tridoze_j* ]] && echo "[dry]   $PY $HERE/../joint_ensemble/member_predictions.py $RUN $arm $d >> $out/training.log 2>&1"
    echo "[dry]   $out/config-burst/training.json: $(tr -d ' \n' < "$out/config-burst/training.json")"
    continue
  fi
  echo "==> $arm draw$d $(date +%T) cap ${cap}s"
  mkdir -p "$out"
  "${cmd[@]}" >> "$out/training.log" 2>&1 || { emit "FAILED $arm draw$d (see $arm/draw$d/training.log)"; continue; }
  if [[ $arm == tridoze_j* ]]; then
    $PY "$HERE/../joint_ensemble/member_predictions.py" "$RUN" "$arm" "$d" >> "$out/training.log" 2>&1 || emit "FAILED members $arm draw$d"
  fi
  emit "$($PY "$HERE/fit_summary.py" "$out/training.log" "$arm" "$d" "$cap")"
done
emit "burst $N end $(date '+%F %T') elapsed $(( $(date +%s) - START ))s"

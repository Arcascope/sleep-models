source /home/eric/Engineering/Work/autopisces/sleep-models-hr-offerings/watchtree/results/accel-hr-v2-2026-10-28/setup/env.sh
export PYTHONPATH=$PYTHONPATH:$R/setup
export N=$WT/watchtree/results/accel-hr-envelope-2026-10-29
export PYTHONPATH=$N/setup:$PYTHONPATH
$PY $N/setup/build_est.py "$@"
echo done > $N/est/build_${HP:-0}_$1.done

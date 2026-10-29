export WT=/home/eric/Engineering/Work/autopisces/sleep-models-hr-offerings
export PYTHONPATH=$WT/watchtree/src:$WT/core/src:$WT/watchtree/results/hr-offerings-2026-10-28/setup
export WATCHTREE_DATA_ROOT="/media/eric/Extreme SSD/pisces2-data"
export PY=/home/eric/venvs/arcasleep/bin/python
export R=$WT/watchtree/results/accel-hr-v2-2026-10-28
export OLD=$WT/watchtree/results/hr-offerings-2026-10-28
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS; do export $v=1; done

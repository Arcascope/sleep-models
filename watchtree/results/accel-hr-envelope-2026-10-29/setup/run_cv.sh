source /home/eric/Engineering/Work/autopisces/sleep-models-hr-offerings/watchtree/results/accel-hr-v2-2026-10-28/setup/env.sh
export PYTHONPATH=$PYTHONPATH:$R/setup
export N=$WT/watchtree/results/accel-hr-envelope-2026-10-29
TAG=${HP:+hp_}
$PY $N/setup/cv.py "$1" > $N/cv_${TAG}$1.log 2>&1

#!/bin/bash
set -euo pipefail
WT=/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees
export PYTHONPATH=$WT/watchtree/src:$WT/core/src
export WATCHTREE_DATA_ROOT="/media/eric/Extreme SSD/pisces2-data"
/home/eric/venvs/arcasleep/bin/python "$(dirname "$0")/explore.py"

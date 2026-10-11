#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IFS=/ read -r ARM FOLD SEED <<< "$1"
[[ "$ARM" =~ ^(head|full|land)$ && "$FOLD" =~ ^[A-E]$ && "$SEED" =~ ^seed[0-2]$ ]] || exit 2
export PYTHONPATH=/opt/arcasleep/core/src:/opt/arcasleep/bidoze/src:/opt/arcasleep/unet/src
python -u "$HERE/train.py" /opt/boat-evidence-data "$2" --arm "$ARM" --fold "$FOLD" --seed "${SEED#seed}" --max-epochs 100

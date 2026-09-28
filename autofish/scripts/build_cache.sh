#!/bin/bash

VOLUME_DIR=${1:-$DATA_ROOT_DIR}

if [[ -z "${VOLUME_DIR}" ]]; then
    exit 1;
fi

autofish process -pc config/inputs.json -dd "${VOLUME_DIR}/weaver" -dd "${VOLUME_DIR}/SleepAccel" -dd "${VOLUME_DIR}/SleepAccel-Clinical" --batch-size 16

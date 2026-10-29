"""Foundry: accelerometer, PSG and heart-rate files per recording.

Heart rate is the first two columns (time, bpm) of the recording's ``cleaned_heartrate/`` file;
a recording without one has none.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pisces_lite.datasets.data_set_object import DataSetObject

from arcasleep_watchtree.loaders.common import (
    ACCEL_COLUMNS,
    ACCELEROMETER,
    HEARTRATE,
    HR_COLUMNS,
    PSG,
    PSG_COLUMNS,
    cleaned_files,
    data_set,
    no_heart_rate,
    renamed,
)


def load(root: Path) -> DataSetObject:
    root = Path(root)
    files = cleaned_files(root)
    name = getattr(files.config, "name", None) or root.name

    def heartrate(id_: str) -> pd.DataFrame:
        if id_ not in files.get_feature_files(HEARTRATE):
            return no_heart_rate()
        frame = files.get_feature_data(HEARTRATE, id_).iloc[:, : len(HR_COLUMNS)]
        frame.columns = HR_COLUMNS
        return frame

    return data_set(
        name,
        root,
        files.ids,
        accelerometer=renamed(lambda id_: files.get_feature_data(ACCELEROMETER, id_), ACCEL_COLUMNS),
        psg=renamed(lambda id_: files.get_feature_data(PSG, id_), PSG_COLUMNS),
        heartrate=heartrate,
    )

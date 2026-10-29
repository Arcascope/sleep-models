"""SleepAccel and SleepAccel-Clinical: accelerometer, PSG and heart-rate files per subject.

Heart rate is ``cleaned_heartrate/<id>_cleaned_hr.out``; a subject without that file has none.
"""

from __future__ import annotations

from pathlib import Path

from pisces_lite.datasets.data_set_object import DataSetObject

from arcasleep_tree.loaders.common import (
    ACCEL_COLUMNS,
    ACCELEROMETER,
    PSG,
    PSG_COLUMNS,
    cleaned_files,
    data_set,
    read_hr_out,
    renamed,
)

HR_DIR = "cleaned_heartrate"
HR_SUFFIX = "_cleaned_hr.out"


def load(root: Path) -> DataSetObject:
    root = Path(root)
    files = cleaned_files(root)
    name = getattr(files.config, "name", None) or root.name
    return data_set(
        name,
        root,
        files.ids,
        accelerometer=renamed(lambda id_: files.get_feature_data(ACCELEROMETER, id_), ACCEL_COLUMNS),
        psg=renamed(lambda id_: files.get_feature_data(PSG, id_), PSG_COLUMNS),
        heartrate=lambda id_: read_hr_out(root / HR_DIR / f"{id_}{HR_SUFFIX}"),
    )

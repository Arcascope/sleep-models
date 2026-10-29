"""AAUWSS: the Aalborg University Wearable Sleep Study download.

Labels are the manual annotation workbooks, whose epoch start times are Copenhagen wall-clock
time. The Empatica E4 files start at a Unix time. Everything is placed on Unix time; epochs the
workbook omits are not scored.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.datasets.constants import PSG_MASK
from pisces_lite.datasets.data_set_object import DataSetObject

from arcasleep_watchtree.loaders import empatica
from arcasleep_watchtree.loaders.common import (
    EPOCH_S,
    data_set,
    map_codes,
    psg_frame,
    sheet_records,
)

NAME = "aauwss"
ANNOTATIONS_DIR = "annotations"
ANNOTATION_SUFFIX = "_manual_annotation.xlsx"
EMPATICA_DIR = "empatica"
EPOCH_START = "Event Start Time"
STAGE = "Sleep Stage"
LOCAL_TZ = "Europe/Copenhagen"
EXCEL_EPOCH = pd.Timestamp("1899-12-30")
STAGES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 5}


def night_id(recording: str) -> str:
    """``subject_01`` -> ``s01``."""
    return f"s{re.fullmatch(r'subject_(\d+)', recording).group(1)}"


def read_annotations(path: Path) -> pd.DataFrame:
    """One row per 30-s epoch from the first scored to the last, on Unix time."""
    rows = sheet_records(path)
    local = EXCEL_EPOCH + pd.to_timedelta([r[EPOCH_START] for r in rows], unit="D")
    start = local.round("1s").tz_localize(LOCAL_TZ).asi8 / 1e9
    codes = map_codes([r[STAGE] for r in rows], STAGES)
    index = np.rint((start - start[0]) / EPOCH_S).astype(int)
    if np.any(np.diff(index) < 1):
        raise ValueError(f"{path.name}: scored epochs are not increasing on a 30-s grid")
    filled = np.full(index[-1] + 1, PSG_MASK)
    filled[index] = codes
    return psg_frame(start[0] + EPOCH_S * np.arange(filled.size), filled)


def load(root: Path) -> DataSetObject:
    root = Path(root)
    nights = {}
    for path in sorted((root / ANNOTATIONS_DIR).glob(f"*{ANNOTATION_SUFFIX}")):
        recording = path.name.removesuffix(ANNOTATION_SUFFIX)
        folder = root / EMPATICA_DIR / recording
        if (folder / empatica.ACC_FILE).exists():
            nights[night_id(recording)] = (path, folder)

    def e4(id_: str) -> tuple[empatica.E4, float]:
        folder = nights[id_][1]
        recording = empatica.read_e4(lambda name: (folder / name).read_bytes())
        return recording, recording.acc_start

    return data_set(
        NAME,
        root,
        nights,
        accelerometer=lambda id_: empatica.accelerometer(*e4(id_)),
        psg=lambda id_: read_annotations(nights[id_][0]),
        heartrate=lambda id_: empatica.heart_rate(*e4(id_)),
    )

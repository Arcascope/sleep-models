"""Wearanize+: the open-access raw release (``Wearanize+_OA_raw_v1.1``).

Labels are scorer 1's manual scores. The whole Empatica E4 stream of each night is read. The
manual synchronisation sheet places the PSG and the E4 on one axis, with time zero at lights-out:
a device's second ``s`` is at ``s - <device>_start_sec``. Device seconds start at 1, so the
first sample is at ``1 + 1 / rate``.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
from pisces_lite.datasets.data_set_object import DataSetObject

from arcasleep_watchtree.loaders import empatica
from arcasleep_watchtree.loaders.common import EPOCH_S, data_set, map_codes, psg_frame, sheet_records

NAME = "wearanize"
RELEASE_DIR = "Wearanize+_OA_raw_v1.1"
RECORDINGS_DIR = "1.Raw_data"
EMPATICA_DIR = "3.Empatica"
SCORES_DIR = "2.Sleep_scores/1.PSG_manual_scores_scorer1"
SYNC_XLSX = "3.Manual_synchronization/Manual_sync_zmax_psg_emp_actpal.xlsx"
SYNC_ID = "SubjectID"
PSG_START, PSG_END = "PSG_start_sec", "PSG_end_sec"
EMP_START, EMP_END = "Emp_start_sec", "Emp_end_sec"
#: Device second of a recording's start.
DEVICE_ORIGIN_S = 1.0
#: Sync value marking a recording that could not be synchronised.
NOT_SYNCHRONISED = -999
#: Scorer codes: -1 artifact, 0 wake, 1 N1, 2 N2, 3 N3, 4 REM.
STAGES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}


def night_id(raw_name: str) -> str:
    """``Sub005s1`` -> ``sub005_s1``."""
    match = re.fullmatch(r"sub(\d+)s(\d+)", raw_name.lower())
    return f"sub{match.group(1)}_s{match.group(2)}"


def synchronised(row: dict, start: str, end: str) -> bool:
    return not (row[start] == 0 and row[end] == 0) and row[start] != NOT_SYNCHRONISED


def read_scores(path: Path) -> np.ndarray:
    """First column of a scorer file as WatchTree codes; the header line is skipped."""
    lines = path.read_text().replace("\r", "").split("\n")[1:]
    raw = [int(float(line.replace("\t", ",").split(",")[0])) for line in lines if line.strip()]
    return map_codes(raw, STAGES)


def load(root: Path) -> DataSetObject:
    """``root`` holds the unzipped ``Wearanize+_OA_raw_v1.1`` folder."""
    release = Path(root) / RELEASE_DIR
    nights = {}
    for row in sheet_records(release / SYNC_XLSX):
        raw_name = row[SYNC_ID].strip("'")
        scores = release / SCORES_DIR / f"{raw_name}.txt"
        archives = sorted((release / RECORDINGS_DIR / raw_name / EMPATICA_DIR).glob("*.zip"))
        if (
            synchronised(row, PSG_START, PSG_END)
            and synchronised(row, EMP_START, EMP_END)
            and scores.exists()
            and archives
        ):
            nights[night_id(raw_name)] = (row, scores, archives[0])

    def e4(id_: str) -> tuple[empatica.E4, float]:
        row, _, archive = nights[id_]
        with zipfile.ZipFile(archive) as z:
            recording = empatica.read_e4(z.read)
        start = DEVICE_ORIGIN_S + 1 / recording.acc_rate - float(row[EMP_START])
        return recording, start

    def psg(id_: str):
        row, scores, _ = nights[id_]
        codes = read_scores(scores)
        return psg_frame(DEVICE_ORIGIN_S + EPOCH_S * np.arange(codes.size) - float(row[PSG_START]), codes)

    return data_set(
        NAME,
        Path(root),
        nights,
        accelerometer=lambda id_: empatica.accelerometer(*e4(id_)),
        psg=psg,
        heartrate=lambda id_: empatica.heart_rate(*e4(id_)),
    )

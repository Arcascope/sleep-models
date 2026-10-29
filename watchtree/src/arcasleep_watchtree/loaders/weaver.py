"""Weaver (revised): one CSV per child's night, holding every wrist device the child wore and the PSG.

Each device's recording is its own id, ``<id>_<device>_n<night>``, so one night of PSG appears
under several ids. All columns of a file share one clock. A device without a heart-rate column
has no heart rate.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
from pisces_lite.datasets.data_set_object import DataSetObject

from arcasleep_watchtree.loaders.common import (
    ACCEL_COLUMNS,
    HR_COLUMNS,
    data_set,
    map_codes,
    no_heart_rate,
    psg_frame,
)

NAME = "weaver_revised"
RAW_DIR = "raw"
DEVICES = ("apple", "fitbit", "actigraph", "garmin", "axivity", "pixel")
AXES = ("x", "y", "z")
TIME = "time"
HEART_RATE = "heartrate"
PSG = "psg"
STAGE = "stg"
PSG_FIELDS = (TIME, "epoch", "bp", STAGE)
NIGHT_PREFIX = "n"
#: ``L`` (lights on) and any other value are not scored.
STAGES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 4, "R": 5}


def night_id(child: str, device: str, night: str) -> str:
    return f"{child}_{device}_{NIGHT_PREFIX}{night}"


def header_map(path: Path) -> dict[str, str]:
    """Maps each column's simplified name to its name as written in the file.

    Names are compared lowercase with spaces and '#' removed, so "Actigraph X" and "actigraphx"
    are the same column. One file's PSG columns lack the "psg" prefix; they are given it here.
    """
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    names = {re.sub(r"[\s#]", "", h).lower(): h for h in header}
    if f"{PSG}{TIME}" not in names and all(field in names for field in PSG_FIELDS):
        for field in PSG_FIELDS:
            names[f"{PSG}{field}"] = names.pop(field)
    return names


def read_columns(path: Path, columns: dict[str, str], keys: list[str], names: list[str]) -> pd.DataFrame:
    """The rows of ``keys`` (simplified names) with no value missing, under ``names``."""
    wanted = [columns[k] for k in keys]
    frame = pd.read_csv(path, usecols=wanted)[wanted].dropna()
    frame.columns = names
    return frame


def read_psg(path: Path) -> pd.DataFrame:
    """One row per 30-s epoch; the file repeats each epoch's time and stage on every sample row."""
    columns = header_map(path)
    wanted = [columns[f"{PSG}{TIME}"], columns[f"{PSG}{STAGE}"]]
    frame = pd.read_csv(path, usecols=wanted, dtype={wanted[1]: str})[wanted]
    frame = frame.dropna(subset=[wanted[0]]).drop_duplicates(subset=[wanted[0]])
    return psg_frame(frame[wanted[0]].to_numpy(), map_codes(frame[wanted[1]], STAGES))


def load(root: Path) -> DataSetObject:
    root = Path(root)
    nights: dict[str, tuple[Path, str, dict[str, str]]] = {}
    for path in sorted((root / RAW_DIR).glob("*.csv")):
        columns = header_map(path)
        if f"{PSG}{TIME}" not in columns:
            continue
        child, night = path.stem.rsplit(f"_{NIGHT_PREFIX}", 1)
        for device in DEVICES:
            if all(f"{device}{c}" in columns for c in (TIME, *AXES)):
                nights[night_id(child, device, night)] = (path, device, columns)

    def accelerometer(id_: str) -> pd.DataFrame:
        path, device, columns = nights[id_]
        return read_columns(path, columns, [f"{device}{c}" for c in (TIME, *AXES)], ACCEL_COLUMNS)

    def heartrate(id_: str) -> pd.DataFrame:
        path, device, columns = nights[id_]
        keys = [f"{device}{TIME}", f"{device}{HEART_RATE}"]
        if keys[1] not in columns:
            return no_heart_rate()
        return read_columns(path, columns, keys, HR_COLUMNS)

    return data_set(
        NAME,
        root,
        nights,
        accelerometer=accelerometer,
        psg=lambda id_: read_psg(nights[id_][0]),
        heartrate=heartrate,
    )

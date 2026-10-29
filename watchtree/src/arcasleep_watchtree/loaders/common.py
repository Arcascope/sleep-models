"""What every WatchTree dataset loader returns, and the pieces they share.

A loader reads one original dataset download into a pisces-lite ``DataSetObject`` with three
features, each a frame per id:

- ``accelerometer``: ``timestamp, x, y, z`` in g, on the recorded timestamps;
- ``psg``: ``timestamp, psg``, one row per 30-s epoch, codes 0 wake, 1-3 N1-N3, 5 REM, -1 not scored;
- ``heartrate``: ``timestamp, heart_rate`` in bpm; empty when the recording has none.

All three are on one clock per recording.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.datasets.constants import PSG_COL, TIMESTAMP_COL, X_COL, Y_COL, Z_COL
from pisces_lite.datasets.data_set_object import DataSetObject

ACCELEROMETER = "accelerometer"
PSG = "psg"
HEARTRATE = "heartrate"
HR_COL = "heart_rate"
EPOCH_S = 30.0
NOT_SCORED = -1

ACCEL_COLUMNS = [TIMESTAMP_COL, X_COL, Y_COL, Z_COL]
PSG_COLUMNS = [TIMESTAMP_COL, PSG_COL]
HR_COLUMNS = [TIMESTAMP_COL, HR_COL]

Reader = Callable[[str], pd.DataFrame]

_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def accel_frame(t: np.ndarray, xyz: np.ndarray) -> pd.DataFrame:
    xyz = np.asarray(xyz, float)
    return pd.DataFrame(
        {TIMESTAMP_COL: np.asarray(t, float), X_COL: xyz[:, 0], Y_COL: xyz[:, 1], Z_COL: xyz[:, 2]}
    )


def psg_frame(t: np.ndarray, codes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({TIMESTAMP_COL: np.asarray(t, float), PSG_COL: np.asarray(codes, int)})


def heart_rate_frame(t: np.ndarray, bpm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({TIMESTAMP_COL: np.asarray(t, float), HR_COL: np.asarray(bpm, float)})


def no_heart_rate(_id: str = "") -> pd.DataFrame:
    return heart_rate_frame(np.empty(0), np.empty(0))


def map_codes(raw: Iterable, mapping: dict) -> np.ndarray:
    """``raw`` stage values as WatchTree codes; values not in ``mapping`` are not scored."""
    return np.array([mapping.get(v, NOT_SCORED) for v in raw], dtype=int)


def read_hr_out(path: Path) -> pd.DataFrame:
    """A whitespace-separated ``time bpm`` heart-rate file; empty when the file is missing."""
    if not path.exists():
        return no_heart_rate()
    data = np.loadtxt(path, ndmin=2)
    return heart_rate_frame(data[:, 0], data[:, 1])


def renamed(read: Reader, columns: list[str]) -> Reader:
    """``read`` with its frame's columns given the standard names."""

    def reader(id_: str) -> pd.DataFrame:
        frame = read(id_)
        if frame is not None:
            frame.columns = columns
        return frame

    return reader


def cleaned_files(root: Path) -> DataSetObject:
    """The ``cleaned_<feature>/`` files directly under ``root``."""
    found = [ds for ds in DataSetObject.find_data_sets(root).values() if ds.path == root]
    if len(found) != 1:
        raise ValueError(f"{root}: expected one cleaned_<feature>/ dataset, found {len(found)}")
    return found[0]


def data_set(
    name: str,
    root: Path,
    ids: Iterable[str],
    accelerometer: Reader,
    psg: Reader | None,
    heartrate: Reader,
) -> DataSetObject:
    """A ``DataSetObject`` whose features are read by the given per-id readers.

    ``psg`` is None for a dataset without PSG; ``heartrate`` is always registered.
    """
    ids = sorted(ids)
    ds = DataSetObject(name, root)
    ds.set_feature_loader(ACCELEROMETER, accelerometer, ids)
    if psg is not None:
        ds.set_feature_loader(PSG, psg, ids)
    ds.set_feature_loader(HEARTRATE, heartrate, ids)
    return ds


def _column_index(ref: str) -> int:
    index = 0
    for letter in re.match(r"[A-Z]+", ref).group(0):
        index = index * 26 + ord(letter) - 64
    return index - 1


def read_first_sheet(path: Path) -> list[list]:
    """First sheet of an ``.xlsx`` file as rows of strings and floats."""
    ns = {"m": _XLSX_NS}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{_XLSX_NS}}}t")))
        sheet = sorted(n for n in archive.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        root = ET.fromstring(archive.read(sheet))
    rows = []
    for row in root.iter(f"{{{_XLSX_NS}}}row"):
        cells: dict[int, object] = {}
        for cell in row.findall("m:c", ns):
            value = cell.find("m:v", ns)
            kind = cell.get("t")
            if kind == "inlineStr":
                parsed = "".join(t.text or "" for t in cell.iter(f"{{{_XLSX_NS}}}t"))
            elif value is None or value.text is None:
                continue
            elif kind == "s":
                parsed = shared[int(value.text)]
            elif kind in ("str", "e", "b"):
                parsed = value.text
            else:
                parsed = float(value.text)
            cells[_column_index(cell.get("r"))] = parsed
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i) for i in range(width)])
    return rows


def sheet_records(path: Path) -> list[dict]:
    """The first sheet's rows as dicts keyed by its header row."""
    rows = read_first_sheet(path)
    return [dict(zip(rows[0], values)) for values in rows[1:]]

"""Wearanize+ and EESM19 with the whole wrist stream, for uncropped caches.

The converted layout of these cohorts (``cleaned_accelerometer/`` and
``cleaned_psg/``, time zero at the first scored epoch or at lights-out) keeps
only the accelerometer samples inside the scored window. A dataset directory
for an uncropped cache holds two files:

    pisces_lite_adapter.py   ``from arcasleep.data.cohorts import load_data_sets``
    uncropped_source.json    ``{"cohort": "wearanize" | "eesm19",
                               "raw_root": <raw dataset>,
                               "converted_dir": <converted dataset>}``

The PSG comes from the converted files, so the labels are the converted
dataset's. The accelerometer is read from the raw recording and placed on the
converted time axis, rounded to the precision those files store, so samples
inside the scored window equal the converted ones. Samples before and after it
are kept. Only recordings the converted dataset accepted are listed.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pisces_lite.datasets import (
    PSG_COL,
    TIMESTAMP_COL,
    X_COL,
    Y_COL,
    Z_COL,
    DataSetConfig,
    DataSetObject,
)

SOURCE_FILENAME = "uncropped_source.json"
PSG_SUFFIX = "_cleaned_psg.out"

E4_HZ = 32.0
E4_UNITS_PER_G = 64.0
SYNC_XLSX = "3.Manual_synchronization/Manual_sync_zmax_psg_emp_actpal.xlsx"
GENEACTIV_HZ = 100.0
# Decimals of the converted accelerometer files' time and acceleration columns.
WEARANIZE_DECIMALS = (6, 6)
EESM19_DECIMALS = (3, 4)

_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index(ref: str) -> int:
    index = 0
    for letter in re.match(r"[A-Z]+", ref).group(0):
        index = index * 26 + ord(letter) - 64
    return index - 1


def read_first_sheet(path: Path | str) -> list[list]:
    """First sheet of an ``.xlsx`` file as rows of strings and floats."""
    ns = _XLSX_NS["m"]
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", _XLSX_NS):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{ns}}}t")))
        sheet = sorted(n for n in archive.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        root = ET.fromstring(archive.read(sheet))
    rows = []
    for row in root.iter(f"{{{ns}}}row"):
        cells: dict[int, object] = {}
        for cell in row.findall("m:c", _XLSX_NS):
            value = cell.find("m:v", _XLSX_NS)
            kind = cell.get("t")
            if kind == "inlineStr":
                parsed = "".join(t.text or "" for t in cell.iter(f"{{{ns}}}t"))
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


def wearanize_subject_id(raw_name: str) -> str:
    """``Sub001s1`` -> ``sub001_s1``."""
    match = re.fullmatch(r"sub(\d+)s(\d+)", raw_name.strip("'").lower())
    if match is None:
        raise ValueError(f"unrecognised subject name {raw_name!r}")
    return f"sub{match.group(1)}_s{match.group(2)}"


def _accel_frame(
    times: np.ndarray, acc: np.ndarray, decimals: tuple[int, int]
) -> pd.DataFrame:
    time_decimals, acc_decimals = decimals
    return pd.DataFrame(
        {
            TIMESTAMP_COL: np.round(times, time_decimals),
            X_COL: np.round(acc[:, 0], acc_decimals),
            Y_COL: np.round(acc[:, 1], acc_decimals),
            Z_COL: np.round(acc[:, 2], acc_decimals),
        }
    )


def _read_e4_acc(archive: Path) -> np.ndarray:
    """``(n, 3)`` Empatica ACC in 1/64 g from an Empatica zip, after checking the rate."""
    with zipfile.ZipFile(archive) as z:
        text = z.read("ACC.csv")
    rate = float(text.split(b"\n", 2)[1].split(b",")[0])
    if rate != E4_HZ:
        raise ValueError(f"E4 ACC rate {rate} Hz, expected {E4_HZ}")
    return pd.read_csv(
        io.BytesIO(text), skiprows=2, header=None, dtype=np.float32
    ).to_numpy()


def _wearanize_loader(converted: Path, raw_root: Path) -> tuple[list[str], Callable]:
    rows = read_first_sheet(raw_root / SYNC_XLSX)
    header = rows[0]
    sync = {}
    for values in rows[1:]:
        row = dict(zip(header, values))
        sync[wearanize_subject_id(row["SubjectID"])] = row

    def load(subject_id: str) -> pd.DataFrame:
        row = sync[subject_id]
        raw_name = row["SubjectID"].strip("'")
        archives = sorted((raw_root / "1.Raw_data" / raw_name / "3.Empatica").glob("*.zip"))
        acc = _read_e4_acc(archives[0])
        # Sample i is recorded at device time 1 + (i + 1) / fs; time zero is
        # the lights-out moment on the Empatica clock.
        times = 1.0 + (np.arange(acc.shape[0]) + 1.0) / E4_HZ - float(row["Emp_start_sec"])
        return _accel_frame(times, acc.astype(np.float64) / E4_UNITS_PER_G, WEARANIZE_DECIMALS)

    return list(sync), load


def _eesm19_loader(converted: Path, raw_root: Path) -> tuple[list[str], Callable]:
    manifest = {
        entry["id"]: entry
        for entry in json.loads((converted / "conversion_manifest.json").read_text())
        if entry["status"] == "converted"
    }

    def load(subject_id: str) -> pd.DataFrame:
        entry = manifest[subject_id]
        sub, ses = entry["raw"].split("/")
        scores = pd.read_csv(
            raw_root / sub / ses / "eeg" / f"{sub}_{ses}_task-sleep_acq-scoring1_events.tsv",
            sep="\t",
        )
        zero = float(scores["onset"].iloc[0])
        path = sorted((raw_root / "sourcedata" / sub / ses / "actigraphy").glob("*.csv"))[0]
        with open(path, errors="replace") as f:
            for skip, line in enumerate(f):
                if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d:\d{3},", line):
                    break
        acc = pd.read_csv(
            path, skiprows=skip, header=None, usecols=[1, 2, 3], dtype=np.float32
        ).to_numpy()
        # The recording is on a 100 Hz clock; its alignment offset is the
        # watch time of EDF time zero.
        times = (
            np.arange(acc.shape[0]) / GENEACTIV_HZ
            - float(entry["alignment"]["offset_s"])
            - zero
        )
        return _accel_frame(times, acc.astype(np.float64), EESM19_DECIMALS)

    return list(manifest), load


_COHORTS = {"wearanize": _wearanize_loader, "eesm19": _eesm19_loader}


def _load_psg(root: Path, subject_id: str) -> pd.DataFrame:
    return pd.read_csv(
        root / "cleaned_psg" / f"{subject_id}{PSG_SUFFIX}",
        sep=" ",
        header=None,
        names=[TIMESTAMP_COL, PSG_COL],
    )


def load_data_sets(root: str | Path) -> dict[str, DataSetObject]:
    """The cohort named in ``root``'s :data:`SOURCE_FILENAME`, in adapter form."""
    root = Path(root)
    source = json.loads((root / SOURCE_FILENAME).read_text())
    cohort = source["cohort"]
    converted = Path(source["converted_dir"])
    ids, load_accel = _COHORTS[cohort](converted, Path(source["raw_root"]))
    on_disk = {p.name[: -len(PSG_SUFFIX)] for p in (converted / "cleaned_psg").glob(f"*{PSG_SUFFIX}")}
    ids = sorted(set(ids) & on_disk)
    data_set = DataSetObject(
        cohort,
        root,
        config=DataSetConfig.from_dict(
            {"name": cohort, "psg": {"dt_seconds": 30.0}, "timestamp": {"unit": "s"}}
        ),
    )
    data_set.set_feature_loader("accelerometer", load_accel, ids=ids)
    data_set.set_feature_loader("psg", lambda sid: _load_psg(converted, sid), ids=ids)
    return {cohort: data_set}

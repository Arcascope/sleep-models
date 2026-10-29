"""Wearanize+ nights from the whole wrist stream, not only the PSG span.

The converted dataset keeps accelerometer samples inside the scored window. Here the raw
Empatica E4 stream is read and placed on the converted time axis (time zero at lights-out, as
in the converted files, from the manual synchronisation sheet), so epochs before and after the
PSG span have real data and windows are warm at the first scored epoch and full at the last.

Epochs outside the PSG span carry label ``OUTSIDE_PSG`` (-3); epochs inside it carry the
converted PSG's labels, masked (-2) where the accelerometer has too few samples.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.datasets import align_trim, mask_data
from pisces_lite.datasets.constants import PSG_COL, TIMESTAMP_COL, X_COL, Y_COL, Z_COL

from arcasleep.processing import clean_accelerometer_rows
from arcasleep_watchtree.data.loaders import DATA_ROOT, Night, wearanize_hr
from arcasleep_watchtree.features.preprocess import to_25hz

OUTSIDE_PSG = -3
PSG_DT = 30
E4_HZ = 32.0
E4_UNITS_PER_G = 64.0
RAW_ROOT = "Wearanize/Wearanize+_OA_raw_v1.1"
SYNC_XLSX = "3.Manual_synchronization/Manual_sync_zmax_psg_emp_actpal.xlsx"
PSG_SUFFIX = "_cleaned_psg.out"
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _column_index(ref: str) -> int:
    index = 0
    for letter in re.match(r"[A-Z]+", ref).group(0):
        index = index * 26 + ord(letter) - 64
    return index - 1


def read_first_sheet(path: Path) -> list[list]:
    """First sheet of an ``.xlsx`` file as rows of strings and floats."""
    ns = {"m": _NS}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{_NS}}}t")))
        sheet = sorted(n for n in archive.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        root = ET.fromstring(archive.read(sheet))
    rows = []
    for row in root.iter(f"{{{_NS}}}row"):
        cells: dict[int, object] = {}
        for cell in row.findall("m:c", ns):
            value = cell.find("m:v", ns)
            kind = cell.get("t")
            if kind == "inlineStr":
                parsed = "".join(t.text or "" for t in cell.iter(f"{{{_NS}}}t"))
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


def _subject_id(raw_name: str) -> str:
    match = re.fullmatch(r"sub(\d+)s(\d+)", raw_name.strip("'").lower())
    return f"sub{match.group(1)}_s{match.group(2)}"


def sync_table(raw_root: Path) -> dict[str, dict]:
    rows = read_first_sheet(raw_root / SYNC_XLSX)
    header = rows[0]
    table = {}
    for values in rows[1:]:
        row = dict(zip(header, values))
        table[_subject_id(row["SubjectID"])] = row
    return table


def wearanize_stream(subject: str, sync: dict[str, dict], raw_root: Path) -> pd.DataFrame:
    """The whole E4 accelerometer stream of one night on the converted time axis, in g."""
    row = sync[subject]
    raw_name = row["SubjectID"].strip("'")
    archive = sorted((raw_root / "1.Raw_data" / raw_name / "3.Empatica").glob("*.zip"))[0]
    with zipfile.ZipFile(archive) as z:
        text = z.read("ACC.csv")
    rate = float(text.split(b"\n", 2)[1].split(b",")[0])
    if rate != E4_HZ:
        raise ValueError(f"E4 ACC rate {rate} Hz, expected {E4_HZ}")
    acc = pd.read_csv(io.BytesIO(text), skiprows=2, header=None, dtype=np.float32).to_numpy()
    acc = acc.astype(np.float64) / E4_UNITS_PER_G
    times = 1.0 + (np.arange(acc.shape[0]) + 1.0) / E4_HZ - float(row["Emp_start_sec"])
    return pd.DataFrame(
        {
            TIMESTAMP_COL: np.round(times, 6),
            X_COL: np.round(acc[:, 0], 6),
            Y_COL: np.round(acc[:, 1], 6),
            Z_COL: np.round(acc[:, 2], 6),
        }
    )


def align_uncropped(
    accel: pd.DataFrame, psg: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Trim ``accel`` to whole epochs on the PSG's epoch phase; label every epoch.

    Returns the trimmed stream, the labels (``OUTSIDE_PSG`` outside the span that
    ``align_trim`` keeps) and the time of epoch 0.
    """
    accel = accel.sort_values(by=TIMESTAMP_COL)
    _, inside = align_trim(accel, psg)
    if inside.empty:
        raise ValueError("accelerometer and PSG do not overlap")
    phase = float(psg[TIMESTAMP_COL].min()) % PSG_DT
    first, last = accel[TIMESTAMP_COL].iloc[0], accel[TIMESTAMP_COL].iloc[-1]
    start = phase + PSG_DT * np.ceil((first - phase) / PSG_DT)
    end = phase + PSG_DT * np.floor((last - phase) / PSG_DT)
    accel = accel[(accel[TIMESTAMP_COL] >= start) & (accel[TIMESTAMP_COL] <= end)]
    n_epochs = int(round((end - start) / PSG_DT)) + 1
    from_ = int(round((inside[TIMESTAMP_COL].iloc[0] - start) / PSG_DT))
    labels = np.full(n_epochs, OUTSIDE_PSG, dtype=int)
    masked = mask_data(accel, inside)
    labels[from_ : from_ + len(inside)] = masked[PSG_COL].to_numpy()
    return accel, labels, float(start)


def load_night_uncropped(subject: str, reported: str, sync: dict, converted: Path) -> Night:
    raw_root = DATA_ROOT / RAW_ROOT
    psg = pd.read_csv(
        converted / "cleaned_psg" / f"{subject}{PSG_SUFFIX}",
        sep=" ",
        header=None,
        names=[TIMESTAMP_COL, PSG_COL],
    )
    stream = wearanize_stream(subject, sync, raw_root)
    cleaned = pd.DataFrame(
        clean_accelerometer_rows(stream.to_numpy()), columns=stream.columns
    )
    accel, labels, t0 = align_uncropped(cleaned, psg)
    rows = accel.to_numpy(dtype=np.float64)
    acc25, valid25 = to_25hz(rows[:, 0], rows[:, 1:4], t0, labels.size)
    t, bpm = wearanize_hr(subject, rows)
    return Night("wearanize", reported, acc25, valid25, t - t0, bpm, labels, t0)

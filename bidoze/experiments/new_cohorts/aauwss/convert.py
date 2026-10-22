"""Convert the AAUWSS cohort into the arcasleep ``cleaned_*`` layout.

Usage: ``python convert.py <AAUWSS raw dir> <output dir>``

Writes ``cleaned_accelerometer/<id>_cleaned_motion.out``,
``cleaned_psg/<id>_cleaned_psg.out`` and ``data_set.json`` for each
subject, with ids ``s01`` .. ``s13``.

Alignment: the E4 ``ACC.csv`` start is a UTC unix time; the annotation
workbook and the EDF header carry Europe/Copenhagen wall-clock time. The
E4 start is converted to Copenhagen local time and no other offset is
applied. The authors' ``aligned_sleep_data_set`` pickles use exactly this
mapping (checked by ``check_alignment.py``).

Output times are seconds from the first scored epoch. Accelerometer
values are in g (E4 counts / 64). PSG codes: 0 Wake, 1 N1, 2 N2, 3 N3,
5 REM, -1 for epochs the annotation marks "Missing" or omits (the
PSG mask value).
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = datetime(1899, 12, 30)
LOCAL_TZ = ZoneInfo("Europe/Copenhagen")
MASK = -1
STAGE_CODES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 5, "Missing": MASK}
EPOCH_SECONDS = 30
E4_COUNTS_PER_G = 64.0

DATA_SET_JSON = {
    "name": "aauwss",
    "csv": {"delimiter": " "},
    "metadata": {
        "note": (
            "Space-delimited .out files (accel + psg). PSG stages use the 0-5 "
            "convention (0=Wake, 5=REM), so no mapping is applied."
        ),
    },
}


def read_annotations(path: Path) -> pd.DataFrame:
    """Return ``start`` (naive local datetime) and ``stage`` per scored epoch."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = [
                "".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t"))
                for si in ET.fromstring(z.read("xl/sharedStrings.xml"))
            ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{{{NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find("m:v", NS)
            if c.get("t") == "inlineStr":
                cells[col] = "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
            elif v is None:
                continue
            elif c.get("t") == "s":
                cells[col] = strings[int(v.text)]
            else:
                cells[col] = float(v.text)
        rows.append(cells)
    header, body = rows[0], rows[1:]
    assert header["B"] == "Event Start Time" and header["C"] == "Sleep Stage", header
    out = pd.DataFrame(
        {
            "start": [
                EXCEL_EPOCH + timedelta(days=r["B"]) for r in body
            ],
            "stage": [r["C"] for r in body],
        }
    )
    out["start"] = out["start"].dt.round("1s")
    return out


def read_e4_acc(path: Path) -> tuple[float, float, np.ndarray]:
    """Return start unix time, rate in Hz and an ``(n, 3)`` array in g."""
    raw = pd.read_csv(path, header=None).to_numpy(float)
    return raw[0, 0], raw[1, 0], raw[2:] / E4_COUNTS_PER_G


def local_naive(unix_utc: float) -> datetime:
    return datetime.fromtimestamp(unix_utc, LOCAL_TZ).replace(tzinfo=None)


def convert_subject(raw: Path, number: int, out: Path) -> dict:
    sid = f"s{number:02d}"
    ann = read_annotations(raw / "annotations" / f"subject_{number:02d}_manual_annotation.xlsx")
    unknown = set(ann["stage"]) - set(STAGE_CODES)
    if unknown:
        raise ValueError(f"{sid}: unmapped stages {unknown}")

    t0 = ann["start"].iloc[0]
    index = ((ann["start"] - t0).dt.total_seconds() / EPOCH_SECONDS).round().astype(int)
    if not index.is_unique or (index.diff().dropna() < 1).any():
        raise ValueError(f"{sid}: scored epochs are not strictly increasing on a 30 s grid")
    n_epochs = int(index.iloc[-1]) + 1
    codes = np.full(n_epochs, MASK, dtype=int)
    codes[index.to_numpy()] = ann["stage"].map(STAGE_CODES).to_numpy()
    scored_seconds = n_epochs * EPOCH_SECONDS

    start_unix, rate, acc = read_e4_acc(raw / "empatica" / f"subject_{number:02d}" / "ACC.csv")
    e4_start_local = local_naive(start_unix)
    offset = (e4_start_local - t0).total_seconds()
    t = np.arange(len(acc)) / rate + offset
    keep = (t >= 0) & (t < scored_seconds)
    covered = keep.sum() / rate

    acc_out = np.column_stack([t[keep], acc[keep]])
    np.savetxt(
        out / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out",
        acc_out,
        fmt="%.6f",
        delimiter=" ",
    )
    psg_out = np.column_stack(
        [np.arange(n_epochs) * EPOCH_SECONDS, codes]
    )
    np.savetxt(
        out / "cleaned_psg" / f"{sid}_cleaned_psg.out",
        psg_out,
        fmt="%.6f",
        delimiter=" ",
    )
    return {
        "id": sid,
        "scored_start": str(t0),
        "e4_start_local": str(e4_start_local),
        "e4_start_minus_scored_start_s": offset,
        "scored_hours": scored_seconds / 3600,
        "accel_coverage": covered / scored_seconds,
        "stages": {
            name: int((codes == code).sum()) for name, code in STAGE_CODES.items()
        },
    }


def main(raw: Path, out: Path) -> None:
    (out / "cleaned_accelerometer").mkdir(parents=True, exist_ok=True)
    (out / "cleaned_psg").mkdir(parents=True, exist_ok=True)
    summary = [convert_subject(raw, n, out) for n in range(1, 14)]
    (out / "data_set.json").write_text(json.dumps(DATA_SET_JSON, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

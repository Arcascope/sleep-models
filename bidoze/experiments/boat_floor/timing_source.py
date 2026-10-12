"""Somerset timing audit from source files.

For each cleaned night: the raw hypnogram it copies, the anchor encoded in the
filename, and the offset between the cleaned accelerometer clock and the raw
accelerometer unix clock (found by matching sample values, cleaned = raw / 9.8).
The PSG "Start Recording" times are read from the Somfit metrics PDFs and passed
in via START_RECORDING (transcribed from page 1 of each sleepinterval*_metricsref.pdf).
"""

from __future__ import annotations

import csv
import glob
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/media/eric/Extreme SSD/pisces2-data/Somerset")
OUT = Path(sys.argv[1])

# (participant, interval) -> (study date, Start Recording, Stop Recording), from the metrics PDFs.
START_RECORDING = {
    ("601", 2): ("2025-06-13", "15:42:03", "20:59:03"),
    ("601", 4): ("2025-06-23", "15:58:22", "15:58:22"),
    ("604", 2): ("2025-06-13", "14:45:51", "21:15:15"),
    ("604", 3): ("2025-06-14", "13:45:47", "20:32:26"),
    ("607", 1): ("2025-06-12", "15:37:10", "20:33:37"),
    ("607", 2): ("2025-06-14", "15:24:22", "21:28:15"),
    ("609", 1): ("2025-06-12", "14:17:00", "20:57:02"),
    ("609", 2): ("2025-06-13", "14:13:49", "21:00:35"),
    ("610", 1): ("2025-06-12", "12:36:36", "20:28:25"),
    ("610", 3): ("2025-06-15", "14:07:17", "20:21:18"),
    ("611", 1): ("2025-06-12", "12:52:39", "12:52:39"),
    ("611", 3): ("2025-06-24", "14:06:45", "21:32:28"),
    ("612", 2): ("2025-06-13", "13:24:51", "20:55:58"),
    ("612", 3): ("2025-06-14", "13:30:17", "21:17:07"),
    ("612", 5): ("2025-06-17", "13:12:00", "20:16:10"),
    ("613", 1): ("2025-06-12", "12:45:34", "20:00:59"),
    ("613", 2): ("2025-06-13", "12:35:46", "20:19:43"),
    ("613", 3): ("2025-06-14", "12:58:18", "20:20:32"),
}

STAGE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "U": -1}


def hypnograms(pid: str) -> dict[int, list[int]]:
    out = {}
    for f in sorted(glob.glob(str(ROOT / "psg" / pid / "sleepinterval*_hypnogram.txt"))):
        k = int(re.search(r"sleepinterval(\d+)_", f).group(1))
        out[k] = [STAGE[x] for x in re.findall(r"REM|N1|N2|N3|W|U", open(f).read())]
    return out


def accel_offset(pid: str, anchor: int, cleaned: pd.DataFrame) -> tuple[float, int]:
    """Return (raw_unix_s - (anchor + cleaned_t)) for matched samples, and the match count."""
    first = cleaned.iloc[:200]
    target = np.round(first[["ACC_X", "ACC_Y", "ACC_Z"]].values * 9.8, 3)
    deltas = []
    for f in sorted(glob.glob(str(ROOT / "accel" / pid / "*session_1*.csv"))):
        raw = pd.read_csv(f)
        if raw.empty:
            continue
        key = np.round(raw[["x", "y", "z"]].values, 3)
        lookup = {}
        for i, row in enumerate(map(tuple, key)):
            lookup.setdefault(row, []).append(i)
        for j, row in enumerate(map(tuple, target)):
            for i in lookup.get(row, [])[:3]:
                deltas.append(raw.timestamp.values[i] / 1000 - (anchor + first.TIMESTAMP.values[j]))
    if not deltas:
        return float("nan"), 0
    d = np.round(np.array(deltas), 3)
    vals, counts = np.unique(d, return_counts=True)
    return float(vals[np.argmax(counts)]), int(counts.max())


def main() -> None:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "cleaned_psg" / "*.csv"))):
        name = Path(f).stem
        pid, anchor = name.split("_")
        anchor = int(anchor)
        labels = [int(r[1]) for r in list(csv.reader(open(f)))[1:]]
        hyp = hypnograms(pid)
        match = [k for k, s in hyp.items() if s == labels]
        interval = match[0] if len(match) == 1 else None
        date, start, stop = START_RECORDING[(pid, interval)]
        start_naive = datetime.fromisoformat(f"{date}T{start}")
        anchor_utc = datetime.fromtimestamp(anchor, timezone.utc).replace(tzinfo=None)
        # Interpret the anchor in the UTC-4 wall clock that reproduces the PDF dates/minutes.
        anchor_wall = anchor_utc - timedelta(hours=4)
        psg_shift = (start_naive - anchor_wall).total_seconds()
        acc = pd.read_csv(ROOT / "cleaned_accelerometer" / f"{name}.csv")
        off, nmatch = accel_offset(pid, anchor, acc)
        rows.append(
            dict(
                night=f"Somerset_{name}",
                participant=pid,
                raw_interval=f"sleepinterval{interval}",
                labels_identical_to_raw=interval is not None,
                n_epochs=len(labels),
                psg_start_recording_wall=f"{date} {start}",
                anchor_unix=anchor,
                anchor_wall_utc_minus_4=anchor_wall.isoformat(sep=" "),
                cleaned_psg_first_t=0.0,
                cleaned_accel_first_t=float(acc.TIMESTAMP.iloc[0]),
                cleaned_accel_last_t=float(acc.TIMESTAMP.iloc[-1]),
                psg_start_minus_anchor_s=psg_shift,
                raw_accel_unix_minus_cleaned_clock_s=off,
                accel_value_matches=nmatch,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "somerset_source_timing.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()

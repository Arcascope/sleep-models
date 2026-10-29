"""Read PSG cohorts into nights on the shared epoch grid, with heart rate.

The accelerometer and the 30-s PSG epochs go through the same steps as the BiDoze feature cache
(:func:`arcasleep.data.prepare.prepare_subject`): rows cleaned, both streams trimmed to their
overlap on the PSG's epoch grid, epochs without enough accelerometer samples masked. Heart rate
is read per cohort and returned on the accelerometer's clock.

Labels are the cache's raw 5-class codes; :func:`four_class` collapses them to wake, light
(N1, N2), deep (N3, N4) and REM, with negative codes (unscored, masked) staying negative.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.datasets import align_trim, load_subject, mask_data
from pisces_lite.datasets.constants import PSG_COL, TIMESTAMP_COL

from arcasleep.data.prepare import load_dataset
from arcasleep.data.pseudonyms import cache_ids, load_pseudonym_key, pseudonym
from arcasleep.processing import clean_accelerometer_rows
from arcasleep_watchtree.features.preprocess import to_25hz

DATA_ROOT = Path(os.environ.get("WATCHTREE_DATA_ROOT"))
E4_COUNTS_PER_G = 64.0

HeartRate = tuple[np.ndarray, np.ndarray]


@dataclass
class Night:
    cohort: str
    subject: str
    acc25: np.ndarray
    valid25: np.ndarray
    hr_t: np.ndarray
    hr_bpm: np.ndarray
    #: Raw 5-class codes per epoch: 0 wake, 1-4 N1-N4, 5 REM, negative = not scored.
    labels: np.ndarray
    #: Absolute time of epoch 0 on the dataset's own clock.
    t0: float


def four_class(labels: np.ndarray) -> np.ndarray:
    """0 wake, 1 light (N1, N2), 2 deep (N3, N4), 3 REM; -1 where the epoch is not scored."""
    table = np.array([0, 1, 1, 2, 2, 3])
    labels = np.asarray(labels)
    scored = (labels >= 0) & (labels <= 5)
    return np.where(scored, table[np.clip(labels, 0, 5).astype(int)], -1)


def _read_hr_out(path: Path) -> HeartRate:
    """A two-column ``time bpm`` text file on the accelerometer's clock."""
    if not path.exists():
        return np.empty(0), np.empty(0)
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0], data[:, 1]


def _empatica_hr(acc_zip_or_dir, converted_acc: np.ndarray) -> HeartRate:
    """Heart rate of an Empatica E4 recording on the clock of a converted accelerometer file.

    ``converted_acc`` is the converted file's ``(t, x, y, z)`` rows. Its first rows are matched
    to the raw ``ACC.csv`` counts to find which raw sample they are, which fixes the offset
    between the raw files' clock and the converted one; ``HR.csv`` is placed by its own start time.
    """
    if acc_zip_or_dir is None:
        return np.empty(0), np.empty(0)
    acc_raw, hr_raw = acc_zip_or_dir
    head = np.rint(converted_acc[:32, 1:4] * E4_COUNTS_PER_G)
    candidates = np.flatnonzero(
        (acc_raw[2:, 0] == head[0, 0])
        & (acc_raw[2:, 1] == head[0, 1])
        & (acc_raw[2:, 2] == head[0, 2])
    )
    first = next(
        (int(i) for i in candidates if np.array_equal(acc_raw[2 + i : 2 + i + 32], head)), None
    )
    if first is None:
        return np.empty(0), np.empty(0)
    rate = float(acc_raw[1, 0])
    start_of_raw = converted_acc[0, 0] - first / rate
    hr_rate = float(hr_raw[1])
    values = hr_raw[2:]
    return start_of_raw + (hr_raw[0] - acc_raw[0, 0]) + np.arange(values.size) / hr_rate, values


def _read_e4_pair(read_acc: Callable[[], bytes], read_hr: Callable[[], bytes]):
    acc = pd.read_csv(io.BytesIO(read_acc()), header=None).to_numpy(float)
    hr = pd.read_csv(io.BytesIO(read_hr()), header=None).to_numpy(float)[:, 0]
    return acc, hr


def wearanize_hr(subject: str, converted_acc: np.ndarray) -> HeartRate:
    match = re.fullmatch(r"sub(\d+)_s(\d+)", subject)
    folder = DATA_ROOT / "Wearanize/Wearanize+_OA_raw_v1.1/1.Raw_data" / f"Sub{match[1]}s{match[2]}"
    archives = sorted((folder / "3.Empatica").glob("*.zip"))
    if not archives:
        return np.empty(0), np.empty(0)
    with zipfile.ZipFile(archives[0]) as z:
        pair = _read_e4_pair(lambda: z.read("ACC.csv"), lambda: z.read("HR.csv"))
    return _empatica_hr(pair, converted_acc)


def aauwss_hr(subject: str, converted_acc: np.ndarray) -> HeartRate:
    folder = DATA_ROOT / "AAUWSS/empatica" / f"subject_{subject[1:]}"
    pair = _read_e4_pair(
        lambda: (folder / "ACC.csv").read_bytes(), lambda: (folder / "HR.csv").read_bytes()
    )
    return _empatica_hr(pair, converted_acc)


def weaver_hr(subject: str, device: str) -> HeartRate:
    """Heart rate of one device in a Weaver night file, on that device's own clock."""
    child, night = subject.split(f"_{device}_n")
    path = DATA_ROOT / "weaver-2026-10-revised/raw" / f"{child}_n{night}.csv"
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    names = {re.sub(r"[\s#]", "", h).lower(): h for h in header}
    cols = [names[f"{device}time"], names[f"{device}heartrate"]]
    frame = pd.read_csv(path, usecols=cols)[cols].dropna()
    return frame.iloc[:, 0].to_numpy(float), frame.iloc[:, 1].to_numpy(float)


@dataclass(frozen=True)
class Cohort:
    name: str
    directory: str
    #: Keeps a subset of the dataset's subject ids.
    select: Callable[[str], bool] = lambda subject: True
    #: ``(subject, converted_accel_rows) -> (t, bpm)``; None when the cohort has no heart rate.
    heart_rate: Callable[[str, np.ndarray], HeartRate] | None = None


COHORTS: dict[str, Cohort] = {
    "wearanize": Cohort("wearanize", "wearanize-pisces", heart_rate=wearanize_hr),
    "aauwss": Cohort("aauwss", "aauwss-pisces", heart_rate=aauwss_hr),
    "sleepaccel": Cohort(
        "sleepaccel", "SleepAccel",
        heart_rate=lambda s, _: _read_hr_out(
            DATA_ROOT / "SleepAccel/cleaned_heartrate" / f"{s}_cleaned_hr.out"
        ),
    ),
    "weaver_garmin": Cohort(
        "weaver_garmin", "weaver-2026-10-revised",
        select=lambda s: "_garmin_n" in s,
        heart_rate=lambda s, _: weaver_hr(s, "garmin"),
    ),
    "weaver_fitbit": Cohort(
        "weaver_fitbit", "weaver-2026-10-revised",
        select=lambda s: "_fitbit_n" in s,
        heart_rate=lambda s, _: weaver_hr(s, "fitbit"),
    ),
}  # fmt: skip


def subject_ids(cohort: Cohort) -> tuple[list[str], dict[str, str]]:
    """Source ids of the cohort's subjects and the ids they are reported under."""
    directory = DATA_ROOT / cohort.directory
    ids = [s for s in load_dataset(directory).ids if cohort.select(s)]
    return ids, cache_ids(load_pseudonym_key(directory), ids)


def participant_of(cohort: str, subject: str) -> str:
    """The participant a source night id belongs to: all of one person's nights share this."""
    if cohort.startswith("weaver"):
        return subject.split("_")[0]
    match = re.fullmatch(r"(sub\d+)_s\d+", subject)
    return match[1] if match else subject


def person_ids(cohort: Cohort, ids: list[str]) -> dict[str, str]:
    """Each source night id's participant id, pseudonymized with the dataset key if it has one.

    The participant is taken from the source id before hashing, so a keyed dataset's nights of
    one person still share a participant id.
    """
    key = load_pseudonym_key(DATA_ROOT / cohort.directory)
    people = {s: participant_of(cohort.name, s) for s in ids}
    if key is None:
        return people
    return {s: pseudonym(key, p) for s, p in people.items()}


def load_night(cohort: Cohort, subject: str, reported: str | None = None, data_set=None) -> Night:
    """One night on the 25 Hz grid, with its labels and heart rate."""
    data_set = data_set or load_dataset(DATA_ROOT / cohort.directory)
    loaded = load_subject(data_set, subject)
    cleaned = pd.DataFrame(
        clean_accelerometer_rows(loaded.accel_df.to_numpy()), columns=loaded.accel_df.columns
    )
    accel, psg = align_trim(cleaned, loaded.psg_df)
    psg = mask_data(accel, psg)
    t0 = float(psg[TIMESTAMP_COL].iloc[0])
    labels = psg[PSG_COL].to_numpy()
    rows = accel.to_numpy(dtype=np.float64)
    acc25, valid25 = to_25hz(rows[:, 0], rows[:, 1:4], t0, labels.size)
    t, bpm = (np.empty(0), np.empty(0))
    if cohort.heart_rate is not None:
        t, bpm = cohort.heart_rate(subject, rows)
    for frames in getattr(data_set, "_feature_cache", {}).values():
        frames.pop(subject, None)
    return Night(cohort.name, reported or subject, acc25, valid25, t - t0, bpm, labels, t0)

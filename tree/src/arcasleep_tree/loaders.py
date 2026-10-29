"""Read original cohort downloads and write their feature tables as parquet.

A loader reads one dataset download into a pisces-lite ``DataSetObject`` with three features,
each a frame per id, all on one clock per recording:

- ``accelerometer``: ``timestamp, x, y, z`` in g, on the recorded timestamps;
- ``psg``: ``timestamp, psg``, one row per 30-s epoch, in pisces-lite's 5-class codes (0 wake,
  1-4 N1-N4, 5 REM, ``PSG_MASK`` not scored);
- ``heartrate``: ``timestamp, heart_rate`` in bpm; empty when the recording has none.

:func:`build` turns one dataset into ``features-<dataset>.parquet``. Each row is one 30-s epoch of
one recording: ``cohort``, ``subject``, ``person``, ``y4`` (0 wake, 1 light, 2 deep, 3 REM,
``PSG_MASK`` not scored), ``in_psg``, ``epoch`` and the feature columns. With ``uncropped`` the
whole wrist recording is kept and epochs outside the PSG span have ``in_psg == False``; otherwise
both streams are trimmed to their overlap.
"""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.datasets import align_trim, mask_data
from pisces_lite.datasets.constants import (
    PSG_COL,
    PSG_MAPPING_WEAVER,
    PSG_MASK,
    TIMESTAMP_COL,
    X_COL,
    Y_COL,
    Z_COL,
)
from pisces_lite.datasets.data_set_object import DataSetObject

from arcasleep.data.constants import MIN_SCORED_MINUTES
from arcasleep.data.pseudonyms import cache_ids, load_pseudonym_key, pseudonym
from arcasleep.processing import clean_accelerometer_rows
from arcasleep_tree import features

ACCELEROMETER = "accelerometer"
PSG = "psg"
HEARTRATE = "heartrate"
HR_COL = "heart_rate"
EPOCH_S = 30.0
ACCEL_COLUMNS = [TIMESTAMP_COL, X_COL, Y_COL, Z_COL]
PSG_COLUMNS = [TIMESTAMP_COL, PSG_COL]
HR_COLUMNS = [TIMESTAMP_COL, HR_COL]
#: Label of an epoch outside the PSG span of an uncropped recording.
OUTSIDE_PSG = -3
#: Raw 5-class codes (0 wake, 1-4 N1-N4, 5 REM) -> wake, light, deep, REM.
FOUR_CLASS = np.array([0, 1, 1, 2, 2, 3])
#: Separates a recording id's participant from the rest (``sub005_s1``, ``c1_garmin_n1``).
PARTICIPANT_SEPARATOR = "_"

Reader = Callable[[str], pd.DataFrame]


# ---------------------------------------------------------------------------------- shared pieces


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
    """``raw`` stage values as 5-class codes; values not in ``mapping`` are not scored."""
    return np.array([mapping.get(v, PSG_MASK) for v in raw], dtype=int)


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
    name: str, root: Path, ids: Iterable[str], accelerometer: Reader, psg: Reader, heartrate: Reader
) -> DataSetObject:
    """A ``DataSetObject`` whose features are read by the given per-id readers."""
    ids = sorted(ids)
    ds = DataSetObject(name, root)
    ds.set_feature_loader(ACCELEROMETER, accelerometer, ids)
    ds.set_feature_loader(PSG, psg, ids)
    ds.set_feature_loader(HEARTRATE, heartrate, ids)
    return ds


_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


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
    return [dict(zip(rows[0], values, strict=False)) for values in rows[1:]]


# ---------------------------------------------------------------------------------- Empatica E4


E4_ACC_FILE = "ACC.csv"
E4_HR_FILE = "HR.csv"
E4_HEADER_ROWS = 2
E4_UNITS_PER_G = 64.0


@dataclass(frozen=True)
class E4:
    """An Empatica E4 recording.

    The files carry no per-sample timestamps, so sample ``i`` is at ``start + i / rate``.
    """

    acc_start: float
    acc_rate: float
    #: Raw counts, ``(n, 3)``.
    acc: np.ndarray
    hr_start: float
    hr_rate: float
    hr: np.ndarray


def read_e4(read: Callable[[str], bytes]) -> E4:
    """``read(file_name)`` returns the bytes of one file of the recording."""
    acc = pd.read_csv(io.BytesIO(read(E4_ACC_FILE)), header=None).to_numpy(float)
    hr = pd.read_csv(io.BytesIO(read(E4_HR_FILE)), header=None).to_numpy(float)[:, 0]
    return E4(acc[0, 0], acc[1, 0], acc[E4_HEADER_ROWS:], hr[0], hr[1], hr[E4_HEADER_ROWS:])


def e4_accelerometer(e4: E4, start: float) -> pd.DataFrame:
    """The whole stream in g, with sample 0 at ``start``."""
    return accel_frame(start + np.arange(e4.acc.shape[0]) / e4.acc_rate, e4.acc / E4_UNITS_PER_G)


def e4_heart_rate(e4: E4, start: float) -> pd.DataFrame:
    """Heart rate on the clock where accelerometer sample 0 is at ``start``."""
    t = start + (e4.hr_start - e4.acc_start) + np.arange(e4.hr.size) / e4.hr_rate
    return heart_rate_frame(t, e4.hr)


# ---------------------------------------------------------------------------------- cohorts


def sleepaccel(root: Path) -> DataSetObject:
    """SleepAccel and SleepAccel-Clinical.

    Heart rate is ``cleaned_heartrate/<id>_cleaned_hr.out``; a recording without it has none.
    """
    files = cleaned_files(root)
    name = getattr(files.config, "name", None) or root.name
    return data_set(
        name,
        root,
        files.ids,
        accelerometer=renamed(
            lambda id_: files.get_feature_data(ACCELEROMETER, id_), ACCEL_COLUMNS
        ),
        psg=renamed(lambda id_: files.get_feature_data(PSG, id_), PSG_COLUMNS),
        heartrate=lambda id_: read_hr_out(root / "cleaned_heartrate" / f"{id_}_cleaned_hr.out"),
    )


def foundry(root: Path) -> DataSetObject:
    """Foundry.

    Heart rate is the first two columns (time, bpm) of the ``cleaned_heartrate/`` file; a
    recording without one has none.
    """
    files = cleaned_files(root)
    name = getattr(files.config, "name", None) or root.name

    def heartrate(id_: str) -> pd.DataFrame:
        if id_ not in files.get_feature_files(HEARTRATE):
            return no_heart_rate()
        frame = files.get_feature_data(HEARTRATE, id_).iloc[:, : len(HR_COLUMNS)]
        frame.columns = HR_COLUMNS
        return frame

    return data_set(
        name,
        root,
        files.ids,
        accelerometer=renamed(
            lambda id_: files.get_feature_data(ACCELEROMETER, id_), ACCEL_COLUMNS
        ),
        psg=renamed(lambda id_: files.get_feature_data(PSG, id_), PSG_COLUMNS),
        heartrate=heartrate,
    )


AAUWSS_STAGES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 5}
AAUWSS_ANNOTATION_SUFFIX = "_manual_annotation.xlsx"


def _aauwss_annotations(path: Path) -> pd.DataFrame:
    """One row per 30-s epoch from the first scored to the last, on Unix time.

    Epoch start times in the workbook are Copenhagen wall-clock time; epochs it omits are not
    scored.
    """
    rows = sheet_records(path)
    local = pd.Timestamp("1899-12-30") + pd.to_timedelta(
        [r["Event Start Time"] for r in rows], unit="D"
    )
    start = local.round("1s").tz_localize("Europe/Copenhagen").asi8 / 1e9
    codes = map_codes([r["Sleep Stage"] for r in rows], AAUWSS_STAGES)
    index = np.rint((start - start[0]) / EPOCH_S).astype(int)
    if np.any(np.diff(index) < 1):
        raise ValueError(f"{path.name}: scored epochs are not increasing on a 30-s grid")
    filled = np.full(index[-1] + 1, PSG_MASK)
    filled[index] = codes
    return psg_frame(start[0] + EPOCH_S * np.arange(filled.size), filled)


def aauwss(root: Path) -> DataSetObject:
    """AAUWSS: manual annotation workbooks and Empatica E4 files, placed on Unix time."""
    nights = {}
    for path in sorted((root / "annotations").glob(f"*{AAUWSS_ANNOTATION_SUFFIX}")):
        recording = path.name.removesuffix(AAUWSS_ANNOTATION_SUFFIX)
        folder = root / "empatica" / recording
        if (folder / E4_ACC_FILE).exists():
            nights[f"s{re.fullmatch(r'subject_(\d+)', recording).group(1)}"] = (path, folder)

    def e4(id_: str) -> tuple[E4, float]:
        folder = nights[id_][1]
        recording = read_e4(lambda name: (folder / name).read_bytes())
        return recording, recording.acc_start

    return data_set(
        "aauwss",
        root,
        nights,
        accelerometer=lambda id_: e4_accelerometer(*e4(id_)),
        psg=lambda id_: _aauwss_annotations(nights[id_][0]),
        heartrate=lambda id_: e4_heart_rate(*e4(id_)),
    )


WEARANIZE_RELEASE = "Wearanize+_OA_raw_v1.1"
#: Scorer codes: -1 artifact, 0 wake, 1 N1, 2 N2, 3 N3, 4 REM.
WEARANIZE_STAGES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}
#: Device second of a recording's start, and the sync value of a recording that could not be
#: synchronised.
WEARANIZE_ORIGIN_S = 1.0
WEARANIZE_NOT_SYNCHRONISED = -999


def wearanize(root: Path) -> DataSetObject:
    """Wearanize+ open-access raw release.

    Labels are scorer 1's manual scores; the whole E4 stream of each night is read.

    The manual synchronisation sheet places the PSG and the E4 on one axis, with time zero at
    lights-out: a device's second ``s`` is at ``s - <device>_start_sec``.
    """
    release = root / WEARANIZE_RELEASE

    def synchronised(row: dict, start: str, end: str) -> bool:
        return not (row[start] == 0 and row[end] == 0) and row[start] != WEARANIZE_NOT_SYNCHRONISED

    nights = {}
    for row in sheet_records(
        release / "3.Manual_synchronization/Manual_sync_zmax_psg_emp_actpal.xlsx"
    ):
        raw_name = row["SubjectID"].strip("'")
        scores = release / "2.Sleep_scores/1.PSG_manual_scores_scorer1" / f"{raw_name}.txt"
        archives = sorted((release / "1.Raw_data" / raw_name / "3.Empatica").glob("*.zip"))
        if (
            synchronised(row, "PSG_start_sec", "PSG_end_sec")
            and synchronised(row, "Emp_start_sec", "Emp_end_sec")
            and scores.exists()
            and archives
        ):
            match = re.fullmatch(r"sub(\d+)s(\d+)", raw_name.lower())
            nights[f"sub{match.group(1)}_s{match.group(2)}"] = (row, scores, archives[0])

    def e4(id_: str) -> tuple[E4, float]:
        row, _, archive = nights[id_]
        with zipfile.ZipFile(archive) as z:
            recording = read_e4(z.read)
        return recording, WEARANIZE_ORIGIN_S + 1 / recording.acc_rate - float(row["Emp_start_sec"])

    def psg(id_: str) -> pd.DataFrame:
        row, scores, _ = nights[id_]
        lines = scores.read_text().replace("\r", "").split("\n")[1:]
        raw = [int(float(line.replace("\t", ",").split(",")[0])) for line in lines if line.strip()]
        codes = map_codes(raw, WEARANIZE_STAGES)
        return psg_frame(
            WEARANIZE_ORIGIN_S + EPOCH_S * np.arange(codes.size) - float(row["PSG_start_sec"]),
            codes,
        )

    return data_set(
        "wearanize",
        root,
        nights,
        accelerometer=lambda id_: e4_accelerometer(*e4(id_)),
        psg=psg,
        heartrate=lambda id_: e4_heart_rate(*e4(id_)),
    )


WEAVER_DEVICES = ("apple", "fitbit", "actigraph", "garmin", "axivity", "pixel")
WEAVER_PSG_FIELDS = ("time", "epoch", "bp", "stg")


def _weaver_columns(path: Path) -> dict[str, str]:
    """Maps each column's simplified name (lowercase, no spaces or '#') to its name in the file.

    One file's PSG columns lack the "psg" prefix; they are given it here.
    """
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    names = {re.sub(r"[\s#]", "", h).lower(): h for h in header}
    if "psgtime" not in names and all(field in names for field in WEAVER_PSG_FIELDS):
        for field in WEAVER_PSG_FIELDS:
            names[f"psg{field}"] = names.pop(field)
    return names


def _weaver_read(
    path: Path, columns: dict[str, str], keys: list[str], names: list[str]
) -> pd.DataFrame:
    """The rows of ``keys`` (simplified names) with no value missing, under ``names``."""
    wanted = [columns[k] for k in keys]
    frame = pd.read_csv(path, usecols=wanted)[wanted].dropna()
    frame.columns = names
    return frame


def _weaver_psg(path: Path) -> pd.DataFrame:
    """One row per 30-s epoch; the file repeats each epoch's time and stage on every sample row."""
    columns = _weaver_columns(path)
    wanted = [columns["psgtime"], columns["psgstg"]]
    frame = pd.read_csv(path, usecols=wanted, dtype={wanted[1]: str})[wanted]
    frame = frame.dropna(subset=[wanted[0]]).drop_duplicates(subset=[wanted[0]])
    return psg_frame(frame[wanted[0]].to_numpy(), map_codes(frame[wanted[1]], PSG_MAPPING_WEAVER))


def weaver(root: Path) -> DataSetObject:
    """Weaver (revised): one CSV per child's night with every wrist device and the PSG on one clock.

    Each device's recording is its own id, ``<child>_<device>_n<night>``. A device without a
    heart-rate column has no heart rate.
    """
    nights: dict[str, tuple[Path, str, dict[str, str]]] = {}
    for path in sorted((root / "raw").glob("*.csv")):
        columns = _weaver_columns(path)
        if "psgtime" not in columns:
            continue
        child, night = path.stem.rsplit("_n", 1)
        for device in WEAVER_DEVICES:
            if all(f"{device}{c}" in columns for c in ("time", "x", "y", "z")):
                nights[f"{child}_{device}_n{night}"] = (path, device, columns)

    def accelerometer(id_: str) -> pd.DataFrame:
        path, device, columns = nights[id_]
        return _weaver_read(
            path, columns, [f"{device}{c}" for c in ("time", "x", "y", "z")], ACCEL_COLUMNS
        )

    def heartrate(id_: str) -> pd.DataFrame:
        path, device, columns = nights[id_]
        keys = [f"{device}time", f"{device}heartrate"]
        if keys[1] not in columns:
            return no_heart_rate()
        return _weaver_read(path, columns, keys, HR_COLUMNS)

    return data_set(
        "weaver_revised",
        root,
        nights,
        accelerometer=accelerometer,
        psg=lambda id_: _weaver_psg(nights[id_][0]),
        heartrate=heartrate,
    )


#: Dataset directory name -> its loader.
LOADERS = {
    "Wearanize": wearanize,
    "AAUWSS": aauwss,
    "SleepAccel": sleepaccel,
    "SleepAccel-Clinical": sleepaccel,
    "weaver-2026-10-revised": weaver,
    "Foundry": foundry,
}


def load(root: Path | str) -> DataSetObject:
    root = Path(root)
    loader = LOADERS.get(root.name)
    if loader is None:
        raise ValueError(f"{root}: no tree loader for {root.name!r}; known: {sorted(LOADERS)}")
    return loader(root)


# ---------------------------------------------------------------------------------- feature tables


def four_class(labels: np.ndarray) -> np.ndarray:
    """0 wake, 1 light (N1, N2), 2 deep (N3, N4), 3 REM.

    ``PSG_MASK`` where the epoch is not scored.
    """
    labels = np.asarray(labels)
    scored = (labels >= 0) & (labels < FOUR_CLASS.size)
    return np.where(
        scored, FOUR_CLASS[np.clip(labels, 0, FOUR_CLASS.size - 1).astype(int)], PSG_MASK
    )


def participant_of(subject: str) -> str:
    """The participant a recording id belongs to: all of one person's recordings share this."""
    return subject.split(PARTICIPANT_SEPARATOR)[0]


def align_uncropped(
    accel: pd.DataFrame, psg: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Trim ``accel`` to whole epochs on the PSG's epoch phase and label every epoch.

    Returns the trimmed stream, the labels (``OUTSIDE_PSG`` outside the span that ``align_trim``
    keeps, ``PSG_MASK`` where the accelerometer has too few samples) and the time of epoch 0.
    """
    accel = accel.sort_values(by=TIMESTAMP_COL)
    _, inside = align_trim(accel, psg)
    if inside.empty:
        raise ValueError("accelerometer and PSG do not overlap")
    phase = float(psg[TIMESTAMP_COL].min()) % EPOCH_S
    first, last = accel[TIMESTAMP_COL].iloc[0], accel[TIMESTAMP_COL].iloc[-1]
    start = phase + EPOCH_S * np.ceil((first - phase) / EPOCH_S)
    end = phase + EPOCH_S * np.floor((last - phase) / EPOCH_S)
    accel = accel[(accel[TIMESTAMP_COL] >= start) & (accel[TIMESTAMP_COL] <= end)]
    n_epochs = int(round((end - start) / EPOCH_S)) + 1
    from_ = int(round((inside[TIMESTAMP_COL].iloc[0] - start) / EPOCH_S))
    labels = np.full(n_epochs, OUTSIDE_PSG, dtype=int)
    labels[from_ : from_ + len(inside)] = mask_data(accel, inside)[PSG_COL].to_numpy()
    return accel, labels, float(start)


def recording_table(
    data_set: DataSetObject, subject: str, inputs: dict
) -> tuple[pd.DataFrame, np.ndarray]:
    """The feature columns of one recording and its raw labels per epoch."""
    accel = data_set.get_feature_data(ACCELEROMETER, subject)
    psg = data_set.get_feature_data(PSG, subject)
    hr = data_set.get_feature_data(HEARTRATE, subject)
    if accel is None or psg is None:
        raise ValueError("no accelerometer or PSG")
    cleaned = pd.DataFrame(clean_accelerometer_rows(accel.to_numpy()), columns=accel.columns)
    if inputs["uncropped"]:
        cleaned, labels, t0 = align_uncropped(cleaned, psg)
    else:
        cleaned, psg = align_trim(cleaned, psg)
        psg = mask_data(cleaned, psg)
        labels, t0 = psg[PSG_COL].to_numpy(), float(psg[TIMESTAMP_COL].iloc[0])
    rows = cleaned.to_numpy(dtype=np.float64)
    acc25, valid25 = features.to_25hz(rows[:, 0], rows[:, 1:4], t0, labels.size)
    hr = hr.dropna()
    table = features.compute_features(
        acc25, valid25, hr[TIMESTAMP_COL].to_numpy(float) - t0, hr[HR_COL].to_numpy(float), inputs
    )
    return table, labels


_STATE: dict = {}


def _init(data_dir: str, inputs: dict) -> None:
    warnings.simplefilter("ignore")
    _STATE["data_set"] = load(data_dir)
    _STATE["inputs"] = inputs


def _build_one(args: tuple[str, str, str, float]) -> tuple[str, pd.DataFrame | str]:
    subject, reported, person, min_scored_minutes = args
    data_set = _STATE["data_set"]
    try:
        table, labels = recording_table(data_set, subject, _STATE["inputs"])
    except Exception as exc:  # noqa: BLE001 - counted by type in the manifest
        return "failed", type(exc).__name__
    finally:
        for frames in data_set._feature_cache.values():
            frames.pop(subject, None)
    if np.count_nonzero(labels >= 0) * EPOCH_S / 60.0 < min_scored_minutes:
        return "short", ""
    table.insert(0, "in_psg", labels != OUTSIDE_PSG)
    table.insert(0, "y4", four_class(labels).astype(np.int8))
    table.insert(0, "person", person)
    table.insert(0, "subject", reported)
    table.insert(0, "cohort", data_set.name)
    return "ok", table


def build(
    data_dir: Path | str,
    out: Path | str,
    inputs: dict = features.INPUTS,
    min_scored_minutes: float = MIN_SCORED_MINUTES,
    workers: int = 8,
    overwrite: bool = False,
) -> dict:
    """Write ``features-<dataset>.parquet`` and its manifest into ``out``; returns the manifest.

    Recordings with under ``min_scored_minutes`` of staged PSG are left out.
    """
    data_set = load(data_dir)
    out = Path(out)
    stem = f"features-{data_set.name}"
    target = out / f"{stem}.parquet"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite to rebuild it")

    ids = list(data_set.ids)
    key = load_pseudonym_key(Path(data_dir))
    reported = cache_ids(key, ids)
    people = {s: participant_of(s) for s in ids}
    if key is not None:
        people = {s: pseudonym(key, p) for s, p in people.items()}

    tables, reasons = [], {"failed": {}, "short": 0}
    jobs = [(s, reported[s], people[s], min_scored_minutes) for s in ids]
    with ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init,
        initargs=(str(data_dir), inputs),
    ) as pool:
        for status, payload in pool.map(_build_one, jobs):
            if status == "ok":
                tables.append(payload)
            elif status == "short":
                reasons["short"] += 1
            else:
                reasons["failed"][payload] = reasons["failed"].get(payload, 0) + 1
    if not tables:
        raise ValueError(f"{data_dir}: no recording was built: {reasons}")

    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(target, index=False)
    scored = table[table.y4 >= 0]
    manifest = {
        "dataset": data_set.name,
        "data_dir": str(data_dir),
        "inputs": inputs,
        "recordings_in_dataset": len(ids),
        "recordings_built": len(tables),
        "recordings_excluded_short": reasons["short"],
        "recordings_failed_by_error": reasons["failed"],
        "epochs": int(len(table)),
        "scored_epochs": int(len(scored)),
        "class_share": {
            str(k): round(float(v), 4)
            for k, v in scored.y4.value_counts(normalize=True).sort_index().items()
        },
    }
    (out / f"manifest-{stem}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def read_tables(paths: Iterable[Path | str], columns: list[str] | None = None) -> pd.DataFrame:
    """Feature parquet files (or directories of them) as one table."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("features-*.parquet")) if p.is_dir() else [p]
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)

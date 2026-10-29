"""Build the WatchTree feature table of one dataset: ``arcasleep process`` with a ``watchtree`` inputs.json.

We use parque for the cached tables format because it allows for:
* typed, mixed type tables
* loading a subset of columns without reading the rest
* compressing repeated data, like the subject ID repeated each row.
* LightGBM to train directly on the loaded tables. This contrasts with
the .npz format we use to cache preprocessed tensors for BiDoze and UNet models.

To use this CLI command:

    arcasleep process -pc watchtree/config/inputs.json -dd <dataset dir> -o feature_cache/watchtree

The dataset directory is read by its loader in :mod:`arcasleep_watchtree.loaders`. Each row is
one 30-s epoch of one recording: ``cohort``, ``subject``, ``person``, ``y4`` (0 wake, 1 light,
2 deep, 3 REM, ``PSG_MASK`` not scored), ``in_psg``, ``epoch`` and the columns ``inputs.json`` lists.
With ``uncropped`` the whole wrist recording is kept and epochs outside the PSG span have
``in_psg`` False; otherwise both streams are trimmed to their overlap. Recordings with under
``min_scored_minutes`` of staged PSG are left out.
"""

from __future__ import annotations

import json
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.datasets import align_trim, mask_data
from pisces_lite.datasets.constants import PSG_COL, PSG_MASK, TIMESTAMP_COL

from arcasleep.data.pseudonyms import cache_ids, load_pseudonym_key, pseudonym
from arcasleep.processing import clean_accelerometer_rows
from arcasleep_watchtree import loaders
from arcasleep_watchtree.data.uncropped import OUTSIDE_PSG, align_uncropped
from arcasleep_watchtree.features.config import FeatureConfig
from arcasleep_watchtree.features.constants import EPOCH_S
from arcasleep_watchtree.features.preprocess import to_25hz
from arcasleep_watchtree.features.table import compute_features
from arcasleep_watchtree.loaders.common import ACCELEROMETER, HEARTRATE, HR_COL, PSG

#: Separates a recording id's participant from the rest (``sub005_s1``, ``c1_garmin_n1``).
PARTICIPANT_SEPARATOR = "_"
UNCROPPED_SUFFIX = "_uncropped"
#: Raw 5-class codes (0 wake, 1-4 N1-N4, 5 REM) -> wake, light, deep, REM.
FOUR_CLASS = np.array([0, 1, 1, 2, 2, 3])

_STATE: dict = {}


def four_class(labels: np.ndarray) -> np.ndarray:
    """0 wake, 1 light (N1, N2), 2 deep (N3, N4), 3 REM; ``PSG_MASK`` where the epoch is not scored."""
    labels = np.asarray(labels)
    scored = (labels >= 0) & (labels < FOUR_CLASS.size)
    return np.where(scored, FOUR_CLASS[np.clip(labels, 0, FOUR_CLASS.size - 1).astype(int)], PSG_MASK)


def participant_of(subject: str) -> str:
    """The participant a recording id belongs to: all of one person's recordings share this."""
    return subject.split(PARTICIPANT_SEPARATOR)[0]


def recording_table(data_set, subject: str, config: FeatureConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """The feature columns of one recording and its raw labels per epoch."""
    accel = data_set.get_feature_data(ACCELEROMETER, subject)
    psg = data_set.get_feature_data(PSG, subject)
    hr = data_set.get_feature_data(HEARTRATE, subject)
    if accel is None or psg is None:
        raise ValueError("no accelerometer or PSG")
    cleaned = pd.DataFrame(clean_accelerometer_rows(accel.to_numpy()), columns=accel.columns)
    if config.uncropped:
        cleaned, labels, t0 = align_uncropped(cleaned, psg)
    else:
        cleaned, psg = align_trim(cleaned, psg)
        psg = mask_data(cleaned, psg)
        labels, t0 = psg[PSG_COL].to_numpy(), float(psg[TIMESTAMP_COL].iloc[0])
    rows = cleaned.to_numpy(dtype=np.float64)
    acc25, valid25 = to_25hz(rows[:, 0], rows[:, 1:4], t0, labels.size)
    hr = hr.dropna()
    table = compute_features(
        acc25, valid25, hr[TIMESTAMP_COL].to_numpy(float) - t0, hr[HR_COL].to_numpy(float)
    )
    return table[["epoch", *config.names]], labels


def _init(data_dir: str, config_path: str) -> None:
    warnings.simplefilter("ignore")
    _STATE["data_set"] = loaders.load(data_dir)
    _STATE["config"] = FeatureConfig.from_json(config_path)


def _build_one(args: tuple[str, str, str, float]) -> tuple[str, pd.DataFrame | str]:
    subject, reported, person, min_scored_minutes = args
    data_set = _STATE["data_set"]
    try:
        table, labels = recording_table(data_set, subject, _STATE["config"])
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


def process(
    config_path: Path | str,
    data_dir: Path | str,
    out: Path | str,
    min_scored_minutes: float,
    workers: int,
    overwrite: bool = False,
) -> dict:
    """Write ``features-<dataset>[_uncropped].parquet`` and its manifest into ``out``."""
    config = FeatureConfig.from_json(config_path)
    data_set = loaders.load(data_dir)
    out = Path(out)
    stem = f"features-{data_set.name}{UNCROPPED_SUFFIX if config.uncropped else ''}"
    target = out / f"{stem}.parquet"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to rebuild it")

    ids = list(data_set.ids)
    key = load_pseudonym_key(Path(data_dir))
    reported = cache_ids(key, ids)
    people = {s: participant_of(s) for s in ids}
    if key is not None:
        people = {s: pseudonym(key, p) for s, p in people.items()}

    tables, reasons = [], {"failed": {}, "short": 0}
    context = multiprocessing.get_context("spawn")
    jobs = [(s, reported[s], people[s], min_scored_minutes) for s in ids]
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=_init, initargs=(str(data_dir), str(config_path))
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
    summary = {
        "dataset": data_set.name,
        "data_dir": str(data_dir),
        "inputs": str(config_path),
        "uncropped": config.uncropped,
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
    (out / f"manifest-{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary

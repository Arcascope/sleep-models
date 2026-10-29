"""Build one feature table per cohort.  

We use parque for the cached tables format because it allows for:
* typed, mixed type tables
* loading a subset of columns without reading the rest
* compressing repeated data, like the subject ID repeated each row.
* LightGBM to train directly on the loaded tables. This contrasts with
the .npz format we use to cache preprocessed tensors for BiDoze and UNet models.

To use this CLI command:

    python -m arcasleep_watchtree.data.build --cohort wearanize --out results/features-2026-10-27

Each row is one 30-s epoch of one night: ``subject``, ``cohort``, ``epoch``, ``y4`` (0 wake,
1 light, 2 deep, 3 REM, -1 not scored) and the features of
:func:`~arcasleep_watchtree.features.table.feature_names`. Nights with under
:data:`~arcasleep.data.constants.MIN_SCORED_MINUTES` of staged PSG are left out, as in the
BiDoze cache.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.data.constants import MIN_SCORED_MINUTES
from arcasleep_watchtree.data.cohorts import (
    COHORTS,
    DATA_ROOT,
    Night,
    four_class,
    load_night,
    person_ids,
    subject_ids,
)
from arcasleep_watchtree.data.uncropped import OUTSIDE_PSG, load_night_uncropped, sync_table
from arcasleep_watchtree.features.constants import EPOCH_S
from arcasleep_watchtree.features.table import compute_features

_STATE: dict = {}


def night_table(night: Night, person: str) -> pd.DataFrame:
    table = compute_features(night.acc25, night.valid25, night.hr_t, night.hr_bpm)
    table.insert(0, "in_psg", night.labels != OUTSIDE_PSG)
    table.insert(0, "y4", four_class(night.labels).astype(np.int8))
    table.insert(0, "person", person)
    table.insert(0, "subject", night.subject)
    table.insert(0, "cohort", night.cohort)
    return table


def _init(cohort_name: str, uncropped: bool = False) -> None:
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset

    cohort = COHORTS[cohort_name]
    _STATE["cohort"] = cohort
    _STATE["data_set"] = load_dataset(DATA_ROOT / cohort.directory)
    _STATE["uncropped"] = uncropped
    if uncropped:
        from arcasleep_watchtree.data.uncropped import RAW_ROOT

        _STATE["sync"] = sync_table(DATA_ROOT / RAW_ROOT)


def _build_one(args: tuple[str, str, str]) -> tuple[str, pd.DataFrame | str]:
    subject, reported, person = args
    try:
        if _STATE["uncropped"]:
            converted = DATA_ROOT / _STATE["cohort"].directory
            night = load_night_uncropped(subject, reported, _STATE["sync"], converted)
        else:
            night = load_night(_STATE["cohort"], subject, reported, _STATE["data_set"])
    except Exception as exc:  # noqa: BLE001 - counted by type in the manifest
        return "failed", type(exc).__name__
    minutes = np.count_nonzero(night.labels >= 0) * EPOCH_S / 60.0
    if minutes < MIN_SCORED_MINUTES:
        return "short", "under minimum scored minutes"
    return "ok", night_table(night, person)


def build_cohort(
    cohort_name: str, out: Path, workers: int, limit: int | None = None, uncropped: bool = False
) -> dict:
    cohort = COHORTS[cohort_name]
    ids, reported = subject_ids(cohort)
    ids = ids[:limit] if limit else ids
    people = person_ids(cohort, ids)
    out.mkdir(parents=True, exist_ok=True)
    tables, reasons = [], {"failed": {}, "short": 0}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=_init, initargs=(cohort_name, uncropped)
    ) as pool:
        for status, payload in pool.map(_build_one, [(s, reported[s], people[s]) for s in ids]):
            if status == "ok":
                tables.append(payload)
            elif status == "short":
                reasons["short"] += 1
            else:
                reasons["failed"][payload] = reasons["failed"].get(payload, 0) + 1
    table = pd.concat(tables, ignore_index=True)
    suffix = "_uncropped" if uncropped else ""
    table.to_parquet(out / f"features-{cohort_name}{suffix}.parquet", index=False)
    scored = table[table.y4 >= 0]
    return {
        "cohort": cohort_name,
        "nights_in_dataset": len(ids),
        "nights_built": len(tables),
        "nights_excluded_short": reasons["short"],
        "nights_failed_by_error": reasons["failed"],
        "epochs": int(len(table)),
        "scored_epochs": int(len(scored)),
        "class_share": {
            str(k): round(float(v), 4)
            for k, v in scored.y4.value_counts(normalize=True).sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--uncropped", action="store_true", help="Wearanize+ only: read the whole wrist stream"
    )
    parser.add_argument("--limit", type=int, default=None, help="first N subjects only")
    args = parser.parse_args()
    summary = build_cohort(args.cohort, args.out, args.workers, args.limit, args.uncropped)
    (args.out / f"manifest-{args.cohort}{"_uncropped" if args.uncropped else ""}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

"""Build one feature table (parquet) per cohort.

    python -m arcasleep_watchtree.build --cohort wearanize --out results/features-2026-10-27

Each row is one 30-s epoch of one night: ``subject``, ``cohort``, ``epoch``, ``y4`` (0 wake,
1 light, 2 deep, 3 REM, -1 not scored) and the features of
:func:`~arcasleep_watchtree.features.feature_names`. Nights with under
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
from arcasleep_watchtree.cohorts import COHORTS, Night, four_class, load_night, subject_ids
from arcasleep_watchtree.constants import EPOCH_S
from arcasleep_watchtree.features import compute_features

_STATE: dict = {}


def night_table(night: Night) -> pd.DataFrame:
    table = compute_features(night.acc25, night.valid25, night.hr_t, night.hr_bpm)
    table.insert(0, "y4", four_class(night.labels).astype(np.int8))
    table.insert(0, "subject", night.subject)
    table.insert(0, "cohort", night.cohort)
    return table


def _init(cohort_name: str) -> None:
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    from arcasleep_watchtree.cohorts import DATA_ROOT

    cohort = COHORTS[cohort_name]
    _STATE["cohort"] = cohort
    _STATE["data_set"] = load_dataset(DATA_ROOT / cohort.directory)


def _build_one(args: tuple[str, str]) -> tuple[str, pd.DataFrame | str]:
    subject, reported = args
    try:
        night = load_night(_STATE["cohort"], subject, reported, _STATE["data_set"])
    except Exception as exc:  # noqa: BLE001 - counted by type in the manifest
        return "failed", type(exc).__name__
    minutes = np.count_nonzero(night.labels >= 0) * EPOCH_S / 60.0
    if minutes < MIN_SCORED_MINUTES:
        return "short", "under minimum scored minutes"
    return "ok", night_table(night)


def build_cohort(cohort_name: str, out: Path, workers: int, limit: int | None = None) -> dict:
    cohort = COHORTS[cohort_name]
    ids, reported = subject_ids(cohort)
    ids = ids[:limit] if limit else ids
    out.mkdir(parents=True, exist_ok=True)
    tables, reasons = [], {"failed": {}, "short": 0}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=_init, initargs=(cohort_name,)
    ) as pool:
        for status, payload in pool.map(_build_one, [(s, reported[s]) for s in ids]):
            if status == "ok":
                tables.append(payload)
            elif status == "short":
                reasons["short"] += 1
            else:
                reasons["failed"][payload] = reasons["failed"].get(payload, 0) + 1
    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(out / f"features-{cohort_name}.parquet", index=False)
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
    parser.add_argument("--limit", type=int, default=None, help="first N subjects only")
    args = parser.parse_args()
    summary = build_cohort(args.cohort, args.out, args.workers, args.limit)
    (args.out / f"manifest-{args.cohort}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

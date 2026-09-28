"""Pool finished runs into cohort summaries of their per-subject metrics.

A finished training or evaluation phase leaves a ``cv_results_<n>_stages.csv``
behind: one row per subject per metric. This module discovers those CSVs, groups
them into cohorts, and macro-averages every metric over the subjects in each
cohort -- the mean over subjects, each weighted once regardless of how many
epochs it has.

Two kinds of directory are understood:

* ``fold<i>``  -- the folds of a cross-validation run. Every fold found is
  pooled into a single ``cv`` cohort, so each subject is scored by the model
  that never trained on it. This is the one cohort whose name is not a dataset.
* anything else -- grouped by the ``dataset`` column inside its CSV, so a phase
  scored on several external cohorts yields one cohort per dataset.

The scan looks at each directory given and its immediate subdirectories, one
level deep.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

#: The results CSV a phase writes; the ``<n>`` is its class count.
RESULTS_GLOB = "cv_results_*_stages.csv"
#: The fold-directory pattern that pools into the CV cohort.
FOLD_DIR = re.compile(r"^fold\d+$")
#: What the pooled folds are called.
CV_COHORT = "cv"

#: Metrics stored as fractions in the CSV but read as percentages by a human.
AS_PERCENT = {"sleep_sensitivity", "wake_specificity"}

#: The headline metrics, in the reading order the figures use. Metrics found
#: beyond these keep their own names and are listed after them, alphabetically.
HEADLINE_METRICS = [
    "sleep_sensitivity",
    "wake_specificity",
    "kappa_sleep_wake",
    "kappa_4stage",
    "tst_error_minutes",
    "tst_mape",
    "waso_mape",
    "light_mape",
    "deep_mape",
    "rem_mape",
]

SUMMARY_COLUMNS = [
    "metric", "cohort", "n", "n_missing",
    "mean", "sd", "sem", "ci95_lo", "ci95_hi",
    "median", "iqr_lo", "iqr_hi", "min", "max",
]


@dataclass(frozen=True)
class Phase:
    """A directory that holds a results CSV, and how it is grouped."""

    directory: Path
    csv_path: Path
    fold: str | None          # the fold's name when it pools into the CV cohort


def _results_csv(directory: Path) -> Path | None:
    """The one results CSV in ``directory``, or None when it holds none.

    More than one means the directory is not a single phase, so it is refused
    rather than guessed at.
    """
    matches = sorted(directory.glob(RESULTS_GLOB))
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"{directory} holds more than one results CSV ({names}); "
            "point summarize at the individual phase directories"
        )
    return matches[0]


def discover_phases(directories: Iterable[Path | str]) -> list[Phase]:
    """Find every results CSV in ``directories`` and their immediate subdirs.

    Each directory is considered itself and one level down; a ``fold<i>``
    basename marks a phase for the CV cohort, anything else is grouped by its
    CSV's datasets later.
    """
    phases: list[Phase] = []
    for raw in directories:
        root = Path(raw)
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        candidates = [root, *(p for p in sorted(root.iterdir()) if p.is_dir())]
        for candidate in candidates:
            csv_path = _results_csv(candidate)
            if csv_path is None:
                continue
            name = candidate.name
            fold = name if FOLD_DIR.match(name) else None
            phases.append(Phase(directory=candidate, csv_path=csv_path, fold=fold))
    if not phases:
        joined = ", ".join(str(d) for d in directories)
        raise ValueError(f"no {RESULTS_GLOB} found in or under: {joined}")
    return phases


def to_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per subject, one column per metric, percentages where expected."""
    wide = frame.pivot_table(
        index="subject_id", columns="metric", values="value", aggfunc="first"
    )
    for metric in AS_PERCENT:
        if metric in wide.columns:
            wide[metric] = wide[metric] * 100.0
    return wide


def describe(values: pd.Series) -> dict:
    """The macro summary of one metric's per-subject values."""
    v = values.dropna().astype(float)
    n = len(v)
    sd = v.std(ddof=1)
    sem = sd / np.sqrt(n)
    return {
        "n": n,
        "n_missing": int(values.isna().sum()),
        "mean": v.mean(),
        "sd": sd,
        "sem": sem,
        "ci95_lo": v.mean() - 1.96 * sem,
        "ci95_hi": v.mean() + 1.96 * sem,
        "median": v.median(),
        "iqr_lo": v.quantile(0.25),
        "iqr_hi": v.quantile(0.75),
        "min": v.min(),
        "max": v.max(),
    }


def _metric_order(available: Iterable[str]) -> list[str]:
    """Headline metrics in reading order, then anything else alphabetically."""
    present = set(available)
    known = [m for m in HEADLINE_METRICS if m in present]
    return known + sorted(m for m in present if m not in set(HEADLINE_METRICS))


def build_cohorts(phases: Iterable[Phase]) -> dict[str, pd.DataFrame]:
    """Name -> per-subject frame, with the folds pooled under ``cv``.

    Insertion order is the cohort order: ``cv`` first, then each dataset in the
    order it is first seen across the phases.
    """
    fold_parts: list[pd.DataFrame] = []
    by_dataset: dict[str, list[pd.DataFrame]] = {}
    for phase in phases:
        frame = pd.read_csv(phase.csv_path)
        if phase.fold is not None:
            fold_parts.append(frame.assign(cv_fold=phase.fold))
        else:
            for dataset, grp in frame.groupby("dataset", sort=False):
                by_dataset.setdefault(str(dataset), []).append(grp)

    cohorts: dict[str, pd.DataFrame] = {}
    if fold_parts:
        subjects = to_subjects(pd.concat(fold_parts, ignore_index=True))
        cohorts[CV_COHORT] = subjects
    for dataset, parts in by_dataset.items():
        cohorts[dataset] = to_subjects(pd.concat(parts, ignore_index=True))
    return cohorts


def summary_table(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (metric, cohort), macro-averaged over subjects."""
    available: set[str] = set()
    for subjects in cohorts.values():
        available.update(subjects.columns)
    order = _metric_order(available)

    rows = []
    for metric in order:
        for cohort, subjects in cohorts.items():
            if metric not in subjects.columns:
                continue
            rows.append({"metric": metric, "cohort": cohort,
                         **describe(subjects[metric])})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summarize(directories: Iterable[Path | str], output_dir: Path | str) -> Path:
    """Discover phases under ``directories`` and write ``metrics_summary.csv``."""
    phases = discover_phases(directories)
    cohorts = build_cohorts(phases)

    table = summary_table(cohorts)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics_summary.csv"
    table.to_csv(out_path, index=False)

    for phase in phases:
        kind = f"cv/{phase.fold}" if phase.fold else "cohort"
        print(f"  {phase.csv_path}  ({kind})")
    for cohort, subjects in cohorts.items():
        print(f"{cohort}: {len(subjects)} subjects")
    print(f"wrote {out_path}")
    return out_path

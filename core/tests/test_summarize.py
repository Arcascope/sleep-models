"""Turning finished phases into a per-cohort metrics_summary.csv.

The summary is pure bookkeeping over the CSVs a run leaves behind, so these
tests build small CSVs and pin down which directory joins which cohort and what
the resulting table says -- no model, no JAX.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from arcasleep.reporting.summary import (
    CV_COHORT,
    build_cohorts,
    discover_phases,
    summarize,
    summary_table,
    to_subjects,
)

HEADER = [
    "run_id", "fold", "epoch", "subject_id",
    "dataset", "training_dataset", "metric", "value",
]


def _write_csv(
    directory: Path,
    rows: list[tuple[str, str, str, float]],
    *,
    filename: str = "cv_results_4_stages.csv",
) -> Path:
    """One CSV from ``(subject_id, dataset, metric, value)`` rows."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for subject, dataset, metric, value in rows:
            writer.writerow(
                ["run", "final_seed_0", 200, subject, dataset, "Train", metric, value]
            )
    return path


def _subject(dataset: str, subject: str, sensitivity: float, wake: float) -> list:
    return [
        (subject, dataset, "sleep_sensitivity", sensitivity),
        (subject, dataset, "wake_specificity", wake),
    ]


# --- discovery --------------------------------------------------------------


def test_discover_finds_the_directory_itself_and_one_level_down(tmp_path: Path) -> None:
    _write_csv(tmp_path / "validation", _subject("SleepAccel-Clinical", "a", 0.9, 0.5))
    _write_csv(tmp_path / "external", _subject("SleepAccel", "b", 0.8, 0.4))

    phases = discover_phases([tmp_path])

    found = {phase.directory.name for phase in phases}
    assert found == {"validation", "external"}
    assert all(phase.fold is None for phase in phases)


def test_discover_pools_fold_directories_and_names_them(tmp_path: Path) -> None:
    for i in range(3):
        _write_csv(
            tmp_path / f"fold{i}",
            _subject("BiDozeWeaverFold0", f"s{i}", 0.9, 0.5),
        )

    phases = discover_phases([tmp_path])

    folds = sorted(phase.fold for phase in phases)
    assert folds == ["fold0", "fold1", "fold2"]


def test_discover_refuses_a_directory_with_two_results_csvs(tmp_path: Path) -> None:
    rows = _subject("A", "s", 0.9, 0.5)
    _write_csv(tmp_path, rows, filename="cv_results_2_stages.csv")
    _write_csv(tmp_path, rows, filename="cv_results_4_stages.csv")

    with pytest.raises(ValueError):
        discover_phases([tmp_path])


def test_discover_refuses_when_nothing_matches(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError):
        discover_phases([tmp_path])


def test_discover_refuses_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        discover_phases([tmp_path / "absent"])


# --- cohorting --------------------------------------------------------------


def test_folds_pool_into_one_cv_cohort(tmp_path: Path) -> None:
    _write_csv(tmp_path / "fold0", _subject("BiDozeWeaverFold0", "s0", 1.0, 0.5))
    _write_csv(tmp_path / "fold1", _subject("BiDozeWeaverFold1", "s1", 0.8, 0.3))

    cohorts = build_cohorts(discover_phases([tmp_path]))

    assert list(cohorts) == [CV_COHORT]
    assert len(cohorts[CV_COHORT]) == 2  # both folds in one frame


def test_non_fold_directories_are_grouped_by_dataset(tmp_path: Path) -> None:
    _write_csv(tmp_path / "final", _subject("SleepAccel", "a", 0.9, 0.5))
    _write_csv(tmp_path / "validation", _subject("SleepAccel-Clinical", "b", 0.7, 0.4))

    cohorts = build_cohorts(discover_phases([tmp_path]))

    assert set(cohorts) == {"SleepAccel", "SleepAccel-Clinical"}


def test_cv_comes_first_and_datasets_follow_in_seen_order(tmp_path: Path) -> None:
    _write_csv(tmp_path / "fold0", _subject("BiDozeWeaverFold0", "s0", 0.9, 0.5))
    _write_csv(tmp_path / "final", _subject("SleepAccel", "a", 0.9, 0.5))
    _write_csv(tmp_path / "validation", _subject("SleepAccel-Clinical", "b", 0.7, 0.4))

    cohorts = build_cohorts(discover_phases([tmp_path]))

    assert list(cohorts) == [CV_COHORT, "SleepAccel", "SleepAccel-Clinical"]


# --- the summary ------------------------------------------------------------


def test_summary_macro_averages_each_metric_over_subjects(tmp_path: Path) -> None:
    _write_csv(
        tmp_path,
        _subject("SleepAccel", "a", 0.6, 0.2) + _subject("SleepAccel", "b", 1.0, 0.4),
    )

    table = summary_table(build_cohorts(discover_phases([tmp_path])))

    row = table[(table.cohort == "SleepAccel")
                & (table.metric == "sleep_sensitivity")].iloc[0]
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(80.0)     # (60 + 100) / 2, scaled to %
    assert row["min"] == pytest.approx(60.0)
    assert row["max"] == pytest.approx(100.0)


def test_summary_scales_sensitivity_and_specificity_to_percent(tmp_path: Path) -> None:
    _write_csv(tmp_path, _subject("A", "a", 0.97, 0.66))

    table = summary_table(build_cohorts(discover_phases([tmp_path])))

    means = dict(zip(table.metric, table["mean"], strict=True))
    assert means["sleep_sensitivity"] == pytest.approx(97.0)
    assert means["wake_specificity"] == pytest.approx(66.0)


def test_summary_keeps_metrics_beyond_the_headline_list(tmp_path: Path) -> None:
    rows = _subject("A", "a", 0.9, 0.5) + [("a", "A", "pred_wake_count", 3.0)]
    _write_csv(tmp_path, rows)

    table = summary_table(build_cohorts(discover_phases([tmp_path])))

    assert "pred_wake_count" in set(table.metric)


def test_summary_drops_nan_from_that_metrics_mean_only(tmp_path: Path) -> None:
    # ``b`` has no tst_mape to divide by, but is a real subject with other
    # metrics, so it must survive as a subject and be counted as missing here.
    frame = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "b"],
            "dataset": ["A", "A", "A", "A"],
            "metric": ["tst_mape", "wake_specificity", "tst_mape", "wake_specificity"],
            "value": [10.0, 0.5, float("nan"), 0.6],
        }
    )
    cohorts = {CV_COHORT: to_subjects(frame)}

    table = summary_table(cohorts)

    row = table[table.metric == "tst_mape"].iloc[0]
    assert row["n"] == 1
    assert row["n_missing"] == 1
    assert row["mean"] == pytest.approx(10.0)


# --- end to end -------------------------------------------------------------


def test_summarize_writes_the_table(tmp_path: Path) -> None:
    _write_csv(tmp_path / "fold0", _subject("BiDozeWeaverFold0", "s0", 0.9, 0.5))
    _write_csv(tmp_path / "fold1", _subject("BiDozeWeaverFold1", "s1", 0.8, 0.4))
    _write_csv(tmp_path / "final", _subject("SleepAccel", "a", 0.7, 0.3))
    _write_csv(tmp_path / "validation", _subject("SleepAccel-Clinical", "b", 0.6, 0.2))
    out = tmp_path / "analysis"

    path = summarize([tmp_path], out)

    assert path == out / "metrics_summary.csv"
    written = pd.read_csv(path)
    assert set(written.cohort) == {CV_COHORT, "SleepAccel", "SleepAccel-Clinical"}
    assert (written.cohort == CV_COHORT).any()


def test_summarize_accepts_the_phase_directories_directly(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "training" / "fold0",
        _subject("BiDozeWeaverFold0", "s0", 0.9, 0.5),
    )
    _write_csv(tmp_path / "validation", _subject("SleepAccel-Clinical", "b", 0.6, 0.2))

    path = summarize(
        [tmp_path / "training", tmp_path / "validation"], tmp_path / "out"
    )

    written = pd.read_csv(path)
    assert set(written.cohort) == {CV_COHORT, "SleepAccel-Clinical"}


def test_balanced_accuracy_is_the_mean_of_sensitivity_and_specificity_in_percent() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["a", "a"],
            "dataset": ["A", "A"],
            "metric": ["sleep_sensitivity", "wake_specificity"],
            "value": [0.9, 0.5],
        }
    )

    subjects = to_subjects(frame)

    assert subjects.loc["a", "balanced_accuracy"] == pytest.approx(70.0)

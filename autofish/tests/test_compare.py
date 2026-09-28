"""Comparing several metrics summaries in one grid.

The drawing is matplotlib's job; what these tests pin down is the data layer
the grid is built from -- how summaries and labels pair up, which cohorts and
panels are discovered, and the arithmetic that places the bars -- plus one
end-to-end smoke test that the files are written.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from autofish.cli import _pair_summaries
from autofish.reporting.compare import (
    PANELS,
    RunSummary,
    _agg_columns,
    available_panels,
    cohort_order,
    cohort_sizes,
    compare,
    default_label,
    draw_comparison,
    load_summaries,
    required_columns,
)

HEADER = ["metric", "cohort", "n", "mean", "ci95_lo", "ci95_hi"]
FULL_HEADER = HEADER + ["median", "iqr_lo", "iqr_hi"]


def _write_summary(directory: Path, rows: list[tuple]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def _write_full_summary(directory: Path, rows: list[tuple]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FULL_HEADER)
        writer.writerows(rows)
    return path


def _row(metric: str, cohort: str, n: int, mean: float, lo: float, hi: float):
    return (metric, cohort, n, mean, lo, hi)


def _full_row(metric: str, cohort: str, n: int, mean: float, lo: float, hi: float,
              median: float, iqr_lo: float, iqr_hi: float):
    return (metric, cohort, n, mean, lo, hi, median, iqr_lo, iqr_hi)


# --- loading ----------------------------------------------------------------


def test_load_summaries_pairs_labels_in_order(tmp_path: Path) -> None:
    a = _write_summary(
        tmp_path / "run_a" / "analysis", [_row("tst_mape", "cv", 9, 5.0, 4.0, 6.0)]
    )
    b = _write_summary(
        tmp_path / "run_b" / "analysis", [_row("tst_mape", "cv", 9, 6.0, 5.0, 7.0)]
    )

    runs = load_summaries([a, b], ["200 epochs", "30 epochs"])

    assert [run.label for run in runs] == ["200 epochs", "30 epochs"]
    assert [run.path for run in runs] == [a, b]


def test_load_summaries_defaults_to_the_run_directory_name(tmp_path: Path) -> None:
    path = _write_summary(
        tmp_path / "ecs-smoke-g6" / "analysis",
        [_row("tst_mape", "cv", 9, 5.0, 4.0, 6.0)],
    )

    runs = load_summaries([path])

    assert runs[0].label == "ecs-smoke-g6"
    assert default_label(path) == "ecs-smoke-g6"


def test_load_summaries_refuses_a_label_count_mismatch(tmp_path: Path) -> None:
    path = _write_summary(
        tmp_path / "run" / "analysis", [_row("tst_mape", "cv", 9, 5.0, 4.0, 6.0)]
    )

    with pytest.raises(ValueError, match="labels for"):
        load_summaries([path], ["a", "b"])


def test_load_summaries_refuses_a_non_summary_csv(tmp_path: Path) -> None:
    path = tmp_path / "metrics_summary.csv"
    path.write_text("metric,cohort\nfoo,cv\n")

    with pytest.raises(ValueError, match="not a metrics summary"):
        load_summaries([path])


def test_load_summaries_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such summary"):
        load_summaries([tmp_path / "absent.csv"])


# --- discovery --------------------------------------------------------------


def _runs(*tables: list[tuple]) -> list[RunSummary]:
    runs = []
    for i, rows in enumerate(tables):
        frame = pd.DataFrame(rows, columns=HEADER)
        runs.append(RunSummary(path=Path(f"run{i}"), label=f"run{i}", table=frame))
    return runs


def test_cohort_order_is_first_appearance_across_runs() -> None:
    runs = _runs(
        [
            _row("tst_mape", "cv", 9, 5.0, 4.0, 6.0),
            _row("tst_mape", "SleepAccel", 3, 5.0, 4.0, 6.0),
        ],
        [
            _row("tst_mape", "cv", 9, 5.0, 4.0, 6.0),
            _row("tst_mape", "SleepAccel-Clinical", 4, 5.0, 4.0, 6.0),
        ],
    )

    assert cohort_order(runs) == ["cv", "SleepAccel", "SleepAccel-Clinical"]


def test_available_panels_keeps_reading_order_and_allows_gaps() -> None:
    runs = _runs(
        [_row("sleep_sensitivity", "cv", 9, 90.0, 88.0, 92.0),
         _row("tst_mape", "cv", 9, 5.0, 4.0, 6.0)],
    )

    order = [p[0] for p in available_panels(runs)]

    assert order == ["sleep_sensitivity", "tst_mape"]
    assert order == [m for m in (p[0] for p in PANELS) if m in order]


def test_cohort_sizes_reads_n_and_reports_a_range_when_runs_disagree() -> None:
    runs = _runs(
        [_row("tst_mape", "cv", 200, 5.0, 4.0, 6.0)],
        [_row("tst_mape", "cv", 202, 5.0, 4.0, 6.0)],
    )

    assert cohort_sizes(runs, "cv") == "n=200–202"


def test_cohort_sizes_single_value(tmp_path: Path) -> None:
    runs = _runs([_row("tst_mape", "SleepAccel", 30, 5.0, 4.0, 6.0)])

    assert cohort_sizes(runs, "SleepAccel") == "n=30"


# --- end to end -------------------------------------------------------------


def test_compare_writes_both_formats(tmp_path: Path) -> None:
    a = _write_summary(
        tmp_path / "a" / "analysis",
        [_row("sleep_sensitivity", "cv", 202, 97.0, 96.0, 98.0),
         _row("sleep_sensitivity", "SleepAccel", 30, 97.5, 96.5, 98.5)],
    )
    b = _write_summary(
        tmp_path / "b" / "analysis",
        [_row("sleep_sensitivity", "cv", 202, 96.5, 95.5, 97.5)],
    )
    out = tmp_path / "figures"

    written = compare([a, b], output_dir=out, labels=["200", "30"])

    assert written == [out / "metrics_grid.png", out / "metrics_grid.pdf"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_draw_comparison_refuses_when_no_summaries() -> None:
    with pytest.raises(ValueError, match="no summaries"):
        draw_comparison([], output_dir=".")


def test_draw_comparison_writes_a_custom_name(tmp_path: Path) -> None:
    a = _write_summary(
        tmp_path / "a" / "analysis",
        [_row("tst_mape", "cv", 9, 5.0, 4.0, 6.0)],
    )

    written = draw_comparison(
        load_summaries([a]), output_dir=tmp_path / "out",
        name="grid_30_vs_200", title="My title",
    )

    assert written[0].name == "grid_30_vs_200.png"


def test_draw_comparison_draws_only_the_requested_metrics(tmp_path: Path) -> None:
    a = _write_summary(
        tmp_path / "a" / "analysis",
        [
            _row("tst_mape", "cv", 9, 5.0, 4.0, 6.0),
            _row("kappa_4stage", "cv", 9, 0.5, 0.4, 0.6),
        ],
    )
    runs = load_summaries([a])

    written = draw_comparison(
        runs, output_dir=tmp_path / "out", metrics=["tst_mape"], ncols=3
    )
    assert written[0].is_file()
    # A selection none of the summaries carries has nothing to draw.
    with pytest.raises(ValueError, match="no headline metrics"):
        draw_comparison(runs, output_dir=tmp_path / "out", metrics=["waso_mape"])


# --- -s pairing -------------------------------------------------------------


def test_pair_summaries_reads_path_then_optional_name() -> None:
    paths, names = _pair_summaries([["a.csv", "200 epochs"], ["b.csv", "30 epochs"]])

    assert paths == [Path("a.csv"), Path("b.csv")]
    assert names == ["200 epochs", "30 epochs"]


def test_pair_summaries_returns_no_names_when_none_are_given() -> None:
    paths, names = _pair_summaries([["a.csv"], ["b.csv"]])

    assert paths == [Path("a.csv"), Path("b.csv")]
    assert names is None


def test_pair_summaries_refuses_a_mix_of_named_and_unnamed() -> None:
    with pytest.raises(SystemExit, match="name every summary or none"):
        _pair_summaries([["a.csv", "200"], ["b.csv"]])


def test_pair_summaries_refuses_more_than_path_and_name() -> None:
    with pytest.raises(SystemExit, match="at most one name"):
        _pair_summaries([["a.csv", "200", "extra"]])


# --- aggregation ------------------------------------------------------------


def test_required_columns_adds_the_agg_columns() -> None:
    assert set(required_columns("mean")) >= {"mean", "ci95_lo", "ci95_hi"}
    assert set(required_columns("median")) >= {"median", "iqr_lo", "iqr_hi"}


def test_load_summaries_refuses_a_table_missing_the_agg_columns(tmp_path: Path) -> None:
    path = _write_summary(
        tmp_path / "run" / "analysis", [_row("tst_mape", "cv", 9, 5.0, 4.0, 6.0)]
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_summaries([path], agg="median")


def test_agg_columns_rejects_an_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        _agg_columns("mode")


def test_load_summaries_accepts_a_full_table_for_median(tmp_path: Path) -> None:
    path = _write_full_summary(
        tmp_path / "run" / "analysis",
        [_full_row("tst_mape", "cv", 9, 5.0, 4.0, 6.0, 4.5, 3.0, 7.0)],
    )

    runs = load_summaries([path], agg="median")

    assert runs[0].table.iloc[0]["median"] == 4.5


def test_compare_draws_with_medians(tmp_path: Path) -> None:
    path = _write_full_summary(
        tmp_path / "a" / "analysis",
        [
            _full_row("sleep_sensitivity", "cv", 202,
                      97.0, 96.0, 98.0, 98.0, 97.0, 99.0),
            _full_row("sleep_sensitivity", "SleepAccel", 30,
                      97.5, 96.5, 98.5, 98.8, 97.5, 99.5),
        ],
    )
    out = tmp_path / "figures"

    written = compare([path], output_dir=out, labels=["200"], agg="median")

    assert written[0].is_file()


# --- signed metrics ----------------------------------------------------------


def test_draw_comparison_survives_a_negative_mean(tmp_path: Path) -> None:
    """A signed metric may sit below zero; the panel must scale to it."""
    a = _write_summary(
        tmp_path / "a" / "analysis",
        [_row("tst_error_minutes", "cv", 202, -9.5, -12.0, -7.0)],
    )
    out = tmp_path / "figures"

    written = draw_comparison(load_summaries([a]), output_dir=out)

    assert written[0].is_file()


def test_draw_comparison_survives_a_mixed_sign_panel(tmp_path: Path) -> None:
    """One cohort negative, one positive: both bars share the panel."""
    a = _write_summary(
        tmp_path / "a" / "analysis",
        [
            _row("tst_error_minutes", "cv", 202, -9.5, -12.0, -7.0),
            _row("tst_error_minutes", "SleepAccel", 30, 4.0, 2.0, 6.0),
        ],
    )
    out = tmp_path / "figures"

    written = draw_comparison(load_summaries([a]), output_dir=out)

    assert written[0].is_file()


def test_draw_comparison_survives_an_all_negative_panel(tmp_path: Path) -> None:
    a = _write_summary(
        tmp_path / "a" / "analysis",
        [
            _row("tst_error_minutes", "cv", 202, -9.5, -12.0, -7.0),
            _row("tst_error_minutes", "SleepAccel", 30, -3.0, -5.0, -1.0),
        ],
    )
    out = tmp_path / "figures"

    written = draw_comparison(load_summaries([a]), output_dir=out)

    assert written[0].is_file()

"""The raw-accelerometer inference helper: file reading, epoch timing, output shape."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from arcasleep.predict import (
    Prediction,
    _resolve_origin,
    _softmax,
    _stage_and_gap_probabilities,
    read_accelerometer,
)


def _prediction() -> Prediction:
    return Prediction(
        epoch_start_s=np.array([100.0, 130.0, 160.0]),
        probabilities=np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]]),
        stage_names=("wake", "light", "deep", "rem"),
        is_gap=np.array([False, True, False]),
    )


def test_gap_epochs_are_labelled_gap_whatever_the_model_guessed() -> None:
    assert list(_prediction().labels) == ["wake", "gap", "rem"]
    assert list(_prediction().stages) == [0, 1, 3]


def test_csv_has_one_row_per_epoch_with_probabilities_and_gap_flag(tmp_path) -> None:
    path = _prediction().to_csv(tmp_path / "out" / "night.csv")

    rows = list(csv.DictReader(path.open()))

    assert [r["stage"] for r in rows] == ["wake", "gap", "rem"]
    assert [r["stage_raw"] for r in rows] == ["wake", "gap", "rem"]
    assert [r["is_gap"] for r in rows] == ["0", "1", "0"]
    assert [r["is_nap"] for r in rows] == ["0", "0", "0"]
    assert rows[2]["start_s"] == "160.000"
    assert float(rows[0]["wake"]) == pytest.approx(0.7)


def test_gap_head_column_follows_the_stage_probabilities(tmp_path) -> None:
    base = _prediction()
    result = Prediction(
        epoch_start_s=base.epoch_start_s,
        probabilities=base.probabilities,
        stage_names=base.stage_names,
        is_gap=base.is_gap,
        gap_head=np.array([0.01, 0.9, 0.0]),
    )

    rows = list(csv.DictReader(result.to_csv(tmp_path / "night.csv").open()))

    assert list(rows[0]) == [
        "epoch", "start_s", "wake", "light", "deep", "rem", "gap_head", "stage",
        "stage_raw", "is_nap", "is_sib", "is_spt", "is_gap",
    ]
    assert [float(r["gap_head"]) for r in rows] == pytest.approx([0.01, 0.9, 0.0])
    assert [r["stage"] for r in rows] == ["wake", "gap", "rem"]


def test_stages_are_renormalized_without_the_gap_head() -> None:
    logits = np.array([[2.0, 1.0, 0.5, 0.1, 3.0], [0.0, 1.0, 2.0, 3.0, -1.0]])

    stages, gap = _stage_and_gap_probabilities(logits, 4)

    full = _softmax(logits)
    np.testing.assert_allclose(stages, full[:, :4] / full[:, :4].sum(-1, keepdims=True))
    np.testing.assert_allclose(stages.sum(-1), 1.0)
    np.testing.assert_allclose(gap, full[:, 4])
    total = stages.sum(-1) + gap
    assert np.all((total >= 1.0) & (total < 2.0))


def test_a_model_without_a_gap_head_has_no_gap_probability() -> None:
    stages, gap = _stage_and_gap_probabilities(np.array([[2.0, 1.0, 0.5, 0.1]]), 4)

    assert gap is None
    np.testing.assert_allclose(stages.sum(-1), 1.0)

def test_filtered_prediction_reports_stage_and_nap_separately(tmp_path) -> None:
    prediction = Prediction(
        epoch_start_s=np.array([0.0, 30.0, 60.0]),
        probabilities=np.array(
            [[0.1, 0.6, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]]
        ),
        stage_names=("wake", "light", "deep", "rem"),
        is_gap=np.zeros(3, dtype=bool),
        stage=np.array(["light", "wake", "wake"], dtype=object),
        stage_raw=np.array(["light", "light", "rem"], dtype=object),
        is_nap=np.array([0, 1, 0], dtype=np.int8),
    )

    path = prediction.to_csv(tmp_path / "night.csv")
    rows = list(csv.DictReader(path.open()))

    assert [r["stage"] for r in rows] == ["light", "wake", "wake"]
    assert [r["stage_raw"] for r in rows] == ["light", "light", "rem"]
    assert [r["is_nap"] for r in rows] == ["0", "1", "0"]
    assert list(prediction.labels) == ["light", "wake", "wake"]
    assert list(prediction.raw_labels) == ["light", "light", "rem"]


@pytest.mark.parametrize(
    "text",
    [
        "t,x,y,z\n0,0.1,0.2,0.9\n0.1,0.1,0.2,0.9\n",
        "0,0.1,0.2,0.9\n0.1,0.1,0.2,0.9\n",
        "0 0.1 0.2 0.9\n0.1 0.1 0.2 0.9\n",
        "timestamp x y z extra\n0 0.1 0.2 0.9 7\n0.1 0.1 0.2 0.9 7\n",
    ],
)
def test_reads_comma_or_space_separated_files_with_or_without_a_header(tmp_path, text) -> None:
    path = tmp_path / "night.txt"
    path.write_text(text)

    accel = read_accelerometer(path)

    np.testing.assert_allclose(accel, [[0, 0.1, 0.2, 0.9], [0.1, 0.1, 0.2, 0.9]])


@pytest.mark.parametrize(
    ("text", "time_unit", "expected"),
    [
        # Milliseconds from the header, even for a clock starting at 0.
        ("timestampMs,x,y,z\n0,0.1,0.2,0.9\n20,0.1,0.2,0.9\n", None, [0.0, 0.02]),
        # Milliseconds and microseconds from Unix-time magnitude.
        ("t,x,y,z\n1793022216141,0.1,0.2,0.9\n1793022216161,0.1,0.2,0.9\n", None,
         [1793022216.141, 1793022216.161]),
        ("1793022216141000 0.1 0.2 0.9\n1793022216161000 0.1 0.2 0.9\n", None,
         [1793022216.141, 1793022216.161]),
        # Unix seconds stay seconds.
        ("t,x,y,z\n1793022216.141,0.1,0.2,0.9\n1793022216.161,0.1,0.2,0.9\n", None,
         [1793022216.141, 1793022216.161]),
        # An explicit unit overrides detection.
        ("t,x,y,z\n0,0.1,0.2,0.9\n20,0.1,0.2,0.9\n", "ms", [0.0, 0.02]),
        ("t,x,y,z\n0,0.1,0.2,0.9\n20000,0.1,0.2,0.9\n", "us", [0.0, 0.02]),
        ("timestampMs,x,y,z\n0,0.1,0.2,0.9\n20,0.1,0.2,0.9\n", "s", [0.0, 20.0]),
    ],
)
def test_reads_timestamps_in_seconds_milliseconds_or_microseconds(
    tmp_path, text, time_unit, expected
) -> None:
    path = tmp_path / "night.csv"
    path.write_text(text)

    accel = read_accelerometer(path, time_unit)

    np.testing.assert_allclose(accel[:, 0], expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(accel[:, 1:], [[0.1, 0.2, 0.9]] * 2)


def test_rejects_an_unknown_time_unit(tmp_path) -> None:
    path = tmp_path / "night.csv"
    path.write_text("0,0.1,0.2,0.9\n")
    with pytest.raises(ValueError, match="time_unit"):
        read_accelerometer(path, "min")


def test_origin_defaults_to_the_first_sample_and_unix_snaps_to_whole_epochs() -> None:
    assert _resolve_origin(None, 1_700_000_012.5) == 1_700_000_012.5
    assert _resolve_origin("unix", 1_700_000_012.5) == 1_700_000_010.0
    assert _resolve_origin(5.0, 12.5) == 5.0
    with pytest.raises(ValueError):
        _resolve_origin("psg", 0.0)

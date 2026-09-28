"""The raw-accelerometer inference helper: file reading, epoch timing, output shape."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from autofish.predict import Prediction, _resolve_origin, read_accelerometer


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
    assert [r["is_gap"] for r in rows] == ["0", "1", "0"]
    assert rows[2]["start_s"] == "160.000"
    assert float(rows[0]["wake"]) == pytest.approx(0.7)


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


def test_origin_defaults_to_the_first_sample_and_unix_snaps_to_whole_epochs() -> None:
    assert _resolve_origin(None, 1_700_000_012.5) == 1_700_000_012.5
    assert _resolve_origin("unix", 1_700_000_012.5) == 1_700_000_010.0
    assert _resolve_origin(5.0, 12.5) == 5.0
    with pytest.raises(ValueError, match="origin"):
        _resolve_origin("psg", 0.0)

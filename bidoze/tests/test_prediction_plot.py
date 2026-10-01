"""Prediction hypnograms: CSV round trip, the three layouts, and the CLI."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from bidoze.predict import Prediction, read_prediction_csv

pytest.importorskip("matplotlib")
from bidoze.reporting.prediction_plot import plot_prediction  # noqa: E402

STAGES = ("wake", "light", "deep", "rem")


def _filtered(n: int = 240) -> Prediction:
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(4), size=n)
    raw = np.array(STAGES, dtype=object)[probabilities.argmax(-1)]
    is_sib = rng.random(n) > 0.2
    stage = raw.copy()
    stage[(raw != "wake") & ~is_sib] = "wake"
    is_gap = np.zeros(n, dtype=bool)
    is_gap[100:110] = True
    stage[is_gap] = raw[is_gap] = "gap"
    is_spt = np.zeros(n, dtype=bool)
    is_spt[20:180] = True
    return Prediction(
        epoch_start_s=1_774_800_000.0 + 30.0 * np.arange(n),
        probabilities=probabilities,
        stage_names=STAGES,
        is_gap=is_gap,
        stage=stage,
        stage_raw=raw,
        is_nap=((stage != "wake") & (stage != "gap") & ~is_spt).astype(np.int8),
        is_sib=is_sib,
        is_spt=is_spt,
    )


def test_csv_round_trip_keeps_stages_and_ggir_flags(tmp_path) -> None:
    prediction = _filtered()

    back = read_prediction_csv(prediction.to_csv(tmp_path / "night.csv"))

    assert list(back.labels) == list(prediction.labels)
    assert list(back.raw_labels) == list(prediction.raw_labels)
    np.testing.assert_array_equal(back.is_sib, prediction.is_sib)
    np.testing.assert_array_equal(back.is_spt, prediction.is_spt)
    np.testing.assert_array_equal(back.is_gap, prediction.is_gap)
    np.testing.assert_array_equal(back.nap_flags, prediction.nap_flags)
    np.testing.assert_allclose(back.probabilities, prediction.probabilities, atol=1e-4)


def test_unfiltered_csv_has_empty_ggir_flags_and_reads_back_as_none(tmp_path) -> None:
    prediction = Prediction(
        epoch_start_s=30.0 * np.arange(3),
        probabilities=np.tile([0.1, 0.6, 0.2, 0.1], (3, 1)),
        stage_names=STAGES,
        is_gap=np.zeros(3, dtype=bool),
    )

    back = read_prediction_csv(prediction.to_csv(tmp_path / "night.csv"))

    assert back.is_sib is None and back.is_spt is None


@pytest.mark.parametrize("with_psg", [False, True])
def test_filtered_hypnogram_is_written(tmp_path, with_psg) -> None:
    prediction = _filtered()
    psg = list(prediction.raw_labels) if with_psg else None

    path = plot_prediction(prediction, tmp_path / "night.png", psg=psg)

    assert path.stat().st_size > 10_000


def test_unfiltered_hypnogram_draws_only_the_raw_row(tmp_path) -> None:
    prediction = Prediction(
        epoch_start_s=30.0 * np.arange(120),
        probabilities=np.tile([0.1, 0.6, 0.2, 0.1], (120, 1)),
        stage_names=STAGES,
        is_gap=np.zeros(120, dtype=bool),
    )

    assert plot_prediction(prediction, tmp_path / "night.png").is_file()


def test_psg_must_match_the_epoch_count(tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_prediction(_filtered(), tmp_path / "night.png", psg=["wake"] * 3)


def test_hypnogram_command_draws_from_prediction_csvs(tmp_path, monkeypatch) -> None:
    from bidoze import cli

    prediction = _filtered()
    prediction.to_csv(tmp_path / "night.csv")
    (tmp_path / "night_summary.csv").write_text("subject_id,tst_minutes\nnight,1\n")
    out = tmp_path / "plots"
    monkeypatch.setattr(
        sys, "argv", ["bidoze", "hypnogram", str(tmp_path / "*.csv"), "-o", str(out)]
    )

    assert cli.main() == 0
    assert sorted(p.name for p in out.iterdir()) == ["night.png"]

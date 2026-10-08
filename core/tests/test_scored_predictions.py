"""Per-epoch predictions saved for scored subjects: what a CSV holds and how it reads back."""
from __future__ import annotations

import numpy as np
from pisces_lite.cv import SubjectRef

from arcasleep.data.bundle import CacheBundle
from arcasleep.data.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep.evaluation.predictions import scored_prediction, write_scored_prediction
from arcasleep.predict import read_prediction_csv

FRAMES_PER_EPOCH = 3


def _bundle(epochs: int, *, gap_epochs=(), anglez: np.ndarray | None = None) -> CacheBundle:
    X = np.zeros((1, epochs * FRAMES_PER_EPOCH, 4, 2))
    for k in gap_epochs:
        X[0, k * FRAMES_PER_EPOCH : (k + 1) * FRAMES_PER_EPOCH] = SPECTROGRAM_PADDING_VALUE
    return CacheBundle(
        X=X,
        y=np.zeros((1, epochs), dtype=np.int64),
        datasets=np.array(["CohortB"]),
        subject_ids=np.array(["s1"]),
        anglez_5s=anglez,
    )


def _scored(epochs: int, padded: int):
    """Padded truth (epoch 1 unscored) and a five-class softmax, as evaluate_subject gives."""
    y_true = np.full(padded, -1)
    y_true[:epochs] = np.arange(epochs) % 4
    y_true[1] = -1
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(5), size=padded)
    return y_true, proba


def test_a_scored_subject_keeps_its_own_epochs_and_the_models_probabilities() -> None:
    y_true, proba = _scored(5, padded=8)

    prediction = scored_prediction(_bundle(5, gap_epochs=[2]), y_true, proba, num_classes=4)

    assert prediction.probabilities.shape == (5, 4)
    np.testing.assert_allclose(
        prediction.probabilities, proba[:5, :4] / proba[:5, :4].sum(-1, keepdims=True)
    )
    np.testing.assert_allclose(prediction.gap_head, proba[:5, 4])
    assert list(prediction.is_gap) == [False, False, True, False, False]
    assert list(prediction.psg) == ["wake", "", "deep", "rem", "wake"]
    np.testing.assert_allclose(prediction.epoch_start_s, 30.0 * np.arange(5))


def test_a_recording_longer_than_the_context_keeps_the_epochs_that_were_scored() -> None:
    y_true, proba = _scored(8, padded=8)

    prediction = scored_prediction(_bundle(12), y_true, proba, num_classes=4)

    assert prediction.probabilities.shape[0] == 8


def _still_night_grid(hours: float) -> np.ndarray:
    from arcasleep import ggir

    fs = 4
    t = np.arange(int(hours * 3600 * fs)) / fs
    accel = np.column_stack([t, np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])
    bin_start, anglez = ggir.anglez_bins(accel)
    return ggir.epoch_bins(bin_start, anglez, int(np.ceil((bin_start[-1] + 5) / 30)))


def test_ggir_columns_come_from_the_cached_anglez_and_are_empty_without_it() -> None:
    grid = _still_night_grid(4.0)
    epochs = grid.shape[0]
    y_true, proba = _scored(epochs, padded=epochs)

    judged = scored_prediction(_bundle(epochs, anglez=grid[None]), y_true, proba, num_classes=4)
    unjudged = scored_prediction(_bundle(epochs), y_true, proba, num_classes=4)

    assert judged.is_sib.shape == judged.is_spt.shape == (epochs,)
    assert judged.is_sib.any()
    assert judged.stage is not None
    assert unjudged.is_sib is None and unjudged.is_spt is None and unjudged.stage is None
    assert list(unjudged.labels) == list(unjudged.raw_labels)


def test_the_saved_csv_reads_back_with_its_psg(tmp_path) -> None:
    y_true, proba = _scored(5, padded=8)
    ref = SubjectRef("CohortB", "s1")

    path = write_scored_prediction(tmp_path, ref, _bundle(5, gap_epochs=[2]), y_true, proba, 4)
    back = read_prediction_csv(path)

    assert path.name == "CohortB_s1.csv"
    assert list(back.psg) == ["wake", "", "deep", "rem", "wake"]
    assert list(back.is_gap) == [False, False, True, False, False]
    np.testing.assert_allclose(back.gap_head, proba[:5, 4], atol=1e-4)
    np.testing.assert_allclose(back.probabilities.sum(-1), 1.0, atol=1e-3)


def test_a_prediction_without_psg_reads_back_without_it(tmp_path) -> None:
    from arcasleep.predict import Prediction

    plain = Prediction(
        epoch_start_s=np.array([0.0, 30.0]),
        probabilities=np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1]]),
        stage_names=("wake", "light", "deep", "rem"),
        is_gap=np.array([False, False]),
    )

    assert read_prediction_csv(plain.to_csv(tmp_path / "night.csv")).psg is None

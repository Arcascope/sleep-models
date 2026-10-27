"""Scoring a held-out recording: one fixed window, or overlapping windows over the whole recording."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arcasleep.data.constants import GAP_LABEL
from arcasleep.training.driver import OUTSIDE_PSG_LABEL, evaluate_subject, softmax

FRAMES = 3
CLASSES = 5


class EpochwiseModel:
    """Logits that depend only on the epoch's own frames, so any window layout gives the same output."""

    config = SimpleNamespace(frames_per_epoch=FRAMES)

    def returns_logits(self) -> bool:
        return True

    def infer(self, x: np.ndarray) -> np.ndarray:
        feature = x[..., 0, 0]  # (B, T * frames)
        per_epoch = feature.reshape(feature.shape[0], -1, FRAMES).mean(axis=-1)
        classes = np.arange(CLASSES)
        return -np.abs(per_epoch[..., None] - classes)  # (B, T, CLASSES)


def _bundle(y: np.ndarray) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    stage = rng.integers(0, 4, size=y.shape[0]).astype(float)
    X = np.repeat(stage, FRAMES)[:, None, None] * np.ones((1, 4, 2))
    return SimpleNamespace(X=X[None], y_for_n_classes=lambda n: y[None]), stage


def test_a_windowed_model_scores_every_epoch_of_a_long_recording_against_psg_labels_only() -> None:
    y = np.full(50, OUTSIDE_PSG_LABEL)
    y[15:40] = np.arange(25) % 4
    y[20] = GAP_LABEL
    bundle, stage = _bundle(y)

    y_true, proba = evaluate_subject(
        EpochwiseModel(), bundle, target_x_length=16, target_y_length=16, num_classes=4, window_step=2
    )

    assert y_true.shape == (50,) and proba.shape == (50, CLASSES)
    assert (y_true[:15] == -1).all() and (y_true[40:] == -1).all() and y_true[20] == -1
    scored = np.flatnonzero(y_true >= 0)
    assert set(scored) == set(range(15, 40)) - {20}
    np.testing.assert_array_equal(y_true[scored], y[scored])
    np.testing.assert_allclose(proba[:, :], softmax(EpochwiseModel().infer(np.repeat(stage, FRAMES)[None, :, None, None] * np.ones((1, 1, 4, 2)))[0]), atol=1e-5)


def test_a_model_without_window_step_scores_one_padded_window_as_before() -> None:
    y = np.arange(10) % 4
    bundle, stage = _bundle(y)

    y_true, proba = evaluate_subject(
        EpochwiseModel(), bundle, target_x_length=16, target_y_length=16, num_classes=4
    )

    assert y_true.shape == (16,) and proba.shape == (16, CLASSES)
    np.testing.assert_array_equal(y_true[:10], y)
    assert (y_true[10:] == -1).all()  # padding folds into the gap class, which is not scored

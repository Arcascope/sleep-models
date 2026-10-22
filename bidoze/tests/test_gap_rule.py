"""The gap rule (``gap_from_row``): gap is called from the gap row only, absent epochs are masked."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402

from arcasleep.predict import _stage_and_gap_probabilities  # noqa: E402
from arcasleep.training.balanced import Patience  # noqa: E402
from arcasleep_bidoze.adult import StackedModel  # noqa: E402
from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel  # noqa: E402

T, F, C = 48, 7, 2
GAP = 4


def small_config(**changes) -> BiDozeNetConfig:
    base = dict(num_classes=4, score_gap=True, gap_from_row=True, d_model=16, n_heads=2,
                n_transformer_layers=1, d_ff=16, conv_kernels=[3, 7], dropout=0.0)
    return BiDozeNetConfig.from_dict({**base, **changes})


def recording(rng: np.random.Generator, present: int, lead: int = 0) -> np.ndarray:
    """``(T, F + 1, C)``: ``present`` epochs with data after ``lead`` absent ones, absent after."""
    x = np.zeros((T, F + 1, C), dtype=np.float32)
    x[:, :F] = rng.normal(size=(T, F, C))
    x[:, -1] = 1.0
    x[lead:lead + present, -1] = 0.0
    return x


def probabilities(model: BiDozeNetModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.nn.softmax(model.infer(x), axis=-1))


def test_gap_from_row_needs_score_gap() -> None:
    with pytest.raises(ValueError):
        small_config(score_gap=False)


def test_gap_from_row_is_off_by_default_and_survives_a_config_round_trip() -> None:
    assert BiDozeNetConfig.from_dict({"num_classes": 4, "score_gap": True}).gap_from_row is False
    config = small_config()
    assert BiDozeNetConfig.from_dict(config.to_dict()).gap_from_row is True


def test_the_gap_class_is_called_on_absent_epochs_only() -> None:
    rng = np.random.default_rng(0)
    x = np.stack([recording(rng, 30, lead=5), recording(rng, 20)])
    model = BiDozeNetModel(small_config())
    model._ensure_built((T, F + 1, C), seed=0)

    proba = probabilities(model, x)
    absent = x[:, :, -1, 0] >= 1.0

    assert proba.shape == (2, T, 5)
    assert np.all(proba[absent, GAP] == 1.0)
    assert np.all(proba[~absent, GAP] == 0.0)
    assert np.all(np.argmax(proba[~absent], axis=-1) != GAP)
    np.testing.assert_allclose(proba[~absent].sum(axis=-1), 1.0, atol=1e-5)


def test_absent_epochs_before_the_recording_do_not_change_the_present_epochs() -> None:
    rng = np.random.default_rng(1)
    night = recording(rng, 30)
    shifted = np.tile(night[-1:], (T, 1, 1))
    shifted[10:40] = night[:30]
    model = BiDozeNetModel(small_config())
    model._ensure_built((T, F + 1, C), seed=0)

    a = probabilities(model, night[None])[0]
    b = probabilities(model, shifted[None])[0]

    np.testing.assert_allclose(a[:30], b[10:40], atol=1e-5)


def test_a_balanced_fit_with_the_gap_rule_runs() -> None:
    rng = np.random.default_rng(2)
    x = np.stack([recording(rng, 40), recording(rng, 30, lead=4)])
    y = np.full((2, T), -2, dtype=np.int32)
    y[0, :40] = rng.integers(0, 4, 40)
    y[1, 4:34] = rng.integers(0, 4, 30)
    model = BiDozeNetModel(small_config())

    history = model.fit_balanced(
        x, y, x, y, batch_size=2, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
        patience=Patience(lr_drop=10, early_stop=20, unit="steps"), max_epochs=2, fit_seed=0,
    )

    assert np.all(np.isfinite(history["train_loss"]))
    assert np.all(probabilities(model, x)[x[:, :, -1, 0] < 1.0, GAP] == 0.0)


def test_stacked_seeds_keep_gap_and_stage_probabilities_finite() -> None:
    rng = np.random.default_rng(3)
    x = recording(rng, 30, lead=3)[None]
    models = []
    for seed in (0, 1):
        model = BiDozeNetModel(small_config())
        model._ensure_built((T, F + 1, C), seed=seed)
        models.append(model)

    proba = np.asarray(jax.nn.softmax(StackedModel(models).infer(x), axis=-1))[0]
    absent = x[0, :, -1, 0] >= 1.0

    logits = StackedModel(models).infer(x)[0]
    stages, _ = _stage_and_gap_probabilities(logits, 4)
    assert np.all(np.isfinite(proba))
    assert np.all(np.isfinite(stages))
    np.testing.assert_allclose(stages.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(proba[absent, GAP] == 1.0)
    assert np.all(proba[~absent, GAP] < 1e-30)

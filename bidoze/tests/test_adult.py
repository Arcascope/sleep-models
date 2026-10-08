from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arcasleep_bidoze.adult import (
    DEEP,
    GAP,
    LIGHT,
    REM,
    WAKE,
    adult_probabilities,
    adult_stage_divisor,
    build_adult_bundle,
    effective_weights,
    hybrid_calls,
    stage_counts,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _proba(calls: list[int], rng: np.random.Generator) -> np.ndarray:
    p = rng.random((len(calls), 5)) * 0.1
    p[np.arange(len(calls)), calls] = 1.0
    return p / p.sum(axis=1, keepdims=True)


def test_the_wake_source_decides_sleep_and_wake_and_the_stage_source_the_stage() -> None:
    rng = np.random.default_rng(0)
    wake = _proba([WAKE, LIGHT, DEEP, GAP, REM, WAKE], rng)
    stage = _proba([REM, DEEP, LIGHT, LIGHT, LIGHT, DEEP], rng)

    calls = hybrid_calls(wake, stage)

    assert calls.tolist() == [WAKE, DEEP, LIGHT, GAP, LIGHT, WAKE]


def test_the_divisor_moves_stage_calls_without_touching_wake_calls() -> None:
    wake = np.array([[0.1, 0.9, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0, 0.0]])
    stage = np.array([[0.0, 0.5, 0.4, 0.1, 0.0], [0.0, 0.5, 0.4, 0.1, 0.0]])

    assert hybrid_calls(wake, stage).tolist() == [LIGHT, WAKE]
    assert hybrid_calls(wake, stage, np.array([1.0, 2.0, 1.0, 1.0])).tolist() == [DEEP, WAKE]


def test_the_divisor_is_one_when_training_and_adult_stage_mixes_match_without_balancing() -> None:
    nights = np.array([[100, 400, 150, 200], [80, 420, 120, 230]])

    divisor = adult_stage_divisor(nights, nights, beta=0.0)

    np.testing.assert_allclose(divisor, 1.0)


def test_a_rarer_stage_gets_a_larger_effective_weight() -> None:
    nights = np.array([[100, 600, 50, 200]] * 4)

    w = effective_weights(nights, beta=0.999)

    assert w[DEEP] > w[WAKE] > w[REM] > w[LIGHT]


def test_stage_counts_merge_n1_and_n2_into_light() -> None:
    assert stage_counts(np.array([0, 1, 2, 2, 3, 5, 5, -1])).tolist() == [1, 3, 1, 2]


def test_adult_probabilities_keep_the_models_wake_and_sum_to_one() -> None:
    proba = np.random.default_rng(1).dirichlet(np.ones(4), size=6)

    p = adult_probabilities(proba, np.array([1.0, 2.0, 0.5, 1.0]))

    np.testing.assert_allclose(p[:, WAKE], proba[:, WAKE])
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def _release(directory: Path, seed: int) -> Path:
    from arcasleep.models import MODEL_TYPE_KEY
    from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel

    config = {"num_classes": 4, "score_gap": True, "d_model": 16, "n_heads": 2,
              "n_transformer_layers": 1, "d_ff": 32, "conv_kernels": [3], "dropout": 0.0}
    model = BiDozeNetModel(BiDozeNetConfig.from_dict(config))
    model._ensure_built((120, 61, 2), seed=seed)
    directory.mkdir(parents=True)
    model.save_weights_safetensors(directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({MODEL_TYPE_KEY: "bidoze", **config}))
    (directory / "preprocessing.json").write_text((CONFIG_DIR / "inputs.json").read_text())
    return directory


def test_an_adult_bundle_predicts_a_recording(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("senpy")
    pytest.importorskip("safetensors")
    import arcasleep.artifacts.publish as publish
    from arcasleep.predict import load_predictor

    monkeypatch.setattr(publish, "verify_bundle", lambda directory: None)
    bundle = build_adult_bundle(
        tmp_path / "adult",
        [_release(tmp_path / "s0", 1), _release(tmp_path / "s1", 2)],
        [1.0, 0.8, 1.6, 0.9],
        version="v0.0.0",
    )

    fs, minutes = 25.0, 47
    t = np.arange(int(fs * 60 * minutes)) / fs
    xyz = np.random.default_rng(5).normal(scale=0.05, size=(t.size, 3)) + [0.0, 0.0, 1.0]
    result = load_predictor(bundle).predict(np.column_stack([t, xyz]))

    assert result.probabilities.shape == (minutes * 2, 4)
    assert len(result.labels) == minutes * 2


def test_stacked_seeds_give_the_mean_of_their_probabilities(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    import jax

    from arcasleep.artifacts.publish import load_bundle_model
    from arcasleep_bidoze.adult import StackedModel

    models = [load_bundle_model(_release(tmp_path / f"s{k}", k)) for k in range(3)]
    x = np.random.default_rng(2).normal(size=(2, 120, 61, 2)).astype(np.float32)

    stacked = np.asarray(jax.nn.softmax(StackedModel(models).infer(x), axis=-1))

    mean = np.mean([np.asarray(jax.nn.softmax(m.infer(x), axis=-1)) for m in models], axis=0)
    np.testing.assert_allclose(stacked, mean, rtol=1e-4, atol=1e-6)

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


CONFIG = {"num_classes": 4, "score_gap": True, "d_model": 16, "n_heads": 2,
          "n_transformer_layers": 1, "d_ff": 32, "conv_kernels": [3], "dropout": 0.0}
DIVISOR = {"light": 0.5, "deep": 2.0, "rem": 1.25}


def _models(seeds: list[int]) -> list:
    from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel

    models = []
    for seed in seeds:
        model = BiDozeNetModel(BiDozeNetConfig.from_dict(CONFIG))
        model._ensure_built((120, 61, 2), seed=seed)
        models.append(model)
    return models


def _release(directory: Path, seed: int) -> Path:
    """A single-model bundle: unprefixed weights and no ``members``."""
    from arcasleep.models import MODEL_TYPE_KEY

    directory.mkdir(parents=True)
    _models([seed])[0].save_weights_safetensors(directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({MODEL_TYPE_KEY: "bidoze", **CONFIG}))
    (directory / "preprocessing.json").write_text((CONFIG_DIR / "inputs.json").read_text())
    return directory


def _flat_bundle(directory: Path, seeds: list[int], stage_divisor: dict | None = DIVISOR) -> Path:
    """A flat bundle of one member per seed, with the divisor in ``config.json``."""
    from arcasleep.artifacts.publish import _save_member_weights
    from arcasleep.models import MODEL_TYPE_KEY

    directory.mkdir(parents=True)
    _save_member_weights(_models(seeds), directory / "model.safetensors")
    config = {MODEL_TYPE_KEY: "bidoze", **CONFIG, "members": len(seeds)}
    if stage_divisor is not None:
        config["stage_divisor"] = stage_divisor
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "preprocessing.json").write_text((CONFIG_DIR / "inputs.json").read_text())
    return directory


def _recording(minutes: int = 47) -> np.ndarray:
    fs = 25.0
    t = np.arange(int(fs * 60 * minutes)) / fs
    xyz = np.random.default_rng(5).normal(scale=0.05, size=(t.size, 3)) + [0.0, 0.0, 1.0]
    return np.column_stack([t, xyz])


@pytest.mark.parametrize("members", [1, 3])
def test_a_flat_bundle_scores_the_mean_of_its_members_and_decodes_with_the_divisor(
    tmp_path: Path, members: int
) -> None:
    pytest.importorskip("senpy")
    pytest.importorskip("safetensors")
    from arcasleep.divisor import divisor_vector
    from arcasleep.predict import Predictor

    seeds = list(range(members))
    accel = _recording()
    result = Predictor(_flat_bundle(tmp_path / "flat", seeds)).predict(accel, ggir_sleep_filter=False)

    # Each member alone, as its own single-model bundle: its 4-stage probabilities and gap head.
    alone = [
        Predictor(_release(tmp_path / f"s{k}", k)).predict(accel, return_gap_head=True, ggir_sleep_filter=False)
        for k in seeds
    ]
    five_way = np.mean([a.probabilities * (1.0 - a.gap_head)[:, None] for a in alone], axis=0)
    mean = five_way / five_way.sum(axis=1, keepdims=True)
    divisor = divisor_vector(DIVISOR)
    expected = adult_probabilities(mean, divisor)
    calls = hybrid_calls(mean, mean, divisor)

    np.testing.assert_allclose(result.probabilities, expected, rtol=0, atol=1e-4)
    # An epoch whose two best divided stages are this close can fall either way between GPU and CPU rounding.
    divided = np.sort(mean[:, LIGHT : REM + 1] / divisor[LIGHT : REM + 1], axis=1)
    clear = (calls == WAKE) | (divided[:, -1] - divided[:, -2] > 1e-3)
    names = np.asarray(result.stage_names, dtype=object)[calls]
    assert (result.raw_labels[clear & ~result.is_gap] == names[clear & ~result.is_gap]).all()


def test_a_bundle_without_a_divisor_labels_epochs_by_the_argmax(tmp_path: Path) -> None:
    pytest.importorskip("senpy")
    pytest.importorskip("safetensors")
    from arcasleep.predict import Predictor

    result = Predictor(_flat_bundle(tmp_path / "flat", [0, 1], stage_divisor=None)).predict(
        _recording(), ggir_sleep_filter=False
    )

    keep = ~result.is_gap
    assert (result.raw_labels[keep] == np.asarray(result.stage_names, dtype=object)[result.stages][keep]).all()


def test_stacked_seeds_give_the_mean_of_their_probabilities(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    import jax

    from arcasleep.artifacts.publish import load_bundle_model
    from arcasleep_bidoze.adult import StackedModel

    models = [load_bundle_model(_release(tmp_path / f"s{k}", k)) for k in range(3)]
    x = np.random.default_rng(2).normal(size=(2, 120, 61, 2)).astype(np.float32)

    stacked = np.asarray(jax.nn.softmax(StackedModel(models).infer(x), axis=-1))

    mean = np.mean([np.asarray(jax.nn.softmax(m.infer(x), axis=-1)) for m in models], axis=0)
    # A vmapped batch and separate calls round differently on GPU.
    np.testing.assert_allclose(stacked, mean, rtol=0, atol=1e-4)

"""The UNet network, its loss and sampler, and its saved forms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from arcasleep.models import MODEL_TYPE_KEY, checkpoint_family, get_family  # noqa: E402
from arcasleep_unet.model import UNetConfig, UNetModel  # noqa: E402
from arcasleep_unet.model.definition import (  # noqa: E402
    UNetModule,
    balanced_cross_entropy,
    sample_balanced_segments,
)

RELEASE_CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "model.json").read_text()
)
SMALL = {"m_blocks": 2, "c_u": 4, "head_hidden": 8, "frames_per_epoch": 3, "kernel_temporal": 4}


def _output_shape(config: UNetConfig, input_shape: tuple[int, ...]) -> tuple[int, ...]:
    module = UNetModule(config)
    sample = jax.ShapeDtypeStruct(input_shape, jnp.float32)

    def forward(x):
        variables = module.init(jax.random.PRNGKey(0), x, train=False)
        return module.apply(variables, x, train=False)

    return tuple(jax.eval_shape(forward, sample).shape)


def _built(config: dict, n_frames: int = 12, f_bins: int = 5, seed: int = 0) -> UNetModel:
    model = UNetModel(UNetConfig.from_dict(config))
    input_shape = (n_frames, f_bins, 2)
    model.variables = model._init_variables(jax.random.PRNGKey(seed), input_shape)
    model.input_shape = input_shape
    # Non-trivial BatchNorm statistics, so a round trip that dropped them would show.
    rng = np.random.default_rng(seed)
    model.variables["batch_stats"] = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(rng.uniform(0.5, 1.5, leaf.shape), dtype=leaf.dtype),
        model.variables["batch_stats"],
    )
    return model


def _features(n: int = 2, n_frames: int = 12, f_bins: int = 5) -> np.ndarray:
    return np.random.default_rng(1).normal(size=(n, n_frames, f_bins, 2)).astype(np.float32)


def test_release_config_emits_one_row_per_epoch_with_a_gap_logit() -> None:
    config = UNetConfig.from_dict(RELEASE_CONFIG)
    assert _output_shape(config, (1, 1024 * 15, 60, 2)) == (1, 1024, 5)


def test_without_score_gap_the_head_has_only_the_stage_logits() -> None:
    config = UNetConfig.from_dict({**RELEASE_CONFIG, "score_gap": False})
    assert _output_shape(config, (1, 1024 * 15, 60, 2)) == (1, 1024, 4)


def test_a_frequency_axis_that_is_not_a_power_of_two_runs() -> None:
    model = _built(SMALL, f_bins=5)
    assert model.infer(_features(f_bins=5)).shape == (2, 4, 5)


def test_unknown_config_keys_are_refused() -> None:
    with pytest.raises(ValueError):
        UNetConfig.from_dict({**RELEASE_CONFIG, "dropout": 0.1})


def test_padded_epoch_multiple_makes_the_frame_count_divisible() -> None:
    config = UNetConfig.from_dict(RELEASE_CONFIG)
    assert config.padded_epoch_multiple == 1024
    assert (config.padded_epoch_multiple * config.frames_per_epoch) % config.temporal_multiple == 0


def test_pisces_round_trip_reproduces_logits(tmp_path) -> None:
    model = _built(SMALL)
    path = tmp_path / "model.pisces"
    model.savez(path)

    restored = UNetModel.loadz(path)

    np.testing.assert_array_equal(restored.infer(_features()), model.infer(_features()))
    assert checkpoint_family(path) == "unet"
    assert get_family("unet").load_checkpoint(path).input_shape == model.input_shape


def test_safetensors_release_round_trip_reproduces_logits(tmp_path) -> None:
    pytest.importorskip("safetensors")
    model = _built(SMALL)
    weights = tmp_path / "model.safetensors"
    model.save_weights_safetensors(weights)

    released = get_family("unet").load_release({MODEL_TYPE_KEY: "unet", **SMALL}, weights)

    np.testing.assert_array_equal(released.infer(_features()), model.infer(_features()))


def test_weights_from_another_architecture_are_refused(tmp_path) -> None:
    pytest.importorskip("safetensors")
    weights = tmp_path / "model.safetensors"
    _built(SMALL).save_weights_safetensors(weights)

    other = UNetModel(UNetConfig.from_dict({**SMALL, "c_u": 8}))
    with pytest.raises(ValueError):
        other.load_weights_safetensors(weights)


def test_balanced_loss_is_scaled_cross_entropy_when_classes_are_balanced() -> None:
    beta = 0.999
    logits = jnp.asarray(np.random.default_rng(2).normal(size=(2, 3, 3)), dtype=jnp.float32)
    labels = jnp.asarray([[0, 1, 2], [2, 1, 0]])  # every class twice
    weight = (1.0 - beta) / (1.0 - beta**2)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    plain = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).mean()

    loss = balanced_cross_entropy(logits, labels, num_classes=3, beta=beta)

    # The weight is computed in float32, where 1 - beta**n loses a few digits.
    np.testing.assert_allclose(loss, weight * plain, rtol=1e-4)


def test_masked_epochs_add_nothing_to_the_balanced_loss() -> None:
    logits = jnp.asarray(np.random.default_rng(3).normal(size=(1, 4, 3)), dtype=jnp.float32)
    scored = jnp.asarray([[0, 1, 2, 1]])
    masked = jnp.asarray([[0, 1, 2, -1]])
    other_logits = logits.at[0, 3].set(100.0)

    def total(lg, lb):
        return balanced_cross_entropy(lg, lb, num_classes=3, beta=0.999) * lb.size

    assert float(total(logits, masked)) == pytest.approx(float(total(other_logits, masked)))
    assert float(total(logits, masked)) != pytest.approx(float(total(logits, scored)))


def _nights() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(4)
    nights = []
    for n_epochs in (10, 9, 12):
        y = rng.integers(0, 3, size=n_epochs).astype(np.int32)
        nights.append((rng.normal(size=(n_epochs * 3, 5, 2)).astype(np.float32), y))
    # Stage 3 occurs in only one night.
    nights[2][1][[5, 8]] = 3
    return nights


class _RecordingRng:
    """A generator that remembers the target stage of every segment it samples."""

    def __init__(self, seed: int, num_classes: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.targets: list[int] = []

    def integers(self, high):
        value = self.rng.integers(high)
        # The nights' start ranges and count are chosen to differ from num_classes.
        if high == self.num_classes:
            self.targets.append(int(value))
        return value


def test_each_segment_holds_its_target_stage() -> None:
    rng = _RecordingRng(0, num_classes=4)

    xs, ys = sample_balanced_segments(
        _nights(), rng, n_segments=40, segment_epochs=4, num_classes=4, frames_per_epoch=3
    )

    assert xs.shape == (40, 12, 5, 2) and ys.shape == (40, 4)
    assert len(rng.targets) == 40
    for target, row in zip(rng.targets, ys, strict=True):
        assert target in row


def test_sampling_is_deterministic_under_a_seed() -> None:
    def draw():
        return sample_balanced_segments(
            _nights(), np.random.default_rng(7), n_segments=8, segment_epochs=4,
            num_classes=4, frames_per_epoch=3,
        )

    (xs, ys), (xs_again, ys_again) = draw(), draw()
    np.testing.assert_array_equal(xs, xs_again)
    np.testing.assert_array_equal(ys, ys_again)


def test_a_short_night_pads_its_segment_with_absent_frames_and_gap_labels() -> None:
    night = (np.zeros((2 * 3, 5, 2), dtype=np.float32), np.zeros(2, dtype=np.int32))

    xs, ys = sample_balanced_segments(
        [night], np.random.default_rng(0), n_segments=1, segment_epochs=4,
        num_classes=1, frames_per_epoch=3,
    )

    assert np.all(xs[0, 6:] == -1.0)
    assert list(ys[0]) == [0, 0, -2, -2]


def test_a_tiny_fit_runs_and_keeps_its_best_epoch() -> None:
    nights = _nights()
    model = UNetModel(UNetConfig.from_dict(SMALL))

    history = model.fit(
        nights[:2],
        nights[2:],
        segment_epochs=4,
        batch_size=2,
        initial_lr=1e-3,
        balance_beta=0.999,
        lr_drop_patience=1,
        early_stop_patience=2,
        max_epochs=3,
        fit_seed=0,
    )

    assert 1 <= len(history["train_loss"]) <= 3
    assert model.infer(_features(n=1)).shape == (1, 4, 5)

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arcasleep.data.batching import pool_frames_to_epochs
from arcasleep.data.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep_bidoze.model.definition import BiDozeNetConfig, FramePoolModule

FRAMES = 5
BINS, CHANNELS = 4, 2


def _frames(epochs: int, missing: list[tuple[int, int]]) -> np.ndarray:
    shape = (1, epochs * FRAMES, BINS, CHANNELS)
    x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    for epoch, frame in missing:
        x[0, epoch * FRAMES + frame] = SPECTROGRAM_PADDING_VALUE
    return x


def test_frame_pool_mean_and_missing_fraction_match_the_host_pool() -> None:
    epochs = 3
    x = _frames(epochs, missing=[(1, 0), (1, 3)])
    module = FramePoolModule(frames_per_epoch=FRAMES, features=6)
    params = module.init(jax.random.PRNGKey(0), jnp.asarray(x))
    out = np.asarray(module.apply(params, jnp.asarray(x)))

    host = pool_frames_to_epochs(x, np.zeros((1, epochs)))
    width = BINS * CHANNELS
    assert out.shape == (1, epochs, width + 1 + 6)
    host_mean = host[0, :, :BINS].reshape(epochs, width)
    np.testing.assert_allclose(out[0, :, :width], host_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[0, :, width], [0.0, 2 / FRAMES, 0.0], atol=1e-6)


def test_frame_pool_epoch_with_no_frames_gives_zero_mean_and_full_missing() -> None:
    x = _frames(2, missing=[(0, f) for f in range(FRAMES)])
    module = FramePoolModule(frames_per_epoch=FRAMES, features=3)
    params = module.init(jax.random.PRNGKey(0), jnp.asarray(x))
    out = np.asarray(module.apply(params, jnp.asarray(x)))
    width = BINS * CHANNELS
    np.testing.assert_allclose(out[0, 0, :width], 0.0)
    assert out[0, 0, width] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_pooling": "max"},
        {"frame_pooling": "conv", "frames_per_epoch": 1},
        {"frame_pooling": "conv", "frames_per_epoch": 15, "frame_pool_features": 0},
    ],
)
def test_invalid_frame_pooling_is_refused(overrides: dict) -> None:
    with pytest.raises(ValueError):
        BiDozeNetConfig.from_dict({"num_classes": 4, **overrides})

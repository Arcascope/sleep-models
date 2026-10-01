"""Building a recording's model input: pooled epochs or raw frames."""
from __future__ import annotations

import numpy as np
import pytest

from arcasleep.data.batching import (
    pad_recording_input,
    pool_frames_to_epochs,
    recording_input,
)
from arcasleep.data.constants import SPECTROGRAM_PADDING_VALUE


def _frames(n_epochs: int = 4, frames_per_epoch: int = 3) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n_epochs * frames_per_epoch, 5, 2))


def test_one_frame_per_epoch_is_the_pooled_surface_with_its_gap_row() -> None:
    X = _frames()

    pooled = recording_input(X, 4, frames_per_epoch=1)

    np.testing.assert_array_equal(pooled, pool_frames_to_epochs(X[None], np.zeros((1, 4)))[0])
    assert pooled.shape == (4, 6, 2)


def test_raw_frames_pass_through_and_pad_with_absent_frames() -> None:
    X = _frames()

    raw = recording_input(X, 4, frames_per_epoch=3)
    padded = pad_recording_input(raw, 6, frames_per_epoch=3)

    np.testing.assert_array_equal(raw, X)
    assert padded.shape == (18, 5, 2)
    np.testing.assert_array_equal(padded[:12], X)
    assert np.all(padded[12:] == SPECTROGRAM_PADDING_VALUE)
    assert pad_recording_input(raw, 2, frames_per_epoch=3).shape == (6, 5, 2)


def test_raw_frames_must_match_the_cache_frames_per_label() -> None:
    with pytest.raises(ValueError):
        recording_input(_frames(frames_per_epoch=3), 4, frames_per_epoch=15)

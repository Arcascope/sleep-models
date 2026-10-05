"""ETC wake score and per-night label alignment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arcasleep.data.alignment import (
    SHIFT_KEY,
    STYLES,
    AlignmentSettings,
    align_cache,
    estimate_offset,
    shift_labels,
)
from arcasleep.data.bundle import CacheBundle
from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR
from arcasleep.data.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep.etc import ETCKernel, epoch_energy, wake_score

FRAMES_PER_EPOCH = 3


def _night(epochs: int = 400, shift: int = 0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raw PSG labels with wake bouts, and a spectrogram whose energy rises ``shift`` epochs early.

    ``shift > 0`` is a PSG that runs late: the movement of a wake bout shows in
    the accelerometer ``shift`` epochs before the PSG scores it.
    """
    rng = np.random.default_rng(seed)
    labels = np.full(epochs, 2)
    for start in rng.choice(np.arange(30, epochs - 40), size=12, replace=False):
        labels[start : start + rng.integers(2, 8)] = 0
    awake = np.roll(labels == 0, -shift)
    energy = np.where(awake, 1.0, 0.0) + rng.normal(0, 0.05, epochs)
    X = np.repeat(energy, FRAMES_PER_EPOCH)[:, None, None] * np.ones((1, 4, 2)) / 4.0
    return X, labels


@pytest.mark.parametrize("window", [2, 0, 4])
def test_a_kernel_window_must_be_odd(window: int) -> None:
    with pytest.raises(ValueError):
        ETCKernel(window=window)


@pytest.mark.parametrize("window", [1, 3, 9, 71])
def test_smoothing_keeps_a_burst_where_it_happened(window: int) -> None:
    energy = np.zeros(50)
    energy[25] = 1.0

    assert wake_score(energy, ETCKernel(window, 2.0)).argmax() == 25


def test_a_long_window_reaches_the_kernels_own_spread() -> None:
    assert ETCKernel(1, 2.0).effective_std == 0.0
    assert ETCKernel(3, 2.0).effective_std < 1.0
    assert ETCKernel(71, 2.0).effective_std == pytest.approx(2.0, abs=1e-3)


def test_frames_without_data_are_left_out_of_an_epochs_energy() -> None:
    X = np.ones((3 * FRAMES_PER_EPOCH, 4, 2))
    X[0] = SPECTROGRAM_PADDING_VALUE                      # one missing frame in epoch 0
    X[2 * FRAMES_PER_EPOCH : 3 * FRAMES_PER_EPOCH] = SPECTROGRAM_PADDING_VALUE  # epoch 2 empty

    energy = epoch_energy(X, 3)

    np.testing.assert_allclose(energy, [4.0, 4.0, 4.0])


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("true_shift", [-3, 0, 2, 5])
def test_a_known_offset_is_recovered(style: str, true_shift: int) -> None:
    X, labels = _night(shift=true_shift)
    settings = AlignmentSettings(style=style)
    score = wake_score(epoch_energy(X, labels.shape[0]), settings.kernel)

    offset = estimate_offset(score, labels, settings)

    assert offset.shift == true_shift
    assert offset.dropped is None


@pytest.mark.parametrize("style", STYLES)
def test_a_night_with_no_signal_is_dropped(style: str) -> None:
    _, labels = _night()
    flat = np.random.default_rng(1).normal(0, 1, labels.shape[0])

    offset = estimate_offset(flat, labels, AlignmentSettings(style=style, min_peak=0.9))

    assert offset.shift is None
    assert offset.dropped


def test_a_best_offset_at_the_search_limit_is_dropped() -> None:
    X, labels = _night(shift=6)
    settings = AlignmentSettings(max_shift=6)
    score = wake_score(epoch_energy(X, labels.shape[0]), settings.kernel)

    offset = estimate_offset(score, labels, settings)

    assert offset.best_shift == 6
    assert offset.shift is None


def test_shifting_labels_moves_them_and_leaves_the_vacated_epochs_unscored() -> None:
    labels = np.arange(6)[None]

    np.testing.assert_array_equal(shift_labels(labels, 2), [[2, 3, 4, 5, -1, -1]])
    np.testing.assert_array_equal(shift_labels(labels, -1), [[-1, 0, 1, 2, 3, 4]])
    np.testing.assert_array_equal(shift_labels(labels, 0), labels)


def _write(cache_dir: Path, dataset: str, subject: str, X: np.ndarray, labels: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset}{SUBJECT_SEPARATOR}{subject}{CACHE_SUFFIX}"
    CacheBundle(
        X=X[None], y=labels[None], datasets=np.array([dataset]), subject_ids=np.array([subject])
    ).save_npz(path)
    return path


def test_align_cache_shifts_the_named_dataset_drops_what_it_cannot_and_links_the_rest(
    tmp_path: Path,
) -> None:
    source, target = tmp_path / "src" / "p", tmp_path / "out" / "p"
    X, labels = _night(shift=3)
    _write(source, "Late", "n1", X, labels)
    _write(source, "Late", "flat", np.zeros_like(X), labels)
    other = _write(source, "Other", "s1", *_night(seed=2))

    rows = align_cache(source, target, ["Late"], AlignmentSettings())

    aligned = np.load(target / f"Late{SUBJECT_SEPARATOR}n1{CACHE_SUFFIX}")
    assert int(aligned[SHIFT_KEY]) == 3
    np.testing.assert_array_equal(aligned["y"][0], shift_labels(labels, 3))
    np.testing.assert_array_equal(aligned["X"], X[None])
    assert not (target / f"Late{SUBJECT_SEPARATOR}flat{CACHE_SUFFIX}").exists()
    linked = target / other.name
    assert linked.is_symlink() and linked.resolve() == other.resolve()
    manifest = json.loads((target / "alignment.json").read_text())
    assert manifest["nights"]["Late/n1"] == {"shift": 3}
    assert "dropped" in manifest["nights"]["Late/flat"]
    assert {row["subject_id"] for row in rows} == {"n1", "flat"}
    assert all(f"{style}_shift" in rows[0] for style in STYLES)


def test_a_dry_run_reports_without_writing(tmp_path: Path) -> None:
    source, target = tmp_path / "src" / "p", tmp_path / "out" / "p"
    _write(source, "Late", "n1", *_night(shift=3))

    rows = align_cache(source, target, ["Late"], AlignmentSettings(), dry_run=True)

    assert rows[0]["kappa_shift"] == 3
    assert not target.exists()

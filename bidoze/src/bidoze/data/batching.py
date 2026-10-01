"""Turn per-subject feature bundles into padded, stage-remapped batches."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from pisces_lite.cv import SubjectRef
from pisces_lite.model_io import ModelIOBundle
from pisces_lite.utils import pad_or_truncate

from bidoze.model.constants import GAP_LABEL, SPECTROGRAM_PADDING_VALUE


#: Index, along the frequency axis of a pooled surface, of the row holding each
#: epoch's missing-frame fraction: 0 when every frame had data, 1 when none did.
GAP_ROW = -1


def pool_frames_to_epochs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean-pool frames to one row per ``y`` row, and append the gap row.

    Sub-epoch caches -- the 12 Hz log-PSD config steps every 2 s -- carry
    several frames per 30 s PSG epoch, so ``X`` (``(B, frames, F, C)``) runs
    that many times longer than ``y``. pisces-lite marks a frame with no
    accelerometer data by writing SPECTROGRAM_PADDING_VALUE into every bin and
    channel. That value sits inside the log-PSD range, so it must never be
    averaged into real frames: missing frames are left out of their epoch's
    mean, and the fraction of the epoch they made up is appended as one extra
    row along the frequency axis (:data:`GAP_ROW`), repeated across channels.

    An epoch with no real frames at all takes the recording's median spectrum,
    so the spectral rows stay where real data lives and the gap row alone says
    the data is absent. Returns ``(B, epochs, F + 1, C)``; any trailing
    feature shape works, the gap row always extends the first feature axis.
    """
    B, n_frames = X.shape[:2]
    feature_shape = X.shape[2:]
    frames_per_epoch = max(n_frames // max(y.shape[1], 1), 1)
    n_epochs = n_frames // frames_per_epoch
    frames = X[:, : n_epochs * frames_per_epoch].reshape(
        B, n_epochs, frames_per_epoch, *feature_shape
    )
    feature_axes = tuple(range(3, frames.ndim))
    real = ~np.all(frames == SPECTROGRAM_PADDING_VALUE, axis=feature_axes)
    count = real.sum(axis=2)
    expand = (...,) + (None,) * len(feature_shape)
    sums = np.where(real[expand], frames, 0.0).sum(axis=2)
    pooled = sums / np.maximum(count, 1)[expand]
    for row in range(B):
        empty = count[row] == 0
        if empty.any():
            pooled[row, empty] = _fill_spectrum(pooled[row, ~empty], feature_shape)
    gap = (1.0 - count / frames_per_epoch).astype(pooled.dtype)
    gap_row = np.broadcast_to(gap[expand], (B, n_epochs, 1, *feature_shape[1:]))
    return np.concatenate([pooled, gap_row], axis=2)


def _fill_spectrum(real_epochs: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """The median spectrum of a recording's real epochs (zeros if it has none)."""
    if real_epochs.shape[0] == 0:
        return np.zeros(shape)
    return np.median(real_epochs, axis=0)


def epoch_padding(pooled: np.ndarray, n: int) -> np.ndarray:
    """``n`` absent epochs for one pooled recording ``(epochs, F + 1, C)``.

    Each is the recording's median spectrum with the gap row at 1, the same
    thing :func:`pool_frames_to_epochs` writes for an epoch with no data, so
    padding and genuine dropouts look alike to the model.
    """
    _, missing = split_gap_row(pooled)
    fill = np.empty((n, *pooled.shape[1:]), dtype=pooled.dtype)
    spectrum_shape = (pooled.shape[1] - 1, *pooled.shape[2:])
    fill[:, :GAP_ROW] = _fill_spectrum(pooled[missing < 1.0, :GAP_ROW], spectrum_shape)
    fill[:, GAP_ROW] = 1.0
    return fill


def pad_pooled(pooled: np.ndarray, target_length: int) -> np.ndarray:
    """Pad one pooled recording with absent epochs, or truncate it, to length."""
    if pooled.shape[0] >= target_length:
        return pooled[:target_length]
    return np.concatenate([pooled, epoch_padding(pooled, target_length - pooled.shape[0])])


def split_gap_row(pooled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(spectrum, missing_fraction)`` from a pooled ``(T, F + 1[, C])`` surface."""
    gap = pooled[:, GAP_ROW]
    return pooled[:, :GAP_ROW], gap.reshape(gap.shape[0], -1)[:, 0]


def stack_bundles(
    bundles: Sequence[ModelIOBundle],
    target_x_length: int,
    target_y_length: int,
    y_padding: int = GAP_LABEL,
) -> ModelIOBundle:
    """Pool and pad each subject to the target length, then concatenate."""
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for bundle in bundles:
        pooled = pool_frames_to_epochs(bundle.X, bundle.y)
        Xs.append(np.stack([pad_pooled(row, target_x_length) for row in pooled]))
        ys.append(
            pad_or_truncate(bundle.y, target_length=target_y_length, pad_value=y_padding, axis=1)
        )
    return ModelIOBundle(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        datasets=np.concatenate([b.datasets for b in bundles], axis=0),
        subject_ids=np.concatenate([b.subject_ids for b in bundles], axis=0),
    )


def assemble_batch(
    features_by_ref: Dict[SubjectRef, ModelIOBundle],
    refs: Sequence[SubjectRef],
    target_x_length: int,
    target_y_length: int,
    num_classes: int,
    *,
    with_dataset_labels: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Pad, stack, and stage-remap one set of subjects into ``(X, y)``.

    Cached PSG labels use the 0-5 walch-like convention; the model expects
    ``0..num_classes-1`` plus the gap label. The remap goes through
    ``ModelIOBundle.y_for_n_classes`` so the mapping table stays in one place.
    """
    if not refs:
        raise ValueError("assemble_batch: no subjects")
    bundle = stack_bundles(
        [features_by_ref[ref] for ref in refs],
        target_x_length=target_x_length,
        target_y_length=target_y_length,
    )
    y = bundle.y_for_n_classes(num_classes=num_classes)
    if with_dataset_labels:
        return bundle.X, y, np.asarray(bundle.datasets).astype(str)
    return bundle.X, y

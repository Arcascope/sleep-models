"""The class-balanced training pieces of Olsen et al., IEEE TBME 2022, Section II-E.

Shared by every recipe that trains this way: the effective-number
class-balanced cross-entropy of its eq. 1, pseudo-class-balanced online
segment sampling, the held-out share of the training subjects that the fit
early-stops on, and the confusion its early-stopping kappa is read from.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from pisces_lite.cv import SubjectRef

from arcasleep.data.constants import GAP_LABEL, SPECTROGRAM_PADDING_VALUE
from arcasleep.evaluation.staging_metrics import scored_epoch_mask

#: The learning rate is divided by this after ``lr_drop_patience`` epochs
#: without an improvement in eval kappa.
LR_DROP_FACTOR = math.sqrt(10.0)
#: Recording draws per segment before the sampler gives up on its target class.
CLASS_SAMPLE_MAX_ATTEMPTS = 100


def split_train_eval(
    refs: Sequence[SubjectRef], eval_fraction: float, split_seed: int
) -> tuple[list[SubjectRef], list[SubjectRef]]:
    """Hold out ``eval_fraction`` of the subjects to early-stop on, seeded by ``split_seed``.

    The split is by subject id, so a subject listed under several datasets
    lands on one side.
    """
    groups: dict[str, list[SubjectRef]] = {}
    for ref in refs:
        groups.setdefault(ref.subject_id, []).append(ref)
    group_refs = list(groups.values())

    order = np.random.default_rng(split_seed).permutation(len(group_refs))
    eval_count = max(1, int(round(eval_fraction * len(group_refs))))
    eval_refs = sorted(ref for i in order[:eval_count] for ref in group_refs[i])
    fit_refs = sorted(ref for i in order[eval_count:] for ref in group_refs[i])
    if not fit_refs:
        raise ValueError("the train/eval split left no subjects to fit on")
    return fit_refs, eval_refs


def balanced_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    beta: float,
) -> jnp.ndarray:
    """Effective-number class-balanced cross-entropy (Olsen et al., eq. 1).

    Class counts are the batch's validly labelled epochs; labels outside
    ``0..num_classes-1`` are masked. The sum is normalised by every epoch in
    the batch, masked ones included.
    """
    valid = (labels >= 0) & (labels < num_classes)
    safe_labels = jnp.where(valid, labels, 0)
    counts = jnp.zeros(num_classes).at[safe_labels].add(valid.astype(jnp.float32))
    class_weights = jnp.where(counts > 0, (1.0 - beta) / (1.0 - jnp.power(beta, counts)), 0.0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
    weighted = class_weights[safe_labels] * ce * valid.astype(jnp.float32)
    return jnp.sum(weighted) / float(labels.size)


def sample_balanced_segments(
    nights: Sequence[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    n_segments: int,
    segment_epochs: int,
    num_classes: int,
    frames_per_epoch: int,
    max_attempts: int = CLASS_SAMPLE_MAX_ATTEMPTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-class-balanced online sampling (Olsen et al., Section II-E).

    Each segment draws a target stage uniformly, then draws (recording, start)
    pairs until the segment holds at least one epoch of it; ``max_attempts``
    bounds the search for a stage a short night may lack. Nights are
    ``(frames, labels)`` with ``frames_per_epoch`` frames per label. Segments
    shorter than ``segment_epochs`` are padded with absent frames and the gap
    label.
    """
    fpe = frames_per_epoch
    f_bins, channels = nights[0][0].shape[1:]
    xs = np.full(
        (n_segments, segment_epochs * fpe, f_bins, channels),
        SPECTROGRAM_PADDING_VALUE,
        dtype=np.float32,
    )
    ys = np.full((n_segments, segment_epochs), GAP_LABEL, dtype=np.int32)
    for i in range(n_segments):
        target_class = int(rng.integers(num_classes))
        for _ in range(max_attempts):
            night_x, night_y = nights[int(rng.integers(len(nights)))]
            start = int(rng.integers(max(night_y.shape[0] - segment_epochs, 0) + 1))
            seg_x = night_x[start * fpe : (start + segment_epochs) * fpe]
            seg_y = night_y[start : start + segment_epochs]
            if np.any(seg_y == target_class):
                break
        xs[i, : seg_x.shape[0]] = seg_x
        ys[i, : seg_y.shape[0]] = seg_y
    return xs, ys


def selection_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``(num_classes, num_classes + 1)`` confusion over scored epochs for model selection.

    Rows are the true stage; the extra column holds scored epochs the model
    called gap (any prediction outside ``0..num_classes-1``), so
    :func:`~arcasleep.evaluation.staging_metrics.cohen_kappa_from_confusion`
    counts each of them as a miss.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    scored = scored_epoch_mask(y_true) & (y_true < num_classes)
    pred = np.where((y_pred >= 0) & (y_pred < num_classes), y_pred, num_classes)
    cm = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
    np.add.at(cm, (y_true[scored], pred[scored]), 1)
    return cm

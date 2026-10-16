"""The class-balanced training pieces of Olsen et al., IEEE TBME 2022, Section II-E.

Shared by every recipe that trains this way: the effective-number
class-balanced cross-entropy of its eq. 1, pseudo-class-balanced online
segment sampling, the held-out share of the training subjects that the fit
early-stops on, and the confusion its early-stopping kappa is read from.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from pisces_lite.cv import SubjectRef

from arcasleep.data.cache import participant_id
from arcasleep.data.constants import GAP_LABEL, SPECTROGRAM_PADDING_VALUE
from arcasleep.evaluation.staging_metrics import scored_epoch_mask

#: The learning rate is divided by this when :class:`Patience` runs out.
LR_DROP_FACTOR = math.sqrt(10.0)
#: Units a :class:`Patience` can be counted in.
PATIENCE_UNITS = ("epochs", "steps")
#: Recording draws per segment before the sampler gives up on its target class.
CLASS_SAMPLE_MAX_ATTEMPTS = 100
#: Where a drawn row's recording sits in the fit length: "start" keeps it at
#: epoch 0; "random" moves it to a uniformly drawn epoch that still fits it whole;
#: "circular" rotates the row by a uniformly drawn number of epochs, so a
#: recording can wrap from the row's end to its start; "wake-cut" places a
#: recording that fills at most half the row as "random" does, and rotates a longer
#: one only by offsets whose cut has ``WAKE_CUT_MARGIN`` epochs of wake, unscored or
#: padding on each side.
SEGMENT_PLACEMENTS = ("start", "random", "circular", "wake-cut")
#: Epochs on each side of a "wake-cut" rotation's cut that must hold no sleep.
WAKE_CUT_MARGIN = 60


@dataclass(frozen=True)
class Patience:
    """When the learning rate drops and the fit stops, read from the eval kappa.

    The learning rate drops after ``lr_drop`` ``unit``s without an improvement
    in eval kappa, and again after each further ``lr_drop``; the fit stops after
    ``early_stop``. In ``"steps"`` the rule allows the same number of optimizer
    steps on a small cohort as on a large one; in ``"epochs"`` a cohort with
    fewer training nights gets fewer steps.

    With ``smoothing_steps`` > 0, the kappa compared is the mean of the
    evaluations made in the last ``smoothing_steps`` optimizer steps, so a
    single high evaluation does not reset the patience. No evaluation counts
    as an improvement before the fit has run ``smoothing_steps`` steps. The fit keeps the
    parameters of the epoch with the best unsmoothed kappa either way.
    """

    lr_drop: int
    early_stop: int
    unit: str = "epochs"
    smoothing_steps: int = 0

    def __post_init__(self) -> None:
        if self.unit not in PATIENCE_UNITS:
            raise ValueError(f"patience unit must be one of {PATIENCE_UNITS}, got {self.unit!r}")
        if self.lr_drop < 1 or self.early_stop < 1:
            raise ValueError("patiences must be at least 1")
        if self.smoothing_steps < 0:
            raise ValueError("smoothing_steps must be nonnegative")


@dataclass(frozen=True)
class PlateauDecision:
    """What :meth:`Plateau.update` decided after one epoch's evaluation."""

    smoothed_kappa: float
    improved: bool
    drop_lr: bool
    stop: bool


class Plateau:
    """Tracks the eval kappa over a fit and applies a :class:`Patience`."""

    def __init__(self, patience: Patience, steps_per_epoch: int) -> None:
        scale = steps_per_epoch if patience.unit == "epochs" else 1
        self.lr_drop_steps = patience.lr_drop * scale
        self.early_stop_steps = patience.early_stop * scale
        self.patience = patience
        self.steps_per_epoch = steps_per_epoch
        self.step = 0
        self.best_smoothed = -np.inf
        self.best_step = 0
        self.last_reset_step = 0
        self._window: deque[tuple[int, float]] = deque()

    def update(self, kappa: float) -> PlateauDecision:
        """Record one epoch's eval ``kappa``, after that epoch's optimizer steps."""
        self.step += self.steps_per_epoch
        if np.isfinite(kappa):
            self._window.append((self.step, float(kappa)))
        span = max(self.patience.smoothing_steps, 1)
        while self._window and self._window[0][0] <= self.step - span:
            self._window.popleft()
        smoothed = float(np.mean([k for _, k in self._window])) if self._window else float("nan")

        # Until the window spans smoothing_steps it holds fewer evaluations
        # than it should, and its mean is no smoother than one evaluation.
        window_full = self.step >= self.patience.smoothing_steps
        improved = window_full and np.isfinite(smoothed) and smoothed > self.best_smoothed
        if improved:
            self.best_smoothed = smoothed
            self.best_step = self.step
            self.last_reset_step = self.step
        stop = self.step - self.best_step >= self.early_stop_steps
        drop_lr = not stop and self.step - self.last_reset_step >= self.lr_drop_steps
        if drop_lr:
            self.last_reset_step = self.step
        return PlateauDecision(smoothed, bool(improved), drop_lr, stop)

    def describe_stop(self) -> str:
        """The early-stop message, in the patience's own unit."""
        return f"no eval improvement in {self.patience.early_stop} {self.patience.unit}"


def split_train_eval(
    refs: Sequence[SubjectRef], eval_fraction: float, split_seed: int
) -> tuple[list[SubjectRef], list[SubjectRef]]:
    """Hold out ``eval_fraction`` of the participants to early-stop on, seeded by ``split_seed``.

    The split is by participant, so every recording of a participant (other
    nights, other devices, the same subject listed under several datasets)
    lands on one side.
    """
    groups: dict[str, list[SubjectRef]] = {}
    for ref in refs:
        groups.setdefault(participant_id(ref.subject_id), []).append(ref)
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


def recorded_lengths(ys: np.ndarray) -> np.ndarray:
    """Per row of ``ys`` ``(B, T)``, the epochs up to and including the last one not labelled gap."""
    labelled = np.asarray(ys) != GAP_LABEL
    last = ys.shape[1] - np.argmax(labelled[:, ::-1], axis=1)
    return np.where(labelled.any(axis=1), last, 0)


def placement_offsets(ys: np.ndarray, rng: np.random.Generator, placement: str) -> np.ndarray:
    """Per padded row of ``ys`` ``(B, T)``, the epochs ``placement`` rolls it by.

    See ``SEGMENT_PLACEMENTS``. A roll by ``k`` puts the row's epoch ``(T - k) % T``
    at epoch 0, so the context's edges cut the row there. A "wake-cut" row with no
    such cut, or in a row no longer than ``2 * WAKE_CUT_MARGIN``, is placed as
    "random" does.
    """
    if placement not in SEGMENT_PLACEMENTS:
        raise ValueError(f"placement must be one of {SEGMENT_PLACEMENTS}, got {placement!r}")
    width = ys.shape[1]
    if placement == "start":
        return np.zeros(len(ys), dtype=np.int64)
    if placement == "circular":
        return rng.integers(width, size=len(ys))
    lengths = recorded_lengths(ys)
    offsets = rng.integers(width - lengths + 1)
    if placement == "wake-cut" and width > 2 * WAKE_CUT_MARGIN:
        m = WAKE_CUT_MARGIN
        for i in np.flatnonzero(2 * lengths > width):
            # Wake is 0, unscored -1 and gap -2.
            quiet = (ys[i] <= 0).astype(np.int64)
            wrapped = np.concatenate([quiet[width - m :], quiet, quiet[:m]])
            # quiet_around[c]: the epochs c - m .. c + m - 1 (mod T) hold no sleep.
            quiet_around = np.convolve(wrapped, np.ones(2 * m, dtype=np.int64), "valid")[:width] == 2 * m
            cuts = np.flatnonzero(quiet_around)
            if cuts.size:
                offsets[i] = (width - int(rng.choice(cuts))) % width
    return offsets


def roll_rows(
    xs: np.ndarray,
    ys: np.ndarray,
    offsets: np.ndarray,
    frames_per_epoch: int,
    extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Roll each row of ``xs``, ``ys`` and ``extra`` forward by its ``offsets`` epochs."""
    fpe = frames_per_epoch
    xs, ys = xs.copy(), ys.copy()
    extra = None if extra is None else extra.copy()
    for i, offset in enumerate(offsets):
        if offset:
            xs[i] = np.roll(xs[i], int(offset) * fpe, axis=0)
            ys[i] = np.roll(ys[i], int(offset), axis=0)
            if extra is not None:
                extra[i] = np.roll(extra[i], int(offset), axis=0)
    return xs, ys, extra


def place_rows(
    xs: np.ndarray,
    ys: np.ndarray,
    rng: np.random.Generator,
    frames_per_epoch: int,
    extra: np.ndarray | None = None,
    placement: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Move each padded row's recording to a random epoch of the row.

    Each row ``(frames, labels)`` holds a recording followed by gap padding.
    With "random", the recording moves to an offset drawn uniformly from the
    offsets that keep it whole, and the padding that followed it fills the epochs
    before it. Padding epochs are identical within a row (the recording's median
    spectrum, gap row at 1), so this is a roll. With "circular" the offset is drawn
    from every epoch of the row: a recording longer than the padding is split, its
    end first, then the padding, then its start. "wake-cut" splits only a recording
    longer than half the row, and only where it holds no sleep
    (:func:`placement_offsets`). ``extra`` ``(B, T, ...)``, per-epoch values such as
    soft targets, moves with the labels.
    """
    offsets = placement_offsets(ys, rng, placement)
    return roll_rows(xs, ys, offsets, frames_per_epoch, extra)


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

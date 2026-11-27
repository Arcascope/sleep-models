"""Weak labels for the epochs outside the PSG span of an uncropped cache.

An uncropped recording keeps its whole wrist stream; the epochs outside the PSG
are :data:`~arcasleep.data.constants.OUTSIDE_PSG_LABEL`. A *rule* turns the
recording's 5 s z-angle series (``anglez_5s``, ``(epochs, 6)``, NaN where there
is no data) into a :class:`WeakLabels` with a label, a loss weight and a target
kind per epoch. Rules read only the movement series: no clock time and no
position in the recording.

Two target kinds exist, per epoch:

- a stage label, ``0..num_classes-1`` as the PSG labels are;
- a sleep/wake-only label, ``0`` wake and ``1`` sleep of any stage, for a rule
  that knows whether the wearer slept but not which stage.

:func:`label_recording` applies an ordered list of rules to the outside-PSG
epochs only; the first rule to label an epoch keeps it. Epochs inside the PSG
span carry their PSG labels at weight 1 and are never touched by a rule.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from arcasleep.data.constants import GAP_LABEL, OUTSIDE_PSG_LABEL
from arcasleep.ggir import EPOCH_SECONDS, evaluate_epoch_grid

#: Label of an epoch no rule labelled: masked from the loss.
UNLABELED = -1
#: Sleep/wake-only labels.
WAKE, SLEEP = 0, 1
#: Mean |z-angle change| per 5 s bin (degrees) over the centred window at or
#: above which an epoch is active, and clearly active.
ACTIVE_DEG = 2.0
CLEAR_DEG = 4.0
#: Minutes of the centred window the activity is averaged over.
ACTIVITY_MINUTES = 5.0
#: Loss weights of a clearly active epoch and of a merely active one.
STRONG_WEIGHT = 1.0
WEAK_WEIGHT = 0.3
#: Loss weight of a sleep/wake-only sleep label.
SLEEP_WEIGHT = 0.3
#: Shortest sustained-inactivity bout, in minutes, that the sleep rule labels.
MIN_SIB_BOUT_MINUTES = 20.0


@dataclass(frozen=True)
class WeakLabels:
    """Per-epoch weak labels from one rule, all of length ``epochs``.

    ``labels`` is :data:`UNLABELED` where the rule says nothing, and
    ``weights`` is 0 there. ``sleep_wake`` is ``True`` where ``labels`` is a
    sleep/wake-only label rather than a stage.
    """

    labels: np.ndarray
    weights: np.ndarray
    sleep_wake: np.ndarray

    def __post_init__(self) -> None:
        if not (self.labels.shape == self.weights.shape == self.sleep_wake.shape):
            raise ValueError("labels, weights and sleep_wake must have one shape")
        if (self.weights < 0).any():
            raise ValueError("weights must be nonnegative")
        if (self.weights[self.labels == UNLABELED] != 0).any():
            raise ValueError("an unlabeled epoch must have weight 0")


@dataclass(frozen=True)
class WeakLabelConfig:
    """Loss weights and thresholds of the weak label rules.

    ``strong`` and ``weak`` are the wake rule's weights; ``sleep`` is the sleep
    rule's, which labels only sustained-inactivity bouts of at least
    ``min_sib_bout_minutes`` (0 labels every bout).
    """

    strong: float = STRONG_WEIGHT
    weak: float = WEAK_WEIGHT
    sleep: float = SLEEP_WEIGHT
    min_sib_bout_minutes: float = MIN_SIB_BOUT_MINUTES

    def __post_init__(self) -> None:
        if min(self.strong, self.weak, self.sleep, self.min_sib_bout_minutes) < 0:
            raise ValueError("weak label weights must be nonnegative")


#: ``(anglez_5s, weights) -> WeakLabels`` over a whole recording.
WeakLabelRule = Callable[[np.ndarray, WeakLabelConfig], WeakLabels]


def movement(grid: np.ndarray, minutes: float = ACTIVITY_MINUTES) -> np.ndarray:
    """Mean |z-angle change| per 5 s bin over a window of ``minutes`` centred on each epoch.

    ``grid`` is ``(epochs, 6)``. The window runs over the bins around the
    epoch's six, as the mean of those bins' own centred windows; NaN where a
    window holds data in under a third of its bins.
    """
    flat = grid.reshape(-1)
    change = np.abs(np.diff(flat, prepend=np.nan))
    ok = np.isfinite(change)
    width = max(1, int(round(minutes * 60 / 5)))
    kernel = np.ones(width)
    total = np.convolve(np.where(ok, change, 0.0), kernel, mode="same")
    count = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count < width / 3] = np.nan
    bins = mean.reshape(grid.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(bins, axis=1)


def daytime_wake_rule(grid: np.ndarray, weights: WeakLabelConfig) -> WeakLabels:
    """Wake where the wearer is outside GGIR's sleep window and not inactive, or is moving.

    An epoch with data is wake when it is outside the GGIR sleep period window
    and not in a sustained-inactivity bout, or when its 5-minute centred
    movement (:func:`movement`) is at least :data:`ACTIVE_DEG`. The weight is
    ``weights.strong`` at a movement of at least :data:`CLEAR_DEG` and
    ``weights.weak`` otherwise. Every other epoch is unlabeled.
    """
    grid = np.asarray(grid, dtype=np.float64)
    has_data = np.isfinite(grid).any(axis=1)
    ggir = evaluate_epoch_grid(grid)
    activity = movement(grid, ACTIVITY_MINUTES)
    with np.errstate(invalid="ignore"):
        active = activity >= ACTIVE_DEG
        clear = activity >= CLEAR_DEG
    wake = has_data & ((~ggir.is_spt & ~ggir.is_sib) | active)
    labels = np.where(wake, WAKE, UNLABELED).astype(np.int32)
    weight = np.where(wake, np.where(clear, weights.strong, weights.weak), 0.0)
    return WeakLabels(labels, weight.astype(np.float32), np.zeros(labels.shape, dtype=bool))


def _long_runs(mask: np.ndarray, minutes: float) -> np.ndarray:
    """``mask`` with the runs of ``True`` shorter than ``minutes`` removed."""
    need = int(np.ceil(minutes * 60 / EPOCH_SECONDS))
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    out = np.zeros(mask.shape, dtype=bool)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= need:
            out[start:stop] = True
    return out


def daytime_sleep_rule(grid: np.ndarray, weights: WeakLabelConfig) -> WeakLabels:
    """Sleep where the wearer is inside GGIR's sleep window and a long sustained-inactivity bout.

    An epoch is sleep when it is in the GGIR sleep period window and in a
    sustained-inactivity bout of at least ``weights.min_sib_bout_minutes``, and
    :func:`daytime_wake_rule` does not label it
    wake, so the two rules never label one epoch. The label is sleep/wake-only
    (:data:`SLEEP`, any stage) at ``weights.sleep``. Every other epoch is
    unlabeled.
    """
    grid = np.asarray(grid, dtype=np.float64)
    has_data = np.isfinite(grid).any(axis=1)
    ggir = evaluate_epoch_grid(grid)
    wake = daytime_wake_rule(grid, weights).labels == WAKE
    sleep = has_data & ggir.is_spt & _long_runs(ggir.is_sib, weights.min_sib_bout_minutes) & ~wake
    labels = np.where(sleep, SLEEP, UNLABELED).astype(np.int32)
    weight = np.where(sleep, weights.sleep, 0.0).astype(np.float32)
    return WeakLabels(labels, weight, sleep)


#: Rules by the name ``training.json``'s ``weak_label_rules`` spells.
WEAK_LABEL_RULES: dict[str, WeakLabelRule] = {
    "daytime_wake": daytime_wake_rule,
    "daytime_sleep": daytime_sleep_rule,
}


@dataclass(frozen=True)
class LabelledRecording:
    """One recording's training targets, one entry per epoch.

    ``labels`` holds PSG stages and :data:`~arcasleep.data.constants.GAP_LABEL`
    inside the PSG span, weak labels where a rule labelled an outside epoch,
    and :data:`UNLABELED` elsewhere. ``weights`` is 1 for every epoch inside
    the PSG span, the rule's weight for a weakly labelled epoch and 0 for an
    unlabeled one. ``sleep_wake`` marks sleep/wake-only labels. ``in_psg``
    marks the epochs from the PSG span's first to its last.
    """

    labels: np.ndarray
    weights: np.ndarray
    sleep_wake: np.ndarray
    in_psg: np.ndarray


def label_recording(
    y: np.ndarray,
    anglez_5s: np.ndarray | None,
    rules: Sequence[str] = (),
    weights: WeakLabelConfig = WeakLabelConfig(),
) -> LabelledRecording:
    """Targets for one uncropped recording from its ``y`` and, for the rules, its ``anglez_5s``.

    ``y`` is the recording's labels with the outside-PSG epochs as
    :data:`~arcasleep.data.constants.OUTSIDE_PSG_LABEL`. The named ``rules``
    apply in order to those epochs only. Without rules every outside epoch is
    :data:`UNLABELED`, and ``anglez_5s`` is not read.
    """
    y = np.asarray(y)
    outside = y == OUTSIDE_PSG_LABEL
    inside = np.flatnonzero(~outside)
    if inside.size == 0:
        raise ValueError("the recording has no epoch inside the PSG span")
    in_psg = np.zeros(y.shape, dtype=bool)
    in_psg[inside[0] : inside[-1] + 1] = True

    labels = np.where(outside, UNLABELED, y).astype(np.int32)
    weight = np.where(outside, 0.0, 1.0).astype(np.float32)
    sleep_wake = np.zeros(y.shape, dtype=bool)
    unknown = [name for name in rules if name not in WEAK_LABEL_RULES]
    if unknown:
        raise ValueError(f"unknown weak label rules {unknown}; known: {sorted(WEAK_LABEL_RULES)}")
    if rules and anglez_5s is None:
        raise ValueError("weak label rules need the cache's anglez_5s")
    for name in rules:
        weak = WEAK_LABEL_RULES[name](np.asarray(anglez_5s)[: y.shape[0]], weights)
        if weak.labels.shape != y.shape:
            raise ValueError(f"rule {name} labelled {weak.labels.shape[0]} of {y.shape[0]} epochs")
        take = outside & (labels == UNLABELED) & (weak.labels != UNLABELED)
        labels[take] = weak.labels[take]
        weight[take] = weak.weights[take]
        sleep_wake[take] = weak.sleep_wake[take]
    return LabelledRecording(labels, weight, sleep_wake, in_psg)

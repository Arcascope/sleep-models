"""Stage-level metrics for the Autofish evaluation protocol.

These register into the pisces-lite metric registry, so ``metrics.json`` can
name them alongside the built-ins (``auroc``, ``tst_mape``, ``waso_mape``):

    sleep_sensitivity      sleep-vs-wake recall
    wake_specificity       sleep-vs-wake specificity
    tst_error_minutes      signed TST error, predicted minus true
    tst_abs_error_minutes  the same error's magnitude
    kappa_4stage           Cohen's kappa over the four sleep stages
    kappa_sleep_wake       Cohen's kappa over binarized wake/sleep

Every metric here scores the same hard predictions, produced by
:func:`staged_preds` and governed by one knob, ``sleep_threshold``
(``ctx.extra``, so ``metrics.json`` sets it once for the whole protocol):

* ``None`` — plain argmax over all classes.
* a float — sleep is called when the *summed* sleep-class probability reaches
  the threshold. Summing matters: ``1 - P(wake)`` is not the sleep probability,
  because it also carries the gap logit.

Operating points are usually chosen as the threshold giving 95% mean
sensitivity, averaged over the recordings in a cohort.

Conventions shared with the rest of the reporting stack:

* Gap truth epochs are excluded from every metric here. They are charted, but
  they are not staged.
* A *predicted* gap is neither a sleep call nor a wake call, so it costs both
  sensitivity and specificity a miss.
* A cohort with no true sleep, or no true wake, reports NaN rather than a
  meaningless 0.0 or 1.0 — the same "too little truth to score" rule that
  ``min_stage_minutes`` applies to ``tst_mape``.

Importing this module is what registers the metrics; ``autofish.evaluation``
does that on import so a caller cannot forget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from pisces_lite.metrics.context import EvalContext
from pisces_lite.metrics.specs import hard_preds, psg_dt, register_metric
from pisces_lite.metrics.specs.base import gap_class_indices, sleep_mask


#: Sleep stages the protocol scores: wake, light, deep, REM.
DEFAULT_NUM_STAGES = 4
#: Where the gap class sits in the net's ``num_classes + 1`` output.
GAP_STAGE_INDEX = 4
#: Epochs at this label are dropped from scoring by the metrics logger.
MASKED_LABEL = -1


def scored_epoch_mask(y_true: np.ndarray) -> np.ndarray:
    """Epochs carrying real sleep-stage truth, for stage/TST/WASO metrics."""
    y = np.asarray(y_true)
    return (y >= 0) & (y != GAP_STAGE_INDEX)


def labels_for_metric_scoring(labels: np.ndarray) -> np.ndarray:
    """Drop gap epochs from stage, TST, and WASO scoring.

    Gap epochs are predicted and charted, but they carry no sleep stage, so
    they are masked rather than scored against.
    """
    y = np.asarray(labels).copy()
    y[y == GAP_STAGE_INDEX] = MASKED_LABEL
    return y


def _wake_class(ctx: EvalContext) -> int:
    return int(ctx.extra.get("wake_class", 0))


def sleep_threshold(ctx: EvalContext) -> float | None:
    """The configured operating point, or ``None`` for argmax."""
    value = ctx.extra.get("sleep_threshold")
    return None if value is None else float(value)


def split_class_indices(
    n_classes: int, wake_class: int, gap_classes: Sequence[int]
) -> tuple[List[int], List[int]]:
    """Partition the output classes into ``(sleep, non-sleep)`` index lists.

    Non-sleep is wake plus every gap class, in that order, so a tie between
    wake and gap resolves to wake.
    """
    gaps = [g for g in gap_classes if 0 <= g < n_classes]
    sleep = [c for c in range(n_classes) if c != wake_class and c not in set(gaps)]
    return sleep, [wake_class, *gaps]


def threshold_predictions(
    proba: np.ndarray,
    threshold: float,
    wake_class: int,
    gap_classes: Sequence[int],
) -> np.ndarray:
    """Hard predictions at a sleep operating point.

    ``sleep_prob`` is the sum over the sleep classes. Where it reaches
    ``threshold`` the call is the most likely *sleep* stage; otherwise it is
    whichever of wake or gap is more likely.
    """
    proba = np.asarray(proba)
    sleep, non_sleep = split_class_indices(proba.shape[-1], wake_class, gap_classes)
    if not sleep:
        raise ValueError("threshold_predictions: no sleep classes in the output")

    sleep_prob = proba[..., sleep].sum(axis=-1)
    sleep_call = np.take(sleep, np.argmax(proba[..., sleep], axis=-1))
    non_sleep_call = np.take(non_sleep, np.argmax(proba[..., non_sleep], axis=-1))
    return np.where(sleep_prob >= threshold, sleep_call, non_sleep_call)


def staged_preds(ctx: EvalContext) -> np.ndarray:
    """The hard predictions every metric in this module scores."""
    threshold = sleep_threshold(ctx)
    if threshold is None:
        return np.asarray(hard_preds(ctx))
    return threshold_predictions(
        ctx.y_pred_proba,
        threshold,
        _wake_class(ctx),
        gap_class_indices(ctx),
    )


#: One scored recording: ``(y_true, y_pred_proba)`` with shapes ``(T,)`` and ``(T, C)``.
Recording = Tuple[np.ndarray, np.ndarray]


def recording_sleep_sensitivity(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    wake_class: int = 0,
    gap_classes: Sequence[int] = (GAP_STAGE_INDEX,),
) -> float:
    """Sleep sensitivity for one recording at one operating point.

    NaN when the recording has no true sleep epoch to score against.
    """
    y_true = np.asarray(y_true).reshape(-1)
    proba = np.asarray(proba)
    sleep, _ = split_class_indices(proba.shape[-1], wake_class, gap_classes)
    sleep_set = set(sleep)

    scored = y_true >= 0
    for gap in gap_classes:
        scored = scored & (y_true != gap)
    true_sleep = scored & np.isin(y_true, list(sleep_set))
    if not true_sleep.any():
        return float("nan")

    predicted = threshold_predictions(proba, threshold, wake_class, gap_classes)
    predicted = np.asarray(predicted).reshape(-1)
    return float(np.mean(np.isin(predicted[true_sleep], list(sleep_set))))


def mean_sleep_sensitivity(
    recordings: Sequence[Recording],
    threshold: float,
    wake_class: int = 0,
    gap_classes: Sequence[int] = (GAP_STAGE_INDEX,),
) -> float:
    """Sensitivity averaged over recordings, not over epochs.

    Averaging per recording is what keeps one long night from dominating the
    operating point. Recordings with no true sleep drop out rather than
    counting as zero.
    """
    scores = [
        recording_sleep_sensitivity(y_true, proba, threshold, wake_class, gap_classes)
        for y_true, proba in recordings
    ]
    usable = [s for s in scores if not np.isnan(s)]
    if not usable:
        return float("nan")
    return float(np.mean(usable))


@dataclass(frozen=True)
class ThresholdSearch:
    """The operating point a search settled on, and how it got there."""

    threshold: float
    sensitivity: float
    iterations: int
    attained: bool

    def __str__(self) -> str:
        status = "attained" if self.attained else "NOT attained"
        return (
            f"sleep_threshold={self.threshold:.6f} "
            f"mean_sensitivity={self.sensitivity:.4f} ({status}, "
            f"{self.iterations} iterations)"
        )


def search_sleep_threshold(
    recordings: Sequence[Recording],
    target_sensitivity: float = 0.95,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    tolerance: float = 1e-4,
    max_iterations: int = 60,
    wake_class: int = 0,
    gap_classes: Sequence[int] = (GAP_STAGE_INDEX,),
) -> ThresholdSearch:
    """Binary search for the strictest threshold still meeting a sensitivity target.

    Mean sensitivity falls monotonically as the threshold rises — a higher bar
    calls less sleep — so bisection is well posed and does not depend on the
    shape of the probability distribution the way a quantile estimate would.

    Returns the *largest* threshold whose mean sensitivity is at least
    ``target_sensitivity``, which is the conservative choice: it calls as
    little sleep as the target allows. ``attained`` is False when even ``lo``
    misses the target, in which case ``lo`` is returned with the sensitivity it
    actually achieved.
    """
    if not recordings:
        raise ValueError("search_sleep_threshold: no recordings")
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    if not lo < hi:
        raise ValueError(f"search bounds must satisfy lo < hi; got lo={lo}, hi={hi}")

    def sensitivity_at(threshold: float) -> float:
        return mean_sleep_sensitivity(recordings, threshold, wake_class, gap_classes)

    hi_sensitivity = sensitivity_at(hi)
    if hi_sensitivity >= target_sensitivity:
        return ThresholdSearch(hi, hi_sensitivity, 0, True)

    lo_sensitivity = sensitivity_at(lo)
    if np.isnan(lo_sensitivity):
        raise ValueError("search_sleep_threshold: no recording has any true sleep epoch")
    if lo_sensitivity < target_sensitivity:
        return ThresholdSearch(lo, lo_sensitivity, 0, False)

    # Invariant from here: sensitivity_at(lo) >= target > sensitivity_at(hi).
    iterations = 0
    while hi - lo > tolerance and iterations < max_iterations:
        mid = 0.5 * (lo + hi)
        mid_sensitivity = sensitivity_at(mid)
        if mid_sensitivity >= target_sensitivity:
            lo, lo_sensitivity = mid, mid_sensitivity
        else:
            hi = mid
        iterations += 1

    return ThresholdSearch(lo, lo_sensitivity, iterations, True)


def pooled_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Confusion matrix (rows true, cols predicted) over scored epochs.

    The net also classifies the gap class at index ``num_classes``; epochs
    predicted as gap are dropped, since they are unscored under the protocol.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    valid = (
        scored_epoch_mask(y_true)
        & (y_true < num_classes)
        & (y_pred >= 0)
        & (y_pred < num_classes)
    )
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


def cohen_kappa_from_confusion(cm: np.ndarray) -> float:
    """Cohen's kappa from a confusion matrix (rows true, cols predicted).

    The matrix may be rectangular when one side has a category the other
    cannot produce -- ``kappa_sleep_wake`` gives predicted gaps their own
    column -- in which case only the aligned leading categories contribute
    to the chance agreement, and the extra column's epochs count as
    disagreements.
    """
    total = cm.sum()
    if total == 0:
        return float("nan")
    po = np.trace(cm) / total
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    aligned = min(len(row), len(col))
    pe = float((row[:aligned] * col[:aligned]).sum()) / (total * total)
    if pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def _hard_arrays(ctx: EvalContext) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(ctx.y_true).reshape(-1),
        np.asarray(staged_preds(ctx)).reshape(-1),
    )


def _scored_truth(y_true: np.ndarray, ctx: EvalContext) -> np.ndarray:
    scored = y_true >= 0
    for class_idx in gap_class_indices(ctx):
        scored = scored & (y_true != class_idx)
    return scored


def _num_stages(ctx: EvalContext) -> int:
    names = list(ctx.extra.get("class_names") or [])
    if not names:
        return DEFAULT_NUM_STAGES
    return len(names) - len(gap_class_indices(ctx))


@register_metric("sleep_sensitivity")
def _sleep_sensitivity(ctx: EvalContext) -> float:
    """Share of true sleep epochs called sleep."""
    y_true, y_pred = _hard_arrays(ctx)
    true_sleep = sleep_mask(y_true, ctx)
    if not true_sleep.any():
        return float("nan")
    return float(np.mean(sleep_mask(y_pred, ctx)[true_sleep]))


@register_metric("wake_specificity")
def _wake_specificity(ctx: EvalContext) -> float:
    """Share of true wake epochs called wake."""
    y_true, y_pred = _hard_arrays(ctx)
    wake_class = _wake_class(ctx)
    true_wake = y_true == wake_class
    if not true_wake.any():
        return float("nan")
    return float(np.mean(y_pred[true_wake] == wake_class))


@register_metric("tst_error_minutes")
def _tst_error_minutes(ctx: EvalContext) -> float:
    """Signed TST error in minutes: predicted minus true."""
    y_true, y_pred = _hard_arrays(ctx)
    dt = psg_dt(ctx)
    true_tst = float(np.sum(sleep_mask(y_true, ctx))) * dt
    pred_tst = float(np.sum(sleep_mask(y_pred, ctx))) * dt
    return pred_tst - true_tst


@register_metric("tst_abs_error_minutes")
def _tst_abs_error_minutes(ctx: EvalContext) -> float:
    """Unsigned TST error in minutes.

    ``tst_error_minutes`` keeps its sign, which is what says whether a model
    over- or under-calls sleep — but averaged over a cohort those signs cancel,
    and a cohort whose members miss by a quarter-hour each way reads as
    near-zero error. Report this beside it, not instead of it.
    """
    return abs(_tst_error_minutes(ctx))


@register_metric("kappa_4stage")
def _kappa_4stage(ctx: EvalContext) -> float:
    """Cohen's kappa over the four sleep stages, gap epochs dropped."""
    y_true, y_pred = _hard_arrays(ctx)
    return cohen_kappa_from_confusion(pooled_confusion(y_true, y_pred, _num_stages(ctx)))


@register_metric("kappa_sleep_wake")
def _kappa_sleep_wake(ctx: EvalContext) -> float:
    """Cohen's kappa over wake/sleep, scored truth epochs only.

    A predicted gap is not a wake call, so it gets its own confusion column
    and counts as a miss against true sleep and true wake alike -- the same
    rule the sensitivity and specificity metrics follow.
    """
    y_true, y_pred = _hard_arrays(ctx)
    scored = _scored_truth(y_true, ctx)
    if scored.sum() == 0:
        return float("nan")
    y_true_bin = sleep_mask(y_true, ctx)[scored].astype(np.int64)
    predicted_sleep = sleep_mask(y_pred, ctx)[scored]
    predicted_gap = np.isin(y_pred[scored], list(gap_class_indices(ctx)))
    if np.unique(y_true_bin).size < 2:
        return float("nan")
    # Rows and columns: wake=0, sleep=1, then the gap column. A predicted gap
    # misses both rows.
    return cohen_kappa_from_confusion(
        np.bincount(y_true_bin * 3 + predicted_sleep * 1 + predicted_gap * 2,
                    minlength=6).reshape(2, 3)
    )

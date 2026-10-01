"""The stage-level metrics' scoring rules, without JAX or a model.

These pin down the protocol conventions the docstring states: gap truth is
never scored, a predicted gap is neither a sleep call nor a wake call, and a
cohort with too little truth reports NaN. ``kappa_sleep_wake`` is tested
against a hand-computed confusion matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pisces_lite.metrics.context import EvalContext

from bidoze.evaluation import staging_metrics
from bidoze.evaluation.staging_metrics import (
    cohen_kappa_from_confusion,
    pooled_confusion,
)

CLASS_NAMES = ["wake", "light", "deep", "rem", "gap"]
WAKE, LIGHT, DEEP, REM, GAP = range(5)


def _ctx(
    y_true,
    y_pred,
    extra: dict[str, Any] | None = None,
) -> EvalContext:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    proba = np.eye(5, dtype=np.float64)[np.maximum(y_pred, 0)]
    return EvalContext(
        y_true=y_true,
        y_pred_proba=proba,
        y_pred=y_pred,
        identity={},
        row={},
        extra={"class_names": CLASS_NAMES, **(extra or {})},
    )


# --- kappa_sleep_wake --------------------------------------------------------


def test_kappa_sleep_wake_treats_a_predicted_gap_as_a_miss() -> None:
    """A predicted gap scores wrong against true sleep and true wake alike."""
    # Truth: sleep, sleep, wake. Preds: sleep, gap, gap.
    ctx = _ctx([LIGHT, DEEP, WAKE], [LIGHT, GAP, GAP])

    value = staging_metrics._kappa_sleep_wake(ctx)

    # Confusion, rows and columns wake/sleep/gap: the true wake epoch is
    # predicted gap, one true sleep epoch is a hit and one is a gap miss.
    assert value == cohen_kappa_from_confusion(np.array([[0, 0, 1], [0, 1, 1]]))


def test_kappa_sleep_wake_matches_the_binary_case_without_gaps() -> None:
    """With no predicted gaps the metric is the classic wake/sleep kappa."""
    truth = [WAKE, LIGHT, DEEP, REM, WAKE, LIGHT]
    preds = [WAKE, LIGHT, WAKE, REM, WAKE, WAKE]
    ctx = _ctx(truth, preds)

    binary = staging_metrics._kappa_sleep_wake(ctx)
    # Hand-built wake/sleep confusion for the same calls, gap column empty.
    manual = cohen_kappa_from_confusion(np.array([[2, 0], [2, 2]]))
    assert binary == pytest.approx(manual)


def test_kappa_sleep_wake_is_nan_without_two_truth_classes() -> None:
    ctx = _ctx([WAKE, WAKE, WAKE], [WAKE, LIGHT, GAP])

    assert np.isnan(staging_metrics._kappa_sleep_wake(ctx))


# --- kappa arithmetic --------------------------------------------------------


def test_cohen_kappa_is_nan_on_an_empty_or_degenerate_confusion() -> None:
    assert np.isnan(cohen_kappa_from_confusion(np.zeros((2, 2))))
    assert np.isnan(cohen_kappa_from_confusion(np.array([[4]])))


def test_cohen_kappa_perfect_agreement_is_one() -> None:
    cm = np.array([[10, 0], [0, 10]])
    assert cohen_kappa_from_confusion(cm) == pytest.approx(1.0)


# --- pooled_confusion --------------------------------------------------------


def test_pooled_confusion_drops_gap_truth_and_gap_predictions() -> None:
    y_true = [WAKE, LIGHT, GAP, WAKE]
    y_pred = [LIGHT, LIGHT, LIGHT, GAP]

    cm = pooled_confusion(np.asarray(y_true), np.asarray(y_pred), num_classes=4)

    assert cm.tolist() == [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_pooled_confusion_counts_only_scored_epochs() -> None:
    y_true = [-1, WAKE, LIGHT, -1]
    y_pred = [WAKE, WAKE, WAKE, LIGHT]

    cm = pooled_confusion(np.asarray(y_true), np.asarray(y_pred), num_classes=4)

    assert cm.sum() == 2


# --- waso_error_minutes ------------------------------------------------------

#: Sleep interrupted by two wake epochs, bracketed by wake.
_INTERRUPTED = [WAKE, LIGHT, LIGHT, WAKE, WAKE, DEEP, REM, WAKE]
_UNBROKEN = [WAKE, LIGHT, LIGHT, LIGHT, LIGHT, DEEP, REM, WAKE]


def test_waso_error_minutes_is_zero_for_a_perfect_prediction() -> None:
    ctx = _ctx(_INTERRUPTED, _INTERRUPTED, {"psg_dt_minutes": 0.5})
    assert staging_metrics._waso_error_minutes(ctx) == 0.0


def test_waso_error_minutes_is_predicted_minus_true() -> None:
    missed = _ctx(_INTERRUPTED, _UNBROKEN, {"psg_dt_minutes": 0.5})
    invented = _ctx(_UNBROKEN, _INTERRUPTED, {"psg_dt_minutes": 0.5})

    assert staging_metrics._waso_error_minutes(missed) == pytest.approx(-1.0)
    assert staging_metrics._waso_error_minutes(invented) == pytest.approx(1.0)

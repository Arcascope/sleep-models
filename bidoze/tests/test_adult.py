from __future__ import annotations

import numpy as np

from arcasleep_bidoze.adult import (
    DEEP,
    GAP,
    LIGHT,
    REM,
    WAKE,
    adult_stage_divisor,
    effective_weights,
    hybrid_calls,
    stage_counts,
)


def _proba(calls: list[int], rng: np.random.Generator) -> np.ndarray:
    p = rng.random((len(calls), 5)) * 0.1
    p[np.arange(len(calls)), calls] = 1.0
    return p / p.sum(axis=1, keepdims=True)


def test_the_wake_source_decides_sleep_and_wake_and_the_stage_source_the_stage() -> None:
    rng = np.random.default_rng(0)
    wake = _proba([WAKE, LIGHT, DEEP, GAP, REM, WAKE], rng)
    stage = _proba([REM, DEEP, LIGHT, LIGHT, LIGHT, DEEP], rng)

    calls = hybrid_calls(wake, stage)

    assert calls.tolist() == [WAKE, DEEP, LIGHT, GAP, LIGHT, WAKE]


def test_the_divisor_moves_stage_calls_without_touching_wake_calls() -> None:
    wake = np.array([[0.1, 0.9, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0, 0.0]])
    stage = np.array([[0.0, 0.5, 0.4, 0.1, 0.0], [0.0, 0.5, 0.4, 0.1, 0.0]])

    assert hybrid_calls(wake, stage).tolist() == [LIGHT, WAKE]
    assert hybrid_calls(wake, stage, np.array([1.0, 2.0, 1.0, 1.0])).tolist() == [DEEP, WAKE]


def test_the_divisor_is_one_when_training_and_adult_stage_mixes_match_without_balancing() -> None:
    nights = np.array([[100, 400, 150, 200], [80, 420, 120, 230]])

    divisor = adult_stage_divisor(nights, nights, beta=0.0)

    np.testing.assert_allclose(divisor, 1.0)


def test_a_rarer_stage_gets_a_larger_effective_weight() -> None:
    nights = np.array([[100, 600, 50, 200]] * 4)

    w = effective_weights(nights, beta=0.999)

    assert w[DEEP] > w[WAKE] > w[REM] > w[LIGHT]


def test_stage_counts_merge_n1_and_n2_into_light() -> None:
    assert stage_counts(np.array([0, 1, 2, 2, 3, 5, 5, -1])).tolist() == [1, 3, 1, 2]

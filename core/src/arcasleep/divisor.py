"""Stage decoding with a stage divisor.

A bundle whose ``config.json`` holds a ``stage_divisor`` is scored in two steps:

- **Sleep/wake**: the model's own most probable of wake, light, deep and REM.
- **Light/deep/REM**: on the epochs it calls sleep, its light, deep and REM
  probabilities are divided by the divisor before the argmax.

Stage indices are the model's: 0 wake, 1 light, 2 deep, 3 REM, 4 gap.
"""
from __future__ import annotations

import numpy as np

WAKE, LIGHT, DEEP, REM, GAP = 0, 1, 2, 3, 4
SLEEP_STAGE_NAMES = ("light", "deep", "rem")


def divisor_vector(stage_divisor: dict[str, float]) -> np.ndarray:
    """``(wake, light, deep, REM)`` divisor of a ``config.json`` ``stage_divisor``; wake is 1."""
    return np.array([1.0] + [float(stage_divisor[name]) for name in SLEEP_STAGE_NAMES])


def hybrid_calls(
    wake_proba: np.ndarray, stage_proba: np.ndarray, divisor: np.ndarray | None = None
) -> np.ndarray:
    """Per-epoch calls: the wake source's wake and gap calls; its sleep epochs staged by the stage source.

    Both probability arrays are (epochs, 5). With ``divisor`` None the stage
    source's light/deep/REM argmax is used as is.
    """
    calls = np.asarray(wake_proba).argmax(axis=1)
    sleep = (calls >= LIGHT) & (calls <= REM)
    stages = np.asarray(stage_proba)[sleep, LIGHT : REM + 1]
    if divisor is not None:
        stages = stages / np.asarray(divisor)[LIGHT : REM + 1]
    calls[sleep] = LIGHT + stages.argmax(axis=1)
    return calls


def adult_probabilities(proba: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Per-epoch (wake, light, deep, REM) probabilities reported for a divisor-decoded prediction.

    Wake is the model's probability. The rest, 1 − P(wake), is shared among light,
    deep and REM in proportion to their probabilities divided by the divisor. The
    calls come from :func:`hybrid_calls`, not from the argmax of these.
    """
    proba = np.asarray(proba)
    wake = proba[:, WAKE] / proba[:, : REM + 1].sum(axis=1)
    sleep = proba[:, LIGHT : REM + 1] / np.asarray(divisor)[LIGHT : REM + 1]
    sleep = sleep / sleep.sum(axis=1, keepdims=True)
    return np.column_stack([wake, (1.0 - wake)[:, None] * sleep])

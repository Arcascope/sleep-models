"""Adult scoring: sleep/wake from one model, light/deep/REM from another.

The adult formulation keeps two decisions apart:

- **Sleep/wake** comes from a *wake source*, the pediatric release. Its wake and
  gap calls are used unchanged, so sleep/wake metrics equal the release's.
- **Light/deep/REM** comes from a *stage source*, a BiDoze model (or the mean
  output of several seeds) trained on Weaver plus adult nights. Only epochs the
  wake source calls sleep are staged.

The stage source is trained with the class-balanced loss, which scales each
stage's probability by that stage's effective loss weight, and it learns the
stage mix of its training nights. Before the argmax, its light/deep/REM
probabilities are divided by an *adult stage divisor* that removes both:

    divisor_c = w_c × π_train,c / π_adult,c

where w_c is the stage's effective weight in the training batches, π_train its
prevalence in the training nights and π_adult its prevalence in adult PSG.

Stage indices are the model's: 0 wake, 1 light, 2 deep, 3 REM, 4 gap.
"""
from __future__ import annotations

import numpy as np

WAKE, LIGHT, DEEP, REM, GAP = 0, 1, 2, 3, 4
STAGE_COUNT = 4
# Cache labels: 0 wake, 1 N1, 2 N2, 3 N3, 5 REM.
CACHE_LABEL_TO_STAGE = {0: WAKE, 1: LIGHT, 2: LIGHT, 3: DEEP, 5: REM}


def stage_counts(labels: np.ndarray) -> np.ndarray:
    """Epoch count per stage (wake, light, deep, REM) of one night's cache labels."""
    labels = np.asarray(labels).ravel()
    counts = np.zeros(STAGE_COUNT)
    for raw, stage in CACHE_LABEL_TO_STAGE.items():
        counts[stage] += np.sum(labels == raw)
    return counts


def effective_weights(
    night_counts: np.ndarray,
    beta: float,
    nights_per_batch: int = 2,
    draws: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """Each stage's mean loss weight per epoch under the class-balanced loss.

    ``night_counts`` is (nights, 4) stage counts of the training nights. A batch
    of ``nights_per_batch`` random nights weights stage c by
    (1 − β) / (1 − β^n_c), n_c its epoch count in the batch; the result is the
    epoch-weighted mean over ``draws`` batches, E[n_c w_c] / E[n_c].
    """
    counts = np.asarray(night_counts, dtype=float)
    rng = np.random.default_rng(seed)
    batches = counts[rng.integers(len(counts), size=(draws, nights_per_batch))].sum(axis=1)
    weights = np.where(batches > 0, (1 - beta) / (1 - beta**batches), 0.0)
    return (batches * weights).sum(axis=0) / batches.sum(axis=0)


def adult_stage_divisor(
    train_night_counts: np.ndarray,
    adult_night_counts: np.ndarray,
    beta: float,
    nights_per_batch: int = 2,
) -> np.ndarray:
    """w_c × π_train,c / π_adult,c for wake, light, deep and REM."""
    train = np.asarray(train_night_counts, dtype=float)
    adult = np.asarray(adult_night_counts, dtype=float)
    pi_train = train.sum(axis=0) / train.sum()
    pi_adult = adult.sum(axis=0) / adult.sum()
    return effective_weights(train, beta, nights_per_batch) * pi_train / pi_adult


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

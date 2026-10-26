"""Adult scoring: a BiDoze stage source and an adult stage divisor.

The adult model is BiDoze trained on Weaver plus adult nights, with several seeds
whose 5-way outputs are averaged. An epoch is scored in two steps:

- **Sleep/wake**: the model's own most probable of wake, light, deep, REM and gap.
- **Light/deep/REM**: on the epochs it calls sleep, its light, deep and REM
  probabilities are divided by an *adult stage divisor* before the argmax.

The model is trained with the class-balanced loss, which scales each stage's
probability by that stage's effective loss weight, and it learns the stage mix of
its training nights. The divisor removes both:

    divisor_c = w_c × π_train,c / π_adult,c

where w_c is the stage's effective weight in the training batches, π_train its
prevalence in the training nights and π_adult its prevalence in the adult (SleepAccel and Wearanize+) training nights. Only
light, deep and REM are divided: dividing wake as well moves the sleep/wake
operating point by an amount that depends on the cohort's share of wake.

:func:`hybrid_calls` also takes the sleep/wake decision from a different model
(``wake_proba``), as the research comparisons do.

Stage indices are the model's: 0 wake, 1 light, 2 deep, 3 REM, 4 gap.

An adult release is one flat bundle: the members' weights in one ``model.safetensors``
(``member0/…``, ``member1/…``) and the divisor in ``config.json``. The members run as
one batched call (:class:`StackedModel`); ``arcasleep.predict.Predictor`` scores with it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from arcasleep.divisor import (  # noqa: F401  (re-exported)
    DEEP,
    GAP,
    LIGHT,
    REM,
    SLEEP_STAGE_NAMES,
    WAKE,
    adult_probabilities,
    hybrid_calls,
)

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
    with np.errstate(divide="ignore"):
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


class StackedModel:
    """Several BiDoze models of one architecture, scored as one batched call.

    :meth:`infer` returns the log of the models' mean 5-way softmax, so a softmax
    over its output is their mean probability. It stands in for one model wherever
    logits are read.
    """

    def __init__(self, models: Sequence[Any]):
        import jax
        import jax.numpy as jnp

        if not models:
            raise ValueError("StackedModel needs at least one model")
        first = models[0]
        for model in models[1:]:
            if model.config != first.config or model.input_shape != first.input_shape:
                raise ValueError("every stacked model must share one architecture and input shape")
        self.config = first.config
        self.input_shape = first.input_shape
        self._first = first
        params = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *(m.params for m in models))
        module = first.module

        def mean_log_proba(stacked, x, mask):
            logits = jax.vmap(lambda p: module.apply({"params": p}, x, mask, train=False))(stacked)
            # The floor keeps a probability of exactly 0 (the gap rule's present epochs) finite in the log.
            return jnp.log(jnp.maximum(jax.nn.softmax(logits, axis=-1).mean(axis=0), jnp.finfo(jnp.float32).tiny))

        self._params = params
        self._apply = jax.jit(mean_log_proba)
        self.count = len(models)
        self._parameter_count = sum(m.parameter_count() for m in models)

    def parameter_count(self) -> int:
        """Weights of all members together."""
        return self._parameter_count

    def infer(self, input_x: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        import jax.numpy as jnp

        from arcasleep_bidoze.model.definition import _numpy_feature_activity_mask

        input_x = np.asarray(input_x, dtype=np.float32)
        if valid_mask is None:
            if self.config.score_gap:
                valid_mask = np.ones(input_x.shape[:2], dtype=np.float32)
            else:
                valid_mask = _numpy_feature_activity_mask(input_x)
        else:
            valid_mask = self._first._coerce_valid_mask(input_x, valid_mask)
        out = self._apply(self._params, jnp.asarray(input_x), jnp.asarray(valid_mask))
        return np.asarray(out, dtype=np.float32)

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

An *adult bundle* packages the model for ``arcasleep bidoze predict``: one release
bundle per seed, and ``composite.json`` naming them and the divisor. The seeds run
as one batched call (:class:`StackedModel`).
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


ADULT_KIND = "adult"
SLEEP_STAGE_NAMES = ("light", "deep", "rem")


def adult_probabilities(proba: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Per-epoch (wake, light, deep, REM) probabilities reported for an adult prediction.

    Wake is the model's probability. The rest, 1 − P(wake), is shared among light,
    deep and REM in proportion to their probabilities divided by the divisor. The
    calls come from :func:`hybrid_calls`, not from the argmax of these.
    """
    proba = np.asarray(proba)
    wake = proba[:, WAKE] / proba[:, : REM + 1].sum(axis=1)
    sleep = proba[:, LIGHT : REM + 1] / np.asarray(divisor)[LIGHT : REM + 1]
    sleep = sleep / sleep.sum(axis=1, keepdims=True)
    return np.column_stack([wake, (1.0 - wake)[:, None] * sleep])


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


def build_adult_bundle(
    output_dir: Path | str,
    seed_bundles: Sequence[Path | str],
    divisor: Sequence[float],
    *,
    version: str,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy one release bundle per seed into an adult bundle, with ``composite.json``.

    ``divisor`` is (wake, light, deep, REM), as :func:`adult_stage_divisor` returns
    it; the wake entry is not used. ``provenance`` is recorded in ``composite.json``.
    """
    from arcasleep.artifacts.publish import COMPOSITE_FILENAME, verify_bundle
    from arcasleep.models import MODEL_TYPE_KEY

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass overwrite=True to replace it")
        shutil.rmtree(output_dir)
    if not seed_bundles:
        raise ValueError("an adult bundle needs at least one seed")
    for source in seed_bundles:
        verify_bundle(source)
    seed_paths = []
    for k, source in enumerate(seed_bundles):
        seed_paths.append(f"seed{k}")
        shutil.copytree(source, output_dir / seed_paths[-1])
    spec = {
        MODEL_TYPE_KEY: "bidoze",
        "kind": ADULT_KIND,
        "version": version,
        "seeds": seed_paths,
        "stage_divisor": {
            name: float(value)
            for name, value in zip(SLEEP_STAGE_NAMES, list(divisor)[LIGHT : REM + 1], strict=True)
        },
        "provenance": dict(provenance or {}),
    }
    (output_dir / COMPOSITE_FILENAME).write_text(json.dumps(spec, indent=2) + "\n")
    return output_dir


class AdultPredictor:
    """Scores recordings with an adult bundle, as ``arcasleep.predict.Predictor`` does a release.

    The seeds run as one batched call and their 5-way outputs are averaged. The mean
    decides wake and sleep; on its sleep epochs the stage is the argmax of the mean
    light/deep/REM probabilities divided by the divisor.
    """

    def __init__(self, bundle_dir: Path | str):
        from arcasleep.artifacts.publish import COMPOSITE_FILENAME, load_bundle_model
        from arcasleep.predict import Predictor

        self.bundle_dir = Path(bundle_dir)
        spec = json.loads((self.bundle_dir / COMPOSITE_FILENAME).read_text())
        if spec.get("kind") != ADULT_KIND:
            raise ValueError(f"{bundle_dir} is not an adult bundle (kind {spec.get('kind')!r})")
        seeds = [self.bundle_dir / path for path in spec["seeds"]]
        if not seeds:
            raise ValueError(f"{bundle_dir} names no seed")
        processing = (seeds[0] / "preprocessing.json").read_text()
        for seed in seeds[1:]:
            if (seed / "preprocessing.json").read_text() != processing:
                raise ValueError(f"{seed} preprocesses recordings differently from {seeds[0]}")
        self.ensemble = Predictor(seeds[0])
        self.ensemble.model = StackedModel([load_bundle_model(seed) for seed in seeds])
        stage_divisor = spec["stage_divisor"]
        self.divisor = np.array([1.0] + [float(stage_divisor[n]) for n in SLEEP_STAGE_NAMES])
        self.stage_names = self.ensemble.stage_names
        self.min_sampling_hz = self.ensemble.min_sampling_hz

    def predict(
        self,
        accel: np.ndarray,
        origin: float | str | None = None,
        return_gap_head: bool = False,
        *,
        ggir_sleep_filter: bool = True,
        ggir_error_fails: bool = False,
        subject_id: str | None = None,
    ):
        """Score one recording; the arguments are those of ``arcasleep.predict.predict``."""
        from arcasleep.predict import GAP_LABEL_NAME, finish_prediction

        mean = self.ensemble.predict(
            accel, origin, return_gap_head=return_gap_head, ggir_sleep_filter=False
        )
        calls = hybrid_calls(mean.probabilities, mean.probabilities, self.divisor)
        stage_raw = np.asarray(self.stage_names, dtype=object)[calls]
        stage_raw[mean.is_gap] = GAP_LABEL_NAME
        return finish_prediction(
            accel,
            float(mean.epoch_start_s[0]),
            mean.epoch_start_s,
            adult_probabilities(mean.probabilities, self.divisor),
            self.stage_names,
            mean.is_gap,
            stage_raw,
            gap_head=mean.gap_head,
            ggir_sleep_filter=ggir_sleep_filter,
            ggir_error_fails=ggir_error_fails,
            subject_id=subject_id,
        )

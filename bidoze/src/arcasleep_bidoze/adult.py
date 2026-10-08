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

An *adult bundle* packages the formulation for ``arcasleep bidoze predict``: a
directory holding the wake source's release bundle, one release bundle per
stage-source seed, and ``composite.json`` naming them and the divisor.
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


def adult_probabilities(
    wake_proba: np.ndarray, stage_proba: np.ndarray, divisor: np.ndarray
) -> np.ndarray:
    """Per-epoch (wake, light, deep, REM) probabilities reported for an adult prediction.

    Wake is the wake source's probability. The rest, 1 − P(wake), is shared among
    light, deep and REM in proportion to the stage source's probabilities divided by
    the divisor. The calls come from :func:`hybrid_calls`, not from the argmax of these.
    """
    wake = np.asarray(wake_proba)[:, WAKE]
    sleep = np.asarray(stage_proba)[:, LIGHT : REM + 1] / np.asarray(divisor)[LIGHT : REM + 1]
    sleep = sleep / sleep.sum(axis=1, keepdims=True)
    return np.column_stack([wake, (1.0 - wake)[:, None] * sleep])


def build_adult_bundle(
    output_dir: Path | str,
    wake_bundle: Path | str,
    stage_bundles: Sequence[Path | str],
    divisor: Sequence[float],
    *,
    version: str,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy the wake source and stage-source release bundles into one adult bundle.

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
    for source in (wake_bundle, *stage_bundles):
        verify_bundle(source)
    shutil.copytree(wake_bundle, output_dir / "wake")
    stage_paths = []
    for k, source in enumerate(stage_bundles):
        stage_paths.append(f"stage/seed{k}")
        shutil.copytree(source, output_dir / stage_paths[-1])
    spec = {
        MODEL_TYPE_KEY: "bidoze",
        "kind": ADULT_KIND,
        "version": version,
        "wake_source": "wake",
        "stage_sources": stage_paths,
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

    The wake source decides wake and sleep; on its sleep epochs the stage is the
    argmax of the stage sources' mean light/deep/REM probabilities divided by the
    divisor. Each stage source's probabilities are scaled by 1 − its gap head before
    the mean, as in evaluation.
    """

    def __init__(self, bundle_dir: Path | str):
        from arcasleep.artifacts.publish import COMPOSITE_FILENAME
        from arcasleep.predict import Predictor

        self.bundle_dir = Path(bundle_dir)
        spec = json.loads((self.bundle_dir / COMPOSITE_FILENAME).read_text())
        if spec.get("kind") != ADULT_KIND:
            raise ValueError(f"{bundle_dir} is not an adult bundle (kind {spec.get('kind')!r})")
        self.wake = Predictor(self.bundle_dir / spec["wake_source"])
        self.stages = [Predictor(self.bundle_dir / path) for path in spec["stage_sources"]]
        if not self.stages:
            raise ValueError(f"{bundle_dir} names no stage source")
        wake_processing = (self.bundle_dir / spec["wake_source"] / "preprocessing.json").read_text()
        for path in spec["stage_sources"]:
            if (self.bundle_dir / path / "preprocessing.json").read_text() != wake_processing:
                raise ValueError(f"{path} preprocesses recordings differently from the wake source")
        stage_divisor = spec["stage_divisor"]
        self.divisor = np.array([1.0] + [float(stage_divisor[n]) for n in SLEEP_STAGE_NAMES])
        self.stage_names = self.wake.stage_names
        self.min_sampling_hz = self.wake.min_sampling_hz

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
        """Score one recording; the arguments are those of ``arcasleep.predict.predict``.

        ``gap_head``, when asked for, is the wake source's.
        """
        from arcasleep.predict import GAP_LABEL_NAME, finish_prediction

        wake = self.wake.predict(
            accel, origin, return_gap_head=return_gap_head, ggir_sleep_filter=False
        )
        stage = np.mean([self._stage_proba(s, accel, origin) for s in self.stages], axis=0)
        calls = hybrid_calls(wake.probabilities, stage, self.divisor)
        stage_raw = np.asarray(self.stage_names, dtype=object)[calls]
        stage_raw[wake.is_gap] = GAP_LABEL_NAME
        return finish_prediction(
            accel,
            float(wake.epoch_start_s[0]),
            wake.epoch_start_s,
            adult_probabilities(wake.probabilities, stage, self.divisor),
            self.stage_names,
            wake.is_gap,
            stage_raw,
            gap_head=wake.gap_head,
            ggir_sleep_filter=ggir_sleep_filter,
            ggir_error_fails=ggir_error_fails,
            subject_id=subject_id,
        )

    @staticmethod
    def _stage_proba(source, accel: np.ndarray, origin: float | str | None) -> np.ndarray:
        has_gap_head = source.model.config.output_classes > len(source.stage_names)
        result = source.predict(
            accel, origin, return_gap_head=has_gap_head, ggir_sleep_filter=False
        )
        if result.gap_head is None:
            return result.probabilities
        return result.probabilities * (1.0 - result.gap_head)[:, None]

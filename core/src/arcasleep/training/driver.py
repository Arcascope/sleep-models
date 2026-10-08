"""Running a training job: fit the model, then score the held-out subjects.

One call to :func:`run_training` performs the whole recipe — resolve and
snapshot the configs, discover cached subjects, build deterministic
participant-level folds, fit one model per seed, and score every held-out
subject into the run's metrics CSV.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from pisces_lite.cv import Fold, SubjectRef
from pisces_lite.metrics import MetricsLogger
from pisces_lite.model_io import ModelIOBundle
from pisces_lite.utils import pad_or_truncate

from arcasleep.data.batching import (
    pad_recording_input,
    pool_frames_to_epochs,
    recording_input,
)
from arcasleep.data.cache import (
    discover_cached_subjects,
    load_cached_features,
    resolve_feature_cache_dir,
)
from arcasleep.data.constants import GAP_LABEL
from arcasleep.evaluation.predictions import PREDICTIONS_DIRNAME, write_scored_prediction
from arcasleep.evaluation.staging_metrics import GAP_STAGE_INDEX, labels_for_metric_scoring
from arcasleep.training.config import RunConfig
from arcasleep.training.folds import build_dataset_index, build_folds


RULE = "=" * 80


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis, preserving dtype.

    Deliberately does not upcast: logits arrive as float32 and the
    probabilities stay float32.
    """
    logits = np.asarray(logits)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def evaluate_subject(
    model,
    bundle: ModelIOBundle,
    *,
    target_x_length: int,
    target_y_length: int,
    num_classes: int,
    return_context: bool = False,
) -> tuple[np.ndarray, ...]:
    """Score one held-out subject into ``(y_true, y_pred_proba)``.

    The input is built before padding (:func:`recording_input`): the 12 Hz
    config steps every 2 s, so ``X`` carries several frames per 30 s epoch,
    and a model that reads one row per epoch gets them pooled, with missing
    frames left out and recorded in the gap row. Both arrays then pad to fixed
    lengths so JAX traces ``infer`` once rather than once per subject length,
    and pad epochs — genuinely absent data — fold into the gap class beside
    any gap the subject already carried.

    ``return_context`` appends the pooled/padded spectrogram and the rendered
    truth (gap at :data:`GAP_STAGE_INDEX`, not masked) so a caller can draw a
    hypnogram from the arrays that were scored, without a second infer.
    """
    X = bundle.X[0]
    y = bundle.y_for_n_classes(num_classes)[0]
    frames_per_epoch = model.config.frames_per_epoch

    X_input = recording_input(X, y.shape[0], frames_per_epoch)
    X_padded = pad_recording_input(X_input, target_x_length, frames_per_epoch)
    y_padded = pad_or_truncate(y, target_length=target_y_length, pad_value=GAP_LABEL, axis=0)
    y_scored = np.where(y_padded == GAP_LABEL, GAP_STAGE_INDEX, y_padded)

    logits = model.infer(X_padded[None, ...])
    proba = softmax(logits) if model.returns_logits() else np.asarray(logits)

    y_true = labels_for_metric_scoring(y_scored)
    if return_context:
        if frames_per_epoch > 1:
            X_padded = pool_frames_to_epochs(X_padded[None], np.zeros((1, target_x_length)))[0]
        return y_true, proba[0], X_padded, y_scored
    return y_true, proba[0]


def _banner(message: str) -> None:
    print(f"\n{RULE}\n{message}\n{RULE}\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _snapshot_configs(config: RunConfig, results_dir: Path) -> None:
    """Copy every input config into the run so it is self-describing."""
    config_dir = results_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, source in config.paths.items():
        (config_dir / name).write_text(Path(source).read_text())
    _write_json(config_dir / "metrics_effective.json", config.metrics_raw)


@dataclass(frozen=True)
class RunPaths:
    """Where a run reads its cache from and writes its results to.

    The cache's subdirectory is not here: ``cache_prefix`` belongs to the
    feature surface and so is read from ``inputs.json``.
    """

    results: Path
    feature_cache: Path

    def cache_dir(self, config: RunConfig) -> Path:
        return resolve_feature_cache_dir(self.feature_cache, config.cache_prefix)


def run_training(
    config: RunConfig,
    paths: RunPaths,
    *,
    run_id: str,
    split_seed: int | None = None,
) -> Path:
    """Execute the run and return its results directory."""
    training = config.training
    cache_dir = paths.cache_dir(config)
    paths.results.mkdir(parents=True, exist_ok=True)

    _banner(
        f"RUN: {run_id} ({config.family.name})\n"
        f"cv mode: {config.cv.mode}\n"
        f"train={config.cv.train_sets} test={config.cv.test_sets} "
        f"val={config.cv.validation_sets} deblip={config.cv.deblip_sets}\n"
        f"{config.family.describe_training(config)}\n"
        f"feature cache: {cache_dir}"
    )

    cached = discover_cached_subjects(config.cv.all_sets, cache_dir)
    for name, subjects in sorted(cached.items()):
        print(f"  {name}: {len(subjects)} cached subjects")
    features = load_cached_features(
        subject for subjects in cached.values() for subject in subjects
    )
    datasets = build_dataset_index(cached)

    resolved_split_seed = split_seed if split_seed is not None else training.run_seed
    folds, split_summary = build_folds(config.cv, datasets, split_seed=resolved_split_seed)

    _snapshot_configs(config, paths.results)
    _write_json(paths.results / "config" / "split-manifest.json", split_summary)

    logger = MetricsLogger(config.metrics)
    csv_path = paths.results / f"cv_results_{config.model.num_classes}_stages.csv"
    checkpoint_dir = paths.results / "model_outputs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run seeds: {training.seeds} (split seed {resolved_split_seed})")
    for index, fit_seed in enumerate(training.seeds, start=1):
        _banner(f"SEED {index}/{len(training.seeds)}: {fit_seed}")
        for fold in folds:
            _banner(fold.description)
            if not fold.train_subjects:
                print("  SKIP: no training subjects")
                continue
            _run_fold(
                config,
                paths,
                fold=fold,
                fit_seed=fit_seed,
                split_seed=resolved_split_seed,
                features=features,
                datasets=datasets,
                logger=logger,
                csv_path=csv_path,
                checkpoint_dir=checkpoint_dir,
                run_id=run_id,
            )

    print(f"\nResults: {paths.results}")
    return paths.results


def _fold_label(config, fold: Fold) -> str:
    """What to call this fold in checkpoint names and the results CSV.

    ``cv.fold_label`` names the phase ("fold2", "final"). A phase that holds
    several folds -- loso -- still needs the fold number to stay unique, so the
    two are joined. Without a label it is pisces-lite's bare fold_id.
    """
    label = getattr(config.cv, "fold_label", None)
    if not label:
        return str(fold.fold_id)
    return label if fold.total_folds == 1 else f"{label}_{fold.fold_id}"


def _run_fold(
    config: RunConfig,
    paths: RunPaths,
    *,
    fold: Fold,
    fit_seed: int,
    split_seed: int,
    features: Dict[SubjectRef, ModelIOBundle],
    datasets,
    logger,
    csv_path: Path,
    checkpoint_dir: Path,
    run_id: str,
) -> None:
    training = config.training
    num_classes = config.model.num_classes

    seed_fold_id = f"{_fold_label(config, fold)}_seed_{fit_seed}"
    checkpoint_path = checkpoint_dir / f"model_{seed_fold_id}.pisces"

    if checkpoint_path.exists():
        print(f"  found {checkpoint_path.name}; loading and skipping training")
        model = config.family.load_checkpoint(checkpoint_path)
    else:
        started = time.time()
        model = config.family.train_fold(
            config,
            fold=fold,
            fit_seed=fit_seed,
            split_seed=split_seed,
            features=features,
            datasets=datasets,
        )
        print(f"  trained in {time.time() - started:.1f}s")
        model.savez(checkpoint_path)
        print(f"  saved {checkpoint_path}")

    target_x, target_y = config.family.scoring_lengths(
        config, [int(features[ref].y.shape[1]) for ref in fold.test_subjects]
    )
    train_dataset_desc = "+".join(sorted({r.dataset_name for r in fold.train_subjects}))
    predictions_dir = paths.results / PREDICTIONS_DIRNAME / seed_fold_id
    for test_ref in fold.test_subjects:
        y_true, y_pred_proba = evaluate_subject(
            model,
            features[test_ref],
            target_x_length=target_x,
            target_y_length=target_y,
            num_classes=num_classes,
        )
        logger.record(
            csv_path=csv_path,
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            run_id=run_id,
            fold=seed_fold_id,
            epoch=training.epochs,
            subject_id=test_ref.subject_id,
            dataset=test_ref.dataset_name,
            training_dataset=train_dataset_desc,
        )
        write_scored_prediction(
            predictions_dir, test_ref, features[test_ref], y_true, y_pred_proba, num_classes
        )
    print(f"  scored {len(fold.test_subjects)} held-out subjects -> {csv_path.name}")
    print(f"  per-epoch predictions -> {predictions_dir}")

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
from typing import Dict, List

import numpy as np
from pisces_lite.cv import Fold, SubjectRef
from pisces_lite.metrics import MetricsLogger
from pisces_lite.model_io import ModelIOBundle
from pisces_lite.utils import pad_or_truncate

from bidoze.data.batching import assemble_batch, pad_pooled, pool_frames_to_epochs
from bidoze.data.cache import (
    discover_cached_subjects,
    load_cached_features,
    resolve_feature_cache_dir,
)
from bidoze.evaluation.staging_metrics import GAP_STAGE_INDEX, labels_for_metric_scoring
from bidoze.model import BiDozeNetModel
from bidoze.model.constants import GAP_LABEL
from bidoze.training.config import RunConfig
from bidoze.training.folds import (
    build_dataset_index,
    build_folds,
    deblip_subjects_for_fold,
)


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

    Pooling runs before padding: the 12 Hz config steps every 2 s, so ``X``
    carries several frames per 30 s epoch, and pooling leaves missing frames
    out and records them in the gap row. Both arrays then pad to fixed
    lengths so JAX traces ``infer`` once rather than once per subject length,
    and pad epochs — genuinely absent data — fold into the gap class beside
    any gap the subject already carried.

    ``return_context`` appends the pooled/padded spectrogram and the rendered
    truth (gap at :data:`GAP_STAGE_INDEX`, not masked) so a caller can draw a
    hypnogram from the exact arrays that were scored, without a second infer.
    """
    X = bundle.X[0]
    y = bundle.y_for_n_classes(num_classes)[0]

    X_pooled = pool_frames_to_epochs(X[None, ...], y[None, ...])[0]
    X_padded = pad_pooled(X_pooled, target_x_length)
    y_padded = pad_or_truncate(y, target_length=target_y_length, pad_value=GAP_LABEL, axis=0)
    y_scored = np.where(y_padded == GAP_LABEL, GAP_STAGE_INDEX, y_padded)

    logits = model.infer(X_padded[None, ...])
    proba = softmax(logits) if model.returns_logits() else np.asarray(logits)

    y_true = labels_for_metric_scoring(y_scored)
    if return_context:
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
        f"RUN: {run_id}\n"
        f"cv mode: {config.cv.mode}\n"
        f"train={config.cv.train_sets} test={config.cv.test_sets} "
        f"val={config.cv.validation_sets} deblip={config.cv.deblip_sets}\n"
        f"x_length={training.x_length} y_length={training.y_length} "
        f"sliding_copies={training.sliding_copies} "
        f"curriculum={training.curriculum_loss_mode}\n"
        f"deblip: {training.deblip_schedule} "
        f"{training.deblip_adjustment_epochs} epochs @ {training.deblip_logit_weight}\n"
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
    features: Dict[SubjectRef, ModelIOBundle],
    datasets,
    logger,
    csv_path: Path,
    checkpoint_dir: Path,
    run_id: str,
) -> None:
    training = config.training
    num_classes = config.model.num_classes
    target_x, target_y = training.x_length, training.y_length

    plans_by_dataset = (
        config.cv.train_batch_sampling.mode == "balanced_by_dataset"
        or config.cv.subsample_to_smallest
    )
    if plans_by_dataset:
        X_train, y_train, train_dataset_labels = assemble_batch(
            features, fold.train_subjects, target_x, target_y, num_classes,
            with_dataset_labels=True,
        )
    else:
        X_train, y_train = assemble_batch(
            features, fold.train_subjects, target_x, target_y, num_classes
        )
        train_dataset_labels = None

    X_val, y_val = (None, None)
    if fold.val_subjects:
        X_val, y_val = assemble_batch(
            features, fold.val_subjects, target_x, target_y, num_classes
        )

    print(f"  train X: {X_train.shape}  y: {y_train.shape}")
    if X_val is not None:
        print(f"  val   X: {X_val.shape}  y: {y_val.shape}")

    seed_fold_id = f"{_fold_label(config, fold)}_seed_{fit_seed}"
    checkpoint_path = checkpoint_dir / f"model_{seed_fold_id}.pisces"

    if checkpoint_path.exists():
        print(f"  found {checkpoint_path.name}; loading and skipping training")
        model = BiDozeNetModel.loadz(checkpoint_path, input_shape=tuple(X_train.shape[1:]))
    else:
        deblip_x = None
        if training.deblips:
            deblip_refs = deblip_subjects_for_fold(datasets, config.cv.deblip_sets, fold)
            if deblip_refs:
                deblip_x, _ = assemble_batch(
                    features, deblip_refs, target_x, target_y, num_classes
                )
                print(f"  deblip X: {deblip_x.shape}  ({len(deblip_refs)} subjects)")
            else:
                print("  deblip: no subjects left after holding out test participants")

        model = BiDozeNetModel(config.model)
        started = time.time()
        model.train_with_callbacks(
            train_x=X_train,
            train_y=y_train,
            val_x=X_val,
            val_y=y_val,
            initial_lr=training.lr,
            weight_decay=training.weight_decay,
            epochs=training.epochs,
            batch_size=training.batch_size,
            callbacks=[],
            fit_seed=fit_seed,
            sliding_copies=training.sliding_copies,
            train_dataset_labels=train_dataset_labels,
            train_batch_sampling=config.cv.train_batch_sampling.mode,
            subsample_to_smallest=config.cv.subsample_to_smallest,
            deblip_x=deblip_x,
            deblip_epochs=training.deblip_adjustment_epochs,
            deblip_schedule=training.deblip_schedule,
            deblip_pairing_weight=training.deblip_logit_weight,
            deblip_epoch_seconds=config.deblip_epoch_seconds,
            deblip_min_wake_block_minutes=training.deblip_min_wake_block_minutes,
            deblip_max_sleep_minutes=training.deblip_max_sleep_minutes,
            deblip_epsilon=training.deblip_epsilon,
            curriculum_loss_mode=training.curriculum_loss_mode,
            lr_schedule=training.lr_schedule,
            lr_decay_rate=training.lr_decay_rate,
            trainable_scope=training.trainable_scope,
            anchor_l2_weight=training.anchor_l2_weight,
            disable_swa=training.disable_swa,
            save_history_plot=False,
        )
        if training.deblip_schedule == "final" and deblip_x is not None:
            _run_final_deblip(model, deblip_x, training, config, fit_seed)
        print(f"  trained in {time.time() - started:.1f}s")
        model.savez(checkpoint_path)
        print(f"  saved {checkpoint_path}")

    train_dataset_desc = "+".join(sorted({r.dataset_name for r in fold.train_subjects}))
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
    print(f"  scored {len(fold.test_subjects)} held-out subjects -> {csv_path.name}")


def _run_final_deblip(
    model: BiDozeNetModel,
    deblip_x: np.ndarray,
    training,
    config: RunConfig,
    fit_seed: int,
) -> None:
    """The ``deblip_schedule='final'`` variant: one pass after the whole fit."""
    target_logits = model.soft_deblipped_target_logits(
        deblip_x,
        epoch_seconds=config.deblip_epoch_seconds,
        min_wake_block_minutes=training.deblip_min_wake_block_minutes,
        max_sleep_minutes=training.deblip_max_sleep_minutes,
        epsilon=training.deblip_epsilon,
        batch_size=training.batch_size,
    )
    model.adjust_with_callbacks(
        train_x=deblip_x,
        # Deblip carries no stage labels; every epoch is masked and the phase
        # learns only from the soft targets above.
        train_y=np.full(deblip_x.shape[:2], -1, dtype=np.int32),
        initial_lr=training.lr,
        weight_decay=training.weight_decay,
        epochs=training.deblip_adjustment_epochs,
        batch_size=training.batch_size,
        fit_seed=fit_seed,
        target_x=deblip_x,
        target_logits=target_logits,
        target_logit_pairing_weight=training.deblip_logit_weight,
    )

"""The BiDoze training recipe: its ``training.json`` and one fold's fit."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pisces_lite.cv import Fold, SubjectRef
from pisces_lite.model_io import ModelIOBundle

from arcasleep.data.batching import assemble_batch
from arcasleep.training.config import RunConfig, read_json, reject_unknown, require_one_of
from arcasleep.training.folds import DatasetIndex, deblip_subjects_for_fold

DEBLIP_SCHEDULES = frozenset({"interleaved", "final"})
CURRICULUM_LOSS_MODES = frozenset({"staged", "weighted"})
LR_SCHEDULES = frozenset({"cosine", "constant", "exponential"})
TRAINABLE_SCOPES = frozenset({"all", "gate_head"})

# Deblip shape knobs are not spelled in training.json; these mirror the
# trainer's own defaults so an unset config trains identically.
DEFAULT_DEBLIP_MIN_WAKE_BLOCK_MINUTES = 15.0
DEFAULT_DEBLIP_MAX_SLEEP_MINUTES = 5.0
DEFAULT_DEBLIP_EPSILON = 1e-7


@dataclass(frozen=True)
class TrainingConfig:
    """The fit loop's hyperparameters, spelled as ``training.json`` spells them."""

    run_seed: int
    seed_count: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    x_length: int
    y_length: int
    sliding_copies: int = 1
    curriculum_loss_mode: str = "staged"
    deblip_schedule: str = "interleaved"
    deblip_adjustment_epochs: int = 0
    deblip_logit_weight: float = 0.0
    deblip_min_wake_block_minutes: float = DEFAULT_DEBLIP_MIN_WAKE_BLOCK_MINUTES
    deblip_max_sleep_minutes: float = DEFAULT_DEBLIP_MAX_SLEEP_MINUTES
    deblip_epsilon: float = DEFAULT_DEBLIP_EPSILON
    lr_schedule: str = "cosine"
    lr_decay_rate: float = 1.0
    trainable_scope: str = "all"
    anchor_l2_weight: float = 0.0
    disable_swa: bool = False

    REQUIRED_KEYS = (
        "run_seed",
        "seed_count",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "x_length",
        "y_length",
    )

    @classmethod
    def from_json(cls, path: Path | str) -> TrainingConfig:
        return cls.from_dict(read_json(path))

    @classmethod
    def from_dict(cls, data: dict) -> TrainingConfig:
        known = set(cls.__dataclass_fields__)
        reject_unknown(data, known, "training.json")
        missing = set(cls.REQUIRED_KEYS) - set(data)
        if missing:
            raise ValueError(f"training.json: missing keys {sorted(missing)}")
        config = cls(**{k: v for k, v in data.items() if k in known})
        config.validate()
        return config

    def validate(self) -> None:
        at_least_one = {
            "seed_count": self.seed_count,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "x_length": self.x_length,
            "y_length": self.y_length,
            "sliding_copies": self.sliding_copies,
        }
        for name, value in at_least_one.items():
            if int(value) < 1:
                raise ValueError(f"training.json: {name} must be at least 1")

        require_one_of(
            self.curriculum_loss_mode, CURRICULUM_LOSS_MODES, "training.json curriculum_loss_mode"
        )
        require_one_of(self.deblip_schedule, DEBLIP_SCHEDULES, "training.json deblip_schedule")
        require_one_of(self.lr_schedule, LR_SCHEDULES, "training.json lr_schedule")
        require_one_of(self.trainable_scope, TRAINABLE_SCOPES, "training.json trainable_scope")

        if self.lr <= 0.0:
            raise ValueError("training.json: lr must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("training.json: weight_decay must be nonnegative")
        if not 0.0 < self.lr_decay_rate <= 1.0:
            raise ValueError("training.json: lr_decay_rate must be in (0, 1]")
        if self.anchor_l2_weight < 0.0:
            raise ValueError("training.json: anchor_l2_weight must be nonnegative")
        if self.deblip_adjustment_epochs < 0:
            raise ValueError("training.json: deblip_adjustment_epochs must be nonnegative")
        if self.deblip_logit_weight < 0.0:
            raise ValueError("training.json: deblip_logit_weight must be nonnegative")
        if self.deblip_min_wake_block_minutes <= 0.0:
            raise ValueError("training.json: deblip_min_wake_block_minutes must be positive")
        if self.deblip_max_sleep_minutes <= 0.0:
            raise ValueError("training.json: deblip_max_sleep_minutes must be positive")
        if not 0.0 < self.deblip_epsilon < 0.5:
            raise ValueError("training.json: deblip_epsilon must be in (0, 0.5)")

    @property
    def deblips(self) -> bool:
        """Whether this run has a deblip phase at all."""
        return self.deblip_adjustment_epochs > 0 and self.deblip_logit_weight > 0.0

    @property
    def seeds(self) -> list[int]:
        """The fit seeds this run trains, one model each."""
        return [self.run_seed + offset for offset in range(self.seed_count)]


def deblip_epoch_seconds(config: RunConfig) -> float:
    """Seconds per model epoch, which deblip measures its windows in."""
    return float(config.processing.window_step_seconds)


def train_fold(
    config: RunConfig,
    *,
    fold: Fold,
    fit_seed: int,
    features: Mapping[SubjectRef, ModelIOBundle],
    datasets: Mapping[str, DatasetIndex],
):
    """Fit one BiDoze model on ``fold``'s training subjects."""
    from arcasleep_bidoze.model import BiDozeNetModel

    training: TrainingConfig = config.training
    num_classes = config.model.num_classes
    target_x, target_y = training.x_length, training.y_length
    frames_per_epoch = config.model.frames_per_epoch

    plans_by_dataset = (
        config.cv.train_batch_sampling.mode == "balanced_by_dataset"
        or config.cv.subsample_to_smallest
    )
    if plans_by_dataset:
        X_train, y_train, train_dataset_labels = assemble_batch(
            features, fold.train_subjects, target_x, target_y, num_classes,
            frames_per_epoch=frames_per_epoch, with_dataset_labels=True,
        )
    else:
        X_train, y_train = assemble_batch(
            features, fold.train_subjects, target_x, target_y, num_classes,
            frames_per_epoch=frames_per_epoch,
        )
        train_dataset_labels = None

    X_val, y_val = (None, None)
    if fold.val_subjects:
        X_val, y_val = assemble_batch(
            features, fold.val_subjects, target_x, target_y, num_classes,
            frames_per_epoch=frames_per_epoch,
        )

    print(f"  train X: {X_train.shape}  y: {y_train.shape}")
    if X_val is not None:
        print(f"  val   X: {X_val.shape}  y: {y_val.shape}")

    deblip_x = None
    if training.deblips:
        deblip_refs = deblip_subjects_for_fold(datasets, config.cv.deblip_sets, fold)
        if deblip_refs:
            deblip_x, _ = assemble_batch(
                features, deblip_refs, target_x, target_y, num_classes,
                frames_per_epoch=frames_per_epoch,
            )
            print(f"  deblip X: {deblip_x.shape}  ({len(deblip_refs)} subjects)")
        else:
            print("  deblip: no subjects left after holding out test participants")

    model = BiDozeNetModel(config.model)
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
        deblip_epoch_seconds=deblip_epoch_seconds(config),
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
    return model


def _run_final_deblip(
    model,
    deblip_x: np.ndarray,
    training: TrainingConfig,
    config: RunConfig,
    fit_seed: int,
) -> None:
    """The ``deblip_schedule='final'`` variant: one pass after the whole fit."""
    target_logits = model.soft_deblipped_target_logits(
        deblip_x,
        epoch_seconds=deblip_epoch_seconds(config),
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

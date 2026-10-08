"""The BiDoze training recipe: its ``training.json`` and one fold's fit."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from pisces_lite.cv import Fold, SubjectRef
from pisces_lite.model_io import ModelIOBundle

from arcasleep.data.batching import assemble_batch
from arcasleep.training.balanced import Patience, split_train_eval
from arcasleep.training.config import RunConfig, read_json, reject_unknown, require_one_of
from arcasleep.training.folds import DatasetIndex, deblip_subjects_for_fold

DEBLIP_SCHEDULES = frozenset({"interleaved", "final"})
CURRICULUM_LOSS_MODES = frozenset({"staged", "weighted"})
LR_SCHEDULES = frozenset({"cosine", "constant", "exponential"})
TRAINABLE_SCOPES = frozenset({"all", "gate_head"})
# "curriculum" is the staged focal-loss fit with deblip and SWA; "balanced" is
# the class-balanced cross-entropy fit with plateau drops and early stopping on
# a held-out share of the training subjects, or on cv.json's validation sets
# when it names any (Olsen et al., Section II-E).
RECIPES = frozenset({"curriculum", "balanced"})
CURRICULUM_ONLY_KEYS = (
    "sliding_copies",
    "curriculum_loss_mode",
    "deblip_schedule",
    "deblip_adjustment_epochs",
    "deblip_logit_weight",
    "deblip_min_wake_block_minutes",
    "deblip_max_sleep_minutes",
    "deblip_epsilon",
    "lr_schedule",
    "lr_decay_rate",
    "trainable_scope",
    "anchor_l2_weight",
    "disable_swa",
)
BALANCED_ONLY_KEYS = (
    "balance_beta",
    "patience_unit",
    "lr_drop_patience",
    "early_stop_patience",
    "eval_smoothing_steps",
    "eval_fraction",
    "init_weights",
    "distill_from",
)

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
    # Under "balanced", the most the fit may run before it early-stops.
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    x_length: int
    y_length: int
    recipe: str = "curriculum"
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
    balance_beta: float = 0.999
    # The two patiences count `patience_unit`s, "epochs" or optimizer "steps";
    # eval_smoothing_steps > 0 compares the mean eval kappa over that many steps.
    patience_unit: str = "epochs"
    lr_drop_patience: int = 10
    early_stop_patience: int = 25
    eval_smoothing_steps: int = 0
    # Share of the training subjects held out to early-stop on.
    eval_fraction: float = 0.3
    # A release bundle or a checkpoint (.pisces) whose weights the fit starts from; None starts
    # from random weights. The starting weights are scored as epoch 0, so the fit keeps them
    # unless an epoch beats them.
    init_weights: str | None = None
    # A directory of teacher checkpoints (*.pisces); the fit distills their mean output.
    distill_from: str | None = None

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

        require_one_of(self.recipe, RECIPES, "training.json recipe")
        unused = CURRICULUM_ONLY_KEYS if self.recipe == "balanced" else BALANCED_ONLY_KEYS
        defaults = {field.name: field.default for field in fields(self)}
        changed = sorted(name for name in unused if getattr(self, name) != defaults[name])
        if changed:
            raise ValueError(f"training.json: the {self.recipe} recipe does not use {changed}")

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
        if not 0.0 < self.balance_beta < 1.0:
            raise ValueError("training.json: balance_beta must be in (0, 1)")
        try:
            _ = self.patience
        except ValueError as exc:
            raise ValueError(f"training.json: {exc}") from None
        if not 0.0 < self.eval_fraction < 1.0:
            raise ValueError("training.json: eval_fraction must be in (0, 1)")

    @property
    def patience(self) -> Patience:
        """When the balanced fit drops its learning rate and stops."""
        return Patience(
            lr_drop=self.lr_drop_patience,
            early_stop=self.early_stop_patience,
            unit=self.patience_unit,
            smoothing_steps=self.eval_smoothing_steps,
        )

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
    split_seed: int,
    features: Mapping[SubjectRef, ModelIOBundle],
    datasets: Mapping[str, DatasetIndex],
):
    """Fit one BiDoze model on ``fold``'s training subjects."""
    if config.training.recipe == "balanced":
        return _train_fold_balanced(
            config, fold=fold, fit_seed=fit_seed, split_seed=split_seed, features=features
        )

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


def _check_balanced_cv(config: RunConfig) -> None:
    """Refuse ``cv.json`` options that only the curriculum recipe acts on."""
    cv = config.cv
    unused = {
        "deblip_sets": bool(cv.deblip_sets),
        "train_batch_sampling": cv.train_batch_sampling.mode != "disabled",
        "subsample_to_smallest": cv.subsample_to_smallest,
    }
    named = sorted(name for name, used in unused.items() if used)
    if named:
        raise ValueError(f"cv.json: the balanced recipe does not use {named}")


def _train_fold_balanced(
    config: RunConfig,
    *,
    fold: Fold,
    fit_seed: int,
    split_seed: int,
    features: Mapping[SubjectRef, ModelIOBundle],
):
    """The ``recipe='balanced'`` fit, early-stopping on the validation sets or a held-out share."""
    from arcasleep_bidoze.model import BiDozeNetModel

    _check_balanced_cv(config)
    training: TrainingConfig = config.training
    if fold.val_subjects:
        fit_refs, eval_refs = sorted(fold.train_subjects), sorted(fold.val_subjects)
        print(f"  train/eval: {len(fit_refs)} fit, {len(eval_refs)} eval (validation sets)")
    else:
        fit_refs, eval_refs = split_train_eval(
            fold.train_subjects, training.eval_fraction, split_seed
        )
        print(
            f"  train/eval split: {len(fit_refs)} fit, {len(eval_refs)} eval "
            f"(fraction {training.eval_fraction}, seed {split_seed})"
        )
    num_classes = config.model.num_classes
    frames_per_epoch = config.model.frames_per_epoch
    X_fit, y_fit = assemble_batch(
        features, fit_refs, training.x_length, training.y_length, num_classes,
        frames_per_epoch=frames_per_epoch,
    )
    X_eval, y_eval = assemble_batch(
        features, eval_refs, training.x_length, training.y_length, num_classes,
        frames_per_epoch=frames_per_epoch,
    )
    print(f"  fit  X: {X_fit.shape}  y: {y_fit.shape}")
    print(f"  eval X: {X_eval.shape}  y: {y_eval.shape}")
    soft_targets = None
    if training.distill_from is not None:
        soft_targets = teacher_probabilities(Path(training.distill_from), X_fit)
        print(f"  distilling the mean output of the teachers in {training.distill_from}")

    model = BiDozeNetModel(config.model)
    if training.init_weights is not None:
        _load_init_weights(model, Path(training.init_weights), X_fit.shape[1:])
        print(f"  starting from the weights of {training.init_weights}")
    history = model.fit_balanced(
        X_fit,
        y_fit,
        X_eval,
        y_eval,
        batch_size=training.batch_size,
        initial_lr=training.lr,
        weight_decay=training.weight_decay,
        balance_beta=training.balance_beta,
        patience=training.patience,
        max_epochs=training.epochs,
        fit_seed=fit_seed,
        soft_targets=soft_targets,
        score_start=training.init_weights is not None,
    )
    print(f"  best eval kappa {history['best_eval_kappa']:.4f} at epoch {history['best_epoch']}")
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


def _load_init_weights(model, source: Path, input_shape: tuple[int, ...] | None = None) -> None:
    """Load a release bundle's or a checkpoint's (.pisces) weights into ``model``, whose
    architecture must be the source's."""
    import dataclasses
    import json

    from arcasleep.models import MODEL_TYPE_KEY
    from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel

    if source.suffix == ".pisces":
        loaded = BiDozeNetModel.loadz(source, input_shape=input_shape)
        # Compared as JSON: a checkpoint's config holds lists where model.json's may hold tuples.
        def as_json(config) -> dict:
            return json.loads(json.dumps(dataclasses.asdict(config)))

        if as_json(loaded.config) != as_json(model.config):
            raise ValueError(f"model.json differs from the architecture of {source}")
        model.input_shape = loaded.input_shape
        model.params = loaded.params
        model.model = {"params": model.params}
        return
    bundle_config = json.loads((source / "config.json").read_text())
    bundle_config.pop(MODEL_TYPE_KEY, None)
    if BiDozeNetConfig.from_dict(bundle_config) != model.config:
        raise ValueError(f"model.json differs from the architecture of {source}")
    model.load_weights_safetensors(source / "model.safetensors")


def teacher_probabilities(teacher_dir: Path, X: np.ndarray, batch_size: int = 4) -> np.ndarray:
    """The mean 5-way output of every checkpoint in ``teacher_dir`` on the rows of ``X``."""
    from arcasleep_bidoze.adult import StackedModel
    from arcasleep_bidoze.model import BiDozeNetModel

    paths = sorted(teacher_dir.glob("*.pisces"))
    if not paths:
        raise ValueError(f"no teacher checkpoints (*.pisces) in {teacher_dir}")
    teachers = StackedModel([BiDozeNetModel.loadz(p, input_shape=X.shape[1:]) for p in paths])
    log_mean = [teachers.infer(X[i : i + batch_size]) for i in range(0, X.shape[0], batch_size)]
    return np.exp(np.concatenate(log_mean, axis=0))

"""The UNet training recipe: its ``training.json`` and one fold's fit."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pisces_lite.cv import Fold, SubjectRef
from pisces_lite.model_io import ModelIOBundle

from arcasleep.data.batching import recording_input
from arcasleep.training.config import RunConfig, reject_unknown
from arcasleep.training.folds import DatasetIndex


@dataclass(frozen=True)
class TrainingConfig:
    """The fit loop's hyperparameters, spelled as ``training.json`` spells them."""

    run_seed: int
    seed_count: int
    max_epochs: int
    batch_size: int
    lr: float
    segment_epochs: int
    balance_beta: float = 0.999
    lr_drop_patience: int = 10
    early_stop_patience: int = 25
    # Share of the training subjects held out to early-stop on.
    eval_fraction: float = 0.3

    REQUIRED_KEYS = ("run_seed", "seed_count", "max_epochs", "batch_size", "lr", "segment_epochs")

    @classmethod
    def from_dict(cls, data: dict) -> TrainingConfig:
        reject_unknown(data, set(cls.__dataclass_fields__), "training.json")
        missing = set(cls.REQUIRED_KEYS) - set(data)
        if missing:
            raise ValueError(f"training.json: missing keys {sorted(missing)}")
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("seed_count", "max_epochs", "batch_size", "segment_epochs"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"training.json: {name} must be at least 1")
        if self.lr <= 0.0:
            raise ValueError("training.json: lr must be positive")
        if not 0.0 < self.balance_beta < 1.0:
            raise ValueError("training.json: balance_beta must be in (0, 1)")
        if self.lr_drop_patience < 1 or self.early_stop_patience < 1:
            raise ValueError("training.json: patiences must be at least 1")
        if not 0.0 < self.eval_fraction < 1.0:
            raise ValueError("training.json: eval_fraction must be in (0, 1)")

    @property
    def seeds(self) -> list[int]:
        """The fit seeds this run trains, one model each."""
        return [self.run_seed + offset for offset in range(self.seed_count)]

    @property
    def epochs(self) -> int:
        """The epoch count a results row records: the most the fit may run."""
        return self.max_epochs


def split_train_eval(
    refs: Sequence[SubjectRef], eval_fraction: float, split_seed: int
) -> tuple[list[SubjectRef], list[SubjectRef]]:
    """Hold out ``eval_fraction`` of the subjects to early-stop on, seeded by ``split_seed``.

    The split is by subject id, so a subject listed under several datasets
    lands on one side.
    """
    groups: dict[str, list[SubjectRef]] = {}
    for ref in refs:
        groups.setdefault(ref.subject_id, []).append(ref)
    group_refs = list(groups.values())

    order = np.random.default_rng(split_seed).permutation(len(group_refs))
    eval_count = max(1, int(round(eval_fraction * len(group_refs))))
    eval_refs = sorted(ref for i in order[:eval_count] for ref in group_refs[i])
    fit_refs = sorted(ref for i in order[eval_count:] for ref in group_refs[i])
    if not fit_refs:
        raise ValueError("the train/eval split left no subjects to fit on")
    return fit_refs, eval_refs


def _nights(
    features: Mapping[SubjectRef, ModelIOBundle],
    refs: Sequence[SubjectRef],
    num_classes: int,
    frames_per_epoch: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each subject's ``(frames, labels)`` at its own length."""
    nights = []
    for ref in refs:
        bundle = features[ref]
        y = np.asarray(bundle.y_for_n_classes(num_classes)[0], dtype=np.int32)
        x = recording_input(bundle.X[0], y.shape[0], frames_per_epoch)
        nights.append((np.asarray(x, dtype=np.float32), y))
    return nights


def _check_cv(config: RunConfig) -> None:
    """Refuse ``cv.json`` options that only the BiDoze recipe acts on."""
    cv = config.cv
    unused = {
        "deblip_sets": bool(cv.deblip_sets),
        "validation_sets": bool(cv.validation_sets),
        "train_batch_sampling": cv.train_batch_sampling.mode != "disabled",
        "subsample_to_smallest": cv.subsample_to_smallest,
    }
    named = sorted(name for name, used in unused.items() if used)
    if named:
        raise ValueError(f"cv.json: the unet recipe does not use {named}")


def train_fold(
    config: RunConfig,
    *,
    fold: Fold,
    fit_seed: int,
    split_seed: int,
    features: Mapping[SubjectRef, ModelIOBundle],
    datasets: Mapping[str, DatasetIndex],
):
    """Fit one UNet on ``fold``'s training subjects, early-stopping on a held-out share."""
    from arcasleep_unet.model import UNetModel

    _check_cv(config)
    training: TrainingConfig = config.training
    model_config = config.model
    fit_refs, eval_refs = split_train_eval(
        fold.train_subjects, training.eval_fraction, split_seed
    )
    print(
        f"  train/eval split: {len(fit_refs)} fit, {len(eval_refs)} eval "
        f"(fraction {training.eval_fraction}, seed {split_seed})"
    )
    num_classes = model_config.num_classes
    frames_per_epoch = model_config.frames_per_epoch
    model = UNetModel(model_config)
    history = model.fit(
        _nights(features, fit_refs, num_classes, frames_per_epoch),
        _nights(features, eval_refs, num_classes, frames_per_epoch),
        segment_epochs=training.segment_epochs,
        batch_size=training.batch_size,
        initial_lr=training.lr,
        balance_beta=training.balance_beta,
        lr_drop_patience=training.lr_drop_patience,
        early_stop_patience=training.early_stop_patience,
        max_epochs=training.max_epochs,
        fit_seed=fit_seed,
    )
    print(f"  best eval kappa {history['best_eval_kappa']:.4f} at epoch {history['best_epoch']}")
    return model

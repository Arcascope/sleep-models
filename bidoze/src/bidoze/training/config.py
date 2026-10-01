"""Resolved, validated run configuration.

Five JSON files in ``bidoze/configs/`` define a run, and they are the
contract this module is written against:

``inputs.json``    the feature surface, read as ``ProcessingConfig``
``model.json``     the architecture, read as ``BiDozeNetConfig``
``cv.json``        which datasets train, deblip, validate, and test
``training.json``  the fit loop's hyperparameters
``metrics.json``   the evaluation protocol

:class:`RunConfig` loads all five at once; that is the entry point the trainer
uses. Unknown keys in ``cv.json`` and ``training.json`` are an error rather
than a warning, so a config written against a different trainer fails loudly
instead of quietly training something else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from pisces_lite.cv import CVConfig
from pisces_lite.metrics import MetricsConfig
from pisces_lite.proc import ProcessingConfig

from bidoze.model import BiDozeNetConfig


CONFIG_FILENAMES = ("inputs.json", "model.json", "cv.json", "training.json", "metrics.json")

DEBLIP_SCHEDULES = frozenset({"interleaved", "final"})
CURRICULUM_LOSS_MODES = frozenset({"staged", "weighted"})
LR_SCHEDULES = frozenset({"cosine", "constant", "exponential"})
TRAINABLE_SCOPES = frozenset({"all", "gate_head"})
CV_MODES = frozenset({"transfer", "loso"})

# Deblip shape knobs are not spelled in training.json; these mirror the
# trainer's own defaults so an unset config trains identically.
DEFAULT_DEBLIP_MIN_WAKE_BLOCK_MINUTES = 15.0
DEFAULT_DEBLIP_MAX_SLEEP_MINUTES = 5.0
DEFAULT_DEBLIP_EPSILON = 1e-7

DEFAULT_WASO_BOUT_MAX_WAKE_GAP_MINUTES = 30.0


def _read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def _reject_unknown(data: dict, known: set[str], what: str) -> None:
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{what}: unknown keys {sorted(unknown)}")


def _require_one_of(value: str, allowed: frozenset[str], what: str) -> None:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {sorted(allowed)}, got {value!r}")


@dataclass(frozen=True)
class TrainBatchSamplingConfig:
    """How an epoch's batches are planned when the train pool mixes datasets."""

    mode: str = "disabled"
    epoch_size: str = "cover_largest_once"
    replacement: str = "smaller_datasets"
    uneven_batches: str = "fill_randomly"

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrainBatchSamplingConfig":
        if data is None:
            return cls()
        _reject_unknown(data, set(cls.__dataclass_fields__), "cv.json train_batch_sampling")
        config = cls(**{k: str(v) for k, v in data.items()})
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode == "disabled":
            return
        allowed = {
            "mode": frozenset({"balanced_by_dataset"}),
            "epoch_size": frozenset({"cover_largest_once"}),
            "replacement": frozenset({"smaller_datasets"}),
            "uneven_batches": frozenset({"fill_randomly"}),
        }
        for name, values in allowed.items():
            _require_one_of(getattr(self, name), values, f"cv.json train_batch_sampling.{name}")


@dataclass
class BiDozeCVConfig:
    """Dataset-level split plan, plus the deblip cohorts.

    A dataset may appear in both ``train_sets`` and ``test_sets``; the fold
    builder splits it by participant. ``deblip_sets`` may likewise overlap
    ``test_sets`` — deblip supplies no labels, and the fold builder keeps it to
    the train side of that participant split.
    """

    mode: str
    train_sets: List[str]
    test_sets: List[str]
    validation_sets: List[str] = field(default_factory=list)
    deblip_sets: List[str] = field(default_factory=list)
    max_folds: int | None = None
    # k for the materialised k-fold plan. Training never reads it -- the
    # generated per-fold configs name their datasets outright -- but the split
    # generator does, so k lives with the rest of the split policy.
    n_splits: int | None = None
    # Names the phase this config is: "fold2", "final". Each phase is its own
    # `bidoze train` invocation holding a single fold, so pisces-lite's
    # fold_id is 0 in every one of them; without a label, results from six
    # different models would all record fold 0.
    fold_label: str | None = None
    train_batch_sampling: TrainBatchSamplingConfig = field(default_factory=TrainBatchSamplingConfig)
    subsample_to_smallest: bool = False

    @classmethod
    def from_json(cls, path: Path | str) -> "BiDozeCVConfig":
        return cls.from_dict(_read_json(path))

    @classmethod
    def from_dict(cls, data: dict) -> "BiDozeCVConfig":
        _reject_unknown(data, set(cls.__dataclass_fields__), "cv.json")
        resolved = dict(data)
        resolved["train_batch_sampling"] = TrainBatchSamplingConfig.from_dict(
            resolved.get("train_batch_sampling")
        )
        if not isinstance(resolved.get("subsample_to_smallest", False), bool):
            raise ValueError("cv.json: subsample_to_smallest must be a bool")
        config = cls(**resolved)
        config.validate()
        return config

    def validate(self) -> None:
        train = set(self.train_sets)
        test = set(self.test_sets)
        val = set(self.validation_sets)
        deblip = set(self.deblip_sets)

        _require_one_of(self.mode, CV_MODES, "cv.json mode")
        if not train:
            raise ValueError("cv.json: train_sets is empty")
        if not test:
            raise ValueError("cv.json: test_sets is empty")
        if train & val:
            raise ValueError(
                f"cv.json: train_sets and validation_sets overlap on {sorted(train & val)}"
            )
        if deblip & val:
            raise ValueError(
                f"cv.json: deblip_sets and validation_sets overlap on {sorted(deblip & val)}"
            )
        if self.mode == "loso":
            self.to_pisces_config().validate()
        self.train_batch_sampling.validate()
        if self.subsample_to_smallest and self.train_batch_sampling.mode != "disabled":
            raise ValueError(
                "cv.json: subsample_to_smallest and train_batch_sampling both plan the "
                "epoch's batches; enable only one"
            )

    def to_pisces_config(self) -> CVConfig:
        return CVConfig(
            mode=self.mode,
            train_sets=self.train_sets,
            test_sets=self.test_sets,
            validation_sets=self.validation_sets,
            max_folds=self.max_folds,
        )

    @property
    def all_sets(self) -> List[str]:
        return sorted(
            set(self.train_sets)
            | set(self.deblip_sets)
            | set(self.test_sets)
            | set(self.validation_sets)
        )


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
    def from_json(cls, path: Path | str) -> "TrainingConfig":
        return cls.from_dict(_read_json(path))

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        known = set(cls.__dataclass_fields__)
        _reject_unknown(data, known, "training.json")
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

        _require_one_of(
            self.curriculum_loss_mode, CURRICULUM_LOSS_MODES, "training.json curriculum_loss_mode"
        )
        _require_one_of(self.deblip_schedule, DEBLIP_SCHEDULES, "training.json deblip_schedule")
        _require_one_of(self.lr_schedule, LR_SCHEDULES, "training.json lr_schedule")
        _require_one_of(self.trainable_scope, TRAINABLE_SCOPES, "training.json trainable_scope")

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
    def seeds(self) -> List[int]:
        """The fit seeds this run trains, one model each."""
        return [self.run_seed + offset for offset in range(self.seed_count)]


def load_inputs(path: Path | str) -> tuple[ProcessingConfig, str]:
    """Read ``inputs.json`` into ``(processing config, cache prefix)``.

    ``cache_prefix`` names the directory under the feature-cache root that this
    feature surface's NPZs land in, so one root can hold several. It is
    required: defaulting it would silently read whichever cache happened to sit
    at the root.
    """
    data = _read_json(path)
    kind = str(data.get("type", "")).lower()
    if kind != "nufft":
        raise ValueError(f"inputs.json: bidoze supports type='nufft' only, got {kind!r}")

    prefix = data.get("cache_prefix")
    if not prefix or not str(prefix).strip():
        raise ValueError("inputs.json: missing 'cache_prefix'")

    # ProcessingConfig ignores keys it does not declare, so cache_prefix rides
    # along in the same file without disturbing it.
    return ProcessingConfig.from_dict(data), str(prefix).strip()


def load_processing_config(path: Path | str) -> ProcessingConfig:
    """Read just the processing config out of ``inputs.json``."""
    return load_inputs(path)[0]


def load_model_config(path: Path | str) -> BiDozeNetConfig:
    """Read ``model.json`` into the architecture config."""
    data = _read_json(path)
    data.pop("name", None)
    return BiDozeNetConfig.from_dict(data)


def load_metrics_config(
    path: Path | str, num_classes: int | None = None
) -> tuple[MetricsConfig, dict[str, Any]]:
    """Read ``metrics.json``, returning the config and the resolved raw dict.

    The raw dict is kept because the run snapshots it as an artifact, and
    because some metric specs are built from keys the dataclass does not carry.
    """
    data = _read_json(path)
    if num_classes is not None:
        data.setdefault("num_classes", num_classes)
    data.setdefault("waso_bout_max_wake_gap_minutes", DEFAULT_WASO_BOUT_MAX_WAKE_GAP_MINUTES)
    return MetricsConfig.from_dict(data), data


@dataclass(frozen=True)
class RunConfig:
    """Every config a run needs, loaded and validated together."""

    processing: ProcessingConfig
    cache_prefix: str
    model: BiDozeNetConfig
    cv: BiDozeCVConfig
    training: TrainingConfig
    metrics: MetricsConfig
    metrics_raw: dict[str, Any]
    paths: dict[str, Path]

    @classmethod
    def from_paths(
        cls,
        *,
        inputs: Path | str,
        model: Path | str,
        cv: Path | str,
        training: Path | str,
        metrics: Path | str,
    ) -> "RunConfig":
        paths = {
            "inputs.json": Path(inputs),
            "model.json": Path(model),
            "cv.json": Path(cv),
            "training.json": Path(training),
            "metrics.json": Path(metrics),
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("missing config files:\n  " + "\n  ".join(missing))

        model_config = load_model_config(paths["model.json"])
        metrics_config, metrics_raw = load_metrics_config(
            paths["metrics.json"], num_classes=model_config.num_classes
        )
        processing_config, cache_prefix = load_inputs(paths["inputs.json"])
        return cls(
            processing=processing_config,
            cache_prefix=cache_prefix,
            model=model_config,
            cv=BiDozeCVConfig.from_json(paths["cv.json"]),
            training=TrainingConfig.from_json(paths["training.json"]),
            metrics=metrics_config,
            metrics_raw=metrics_raw,
            paths=paths,
        )

    @property
    def deblip_epoch_seconds(self) -> float:
        """Seconds per model epoch, which deblip measures its windows in."""
        return float(self.processing.window_step_seconds)

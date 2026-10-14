"""Resolved, validated run configuration.

Five JSON files define a run, and they are the contract this module is
written against:

``inputs.json``    the feature surface, read as ``ProcessingConfig``
``model.json``     the architecture, read by the model family
``cv.json``        which datasets train, deblip, validate, and test
``training.json``  the fit loop's hyperparameters, read by the model family
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
from typing import TYPE_CHECKING, Any, List

from pisces_lite.cv import CVConfig as PiscesCVConfig
from pisces_lite.metrics import MetricsConfig
from pisces_lite.proc import ProcessingConfig

if TYPE_CHECKING:
    from arcasleep.models import ModelConfig, ModelFamily, TrainingRecipe


CONFIG_FILENAMES = ("inputs.json", "model.json", "cv.json", "training.json", "metrics.json")

CV_MODES = frozenset({"transfer", "loso"})

DEFAULT_WASO_BOUT_MAX_WAKE_GAP_MINUTES = 30.0


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def reject_unknown(data: dict, known: set[str], what: str) -> None:
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{what}: unknown keys {sorted(unknown)}")


def require_one_of(value: str, allowed: frozenset[str], what: str) -> None:
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
        reject_unknown(data, set(cls.__dataclass_fields__), "cv.json train_batch_sampling")
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
            require_one_of(getattr(self, name), values, f"cv.json train_batch_sampling.{name}")


@dataclass
class CVConfig:
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
    # `train` invocation holding a single fold, so pisces-lite's
    # fold_id is 0 in every one of them; without a label, results from six
    # different models would all record fold 0.
    fold_label: str | None = None
    train_batch_sampling: TrainBatchSamplingConfig = field(
        default_factory=TrainBatchSamplingConfig
    )
    subsample_to_smallest: bool = False

    @classmethod
    def from_json(cls, path: Path | str) -> "CVConfig":
        return cls.from_dict(read_json(path))

    @classmethod
    def from_dict(cls, data: dict) -> "CVConfig":
        reject_unknown(data, set(cls.__dataclass_fields__), "cv.json")
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

        require_one_of(self.mode, CV_MODES, "cv.json mode")
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

    def to_pisces_config(self) -> PiscesCVConfig:
        return PiscesCVConfig(
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


def load_inputs(path: Path | str) -> tuple[ProcessingConfig, str]:
    """Read ``inputs.json`` into ``(processing config, cache prefix)``.

    ``cache_prefix`` names the directory under the feature-cache root that this
    feature surface's NPZs land in, so one root can hold several. It is
    required: defaulting it would silently read whichever cache happened to sit
    at the root.
    """
    data = read_json(path)
    kind = str(data.get("type", "")).lower()
    if kind != "nufft":
        raise ValueError(f"inputs.json: only type='nufft' is supported, got {kind!r}")

    prefix = data.get("cache_prefix")
    if not prefix or not str(prefix).strip():
        raise ValueError("inputs.json: missing 'cache_prefix'")

    return processing_config(data, "inputs.json"), str(prefix).strip()


def processing_config(data: dict, what: str) -> ProcessingConfig:
    """``data`` as a ``ProcessingConfig``; refuses settings the installed pisces-lite does not know.

    ``ProcessingConfig.from_dict`` drops keys it does not declare, so an older
    pisces-lite would otherwise compute different spectra from the ones ``data``
    describes. ``cache_prefix`` names the feature cache and is the one other key
    allowed.
    """
    unknown = set(data) - set(ProcessingConfig.__dataclass_fields__) - {"cache_prefix"}
    if unknown:
        raise ValueError(
            f"{what} sets {sorted(unknown)}, which the installed pisces-lite does not "
            "support; upgrade pisces-lite to the version in arcasleep's requirements"
        )
    return ProcessingConfig.from_dict(data)


def load_processing_config(path: Path | str) -> ProcessingConfig:
    """Read just the processing config out of ``inputs.json``."""
    return load_inputs(path)[0]


def load_model_config(family: ModelFamily, path: Path | str) -> ModelConfig:
    """Read ``model.json`` into the family's architecture config."""
    data = read_json(path)
    data.pop("name", None)
    return family.model_config(data)


def load_metrics_config(
    path: Path | str, num_classes: int | None = None
) -> tuple[MetricsConfig, dict[str, Any]]:
    """Read ``metrics.json``, returning the config and the resolved raw dict.

    The raw dict is kept because the run snapshots it as an artifact, and
    because some metric specs are built from keys the dataclass does not carry.
    """
    data = read_json(path)
    if num_classes is not None:
        data.setdefault("num_classes", num_classes)
    data.setdefault(
        "waso_bout_max_wake_gap_minutes", DEFAULT_WASO_BOUT_MAX_WAKE_GAP_MINUTES
    )
    return MetricsConfig.from_dict(data), data


@dataclass(frozen=True)
class RunConfig:
    """Every config a run needs, loaded and validated together."""

    family: ModelFamily
    processing: ProcessingConfig
    cache_prefix: str
    model: ModelConfig
    cv: CVConfig
    training: TrainingRecipe
    metrics: MetricsConfig
    metrics_raw: dict[str, Any]
    paths: dict[str, Path]

    @classmethod
    def from_paths(
        cls,
        *,
        family: ModelFamily,
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

        model_config = load_model_config(family, paths["model.json"])
        metrics_config, metrics_raw = load_metrics_config(
            paths["metrics.json"], num_classes=model_config.num_classes
        )
        processing_config, cache_prefix = load_inputs(paths["inputs.json"])
        return cls(
            family=family,
            processing=processing_config,
            cache_prefix=cache_prefix,
            model=model_config,
            cv=CVConfig.from_json(paths["cv.json"]),
            training=family.training_config(read_json(paths["training.json"])),
            metrics=metrics_config,
            metrics_raw=metrics_raw,
            paths=paths,
        )

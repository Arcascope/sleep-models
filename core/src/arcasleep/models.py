"""The model families arcasleep trains, scores and releases.

Each family lives in its own import package and exposes a :class:`ModelFamily`
instance named ``FAMILY`` from a ``family`` module. The shared pipeline reaches
a model only through that object, so a model's dependencies are imported only
when a command needs them, and the CLI can list every family on a base install.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pisces_lite.cv import Fold, SubjectRef
    from pisces_lite.model_io import ModelIOBundle

    from arcasleep.training.config import RunConfig
    from arcasleep.training.folds import DatasetIndex


#: Family name -> module holding its ``FAMILY``.
FAMILY_MODULES: dict[str, str] = {
    "bidoze": "arcasleep_bidoze.family",
}

#: The family a release bundle or checkpoint belongs to when it does not say.
DEFAULT_FAMILY = "bidoze"

#: The key in a release bundle's ``config.json`` naming its family.
MODEL_TYPE_KEY = "model_type"


class ModelConfig(Protocol):
    """What the shared pipeline reads from a model's architecture config."""

    num_classes: int
    frames_per_epoch: int

    @property
    def output_classes(self) -> int: ...


class TrainingRecipe(Protocol):
    """What the shared pipeline reads from a family's ``training.json``."""

    run_seed: int

    @property
    def seeds(self) -> list[int]: ...

    @property
    def epochs(self) -> int:
        """The epoch count a results row records."""
        ...


class SleepModel(Protocol):
    """A trained network as the shared pipeline uses it."""

    config: ModelConfig
    input_shape: tuple[int, ...] | None

    def infer(self, input_x) -> Any: ...

    def returns_logits(self) -> bool: ...

    def parameter_count(self) -> int: ...

    def savez(self, filepath: Path | str) -> None: ...

    def save_weights_safetensors(self, path: Path) -> None: ...


class ModelFamily(ABC):
    """One model architecture and its training recipe."""

    #: The CLI group and the bundle ``model_type``.
    name: str
    #: The ``model_class`` its ``.pisces`` checkpoints record.
    checkpoint_class: str

    @abstractmethod
    def model_config(self, data: Mapping[str, Any]) -> ModelConfig:
        """Parse ``model.json``."""

    @abstractmethod
    def training_config(self, data: Mapping[str, Any]) -> TrainingRecipe:
        """Parse ``training.json``."""

    @abstractmethod
    def scoring_lengths(self, config: RunConfig, epoch_counts: Sequence[int]) -> tuple[int, int]:
        """``(x_length, y_length)`` in epochs for scoring recordings of these lengths."""

    @abstractmethod
    def describe_training(self, config: RunConfig) -> str:
        """Lines for the run banner."""

    @abstractmethod
    def train_fold(
        self,
        config: RunConfig,
        *,
        fold: Fold,
        fit_seed: int,
        features: Mapping[SubjectRef, ModelIOBundle],
        datasets: Mapping[str, DatasetIndex],
    ) -> SleepModel:
        """Fit one model on ``fold``'s training subjects."""

    @abstractmethod
    def load_checkpoint(
        self, path: Path, input_shape: tuple[int, ...] | None = None
    ) -> SleepModel:
        """Load a ``.pisces`` checkpoint."""

    @abstractmethod
    def load_release(self, config: Mapping[str, Any], weights: Path) -> SleepModel:
        """Load a release bundle's ``config.json`` and ``model.safetensors``."""

    @abstractmethod
    def describe_architecture(
        self, config: Mapping[str, Any], parameter_count: int | None
    ) -> list[str]:
        """Model card lines on what the network is, read from its config."""


def family_names() -> list[str]:
    return sorted(FAMILY_MODULES)


def get_family(name: str) -> ModelFamily:
    """The family called ``name``, importing its package."""
    try:
        module_name = FAMILY_MODULES[name]
    except KeyError:
        raise ValueError(
            f"unknown model family {name!r}; expected one of {family_names()}"
        ) from None
    return importlib.import_module(module_name).FAMILY


def release_family(config: Mapping[str, Any]) -> str:
    """The family a release bundle's ``config.json`` belongs to."""
    return str(config.get(MODEL_TYPE_KEY, DEFAULT_FAMILY))


def checkpoint_family(path: Path | str) -> str:
    """The family a ``.pisces`` checkpoint belongs to, from its recorded model class.

    A class no family claims is read as :data:`DEFAULT_FAMILY`, whose loader
    then decides whether the weights fit.
    """
    import json
    import zipfile

    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    model_class = metadata.get("model_class")
    for name in family_names():
        if get_family(name).checkpoint_class == model_class:
            return name
    return DEFAULT_FAMILY

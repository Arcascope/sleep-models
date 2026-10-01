"""Arcascope's sleep-stage models: shared data, training, evaluation and release tooling."""

from importlib.metadata import PackageNotFoundError, version as _distribution_version

try:
    __version__ = _distribution_version("arcasleep")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

"""The feature set of ``inputs.json``: which columns are computed, and in what order.

``features.at_time`` lists values of the epoch alone. Every other key of ``features`` is a
per-epoch signal; its ``b_spans`` and ``f_spans`` map a statistic to window lengths in minutes.
A ``b`` window ends at the epoch, an ``f`` window starts after it. A window feature is named
``<signal>_<stat>_<b|f><minutes>``.

Columns come in the order written: ``at_time``, then each signal's ``b_spans`` and ``f_spans``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from arcasleep_tree.features import constants as C

INPUTS_TYPE = "watchtree"

#: Values of the epoch alone that ``at_time`` may name.
AT_TIME = frozenset({
    "jerk", "jerk_peak", "lfm", "angle", "angle_step", "daz", "sib",
    "hr", "hr_std", "hr_n", "hr_missing", "hr_base", "hr_rel",
    "breath_bpm", "breath_peak_share", "breath_entropy", "breath_log_power", "breath_valid",
    "tsb_min", "tnb_min",
})  # fmt: skip
#: Per-epoch signals a window may summarise.
SIGNALS = frozenset({
    "jerk", "jerk_peak", "lfm", "act", "angle", "angle_step", "daz", "sib",
    "hr", "hr_rel", "hr_valid", "breath_bpm", "breath_peak_share", "breath_valid",
})  # fmt: skip
STATS = frozenset({"mean", "max", "std"})
DIRECTIONS = {"b_spans": "b", "f_spans": "f"}
EXPECTED_KEYS = {"type", "features", "resample", "uncropped"}
DEFAULT_NAME = "inputs.json"

@dataclass(frozen=True)
class Window:
    """One windowed feature: ``stat`` of ``signal`` over ``minutes`` before or after the epoch."""

    signal: str
    stat: str
    direction: str
    minutes: int

    @property
    def name(self) -> str:
        return f"{self.signal}_{self.stat}_{self.direction}{self.minutes}"

    @property
    def epochs(self) -> int:
        return self.minutes * 60 // C.EPOCH_S


@dataclass(frozen=True)
class FeatureConfig:
    """What the feature table holds and how recordings are read for it."""

    at_time: tuple[str, ...]
    windows: tuple[Window, ...]
    #: Whether the accelerometer is resampled onto a uniform grid before the features.
    resample: bool
    #: Whether the whole wrist recording is kept, with epochs outside the PSG span unlabelled.
    uncropped: bool

    @property
    def names(self) -> list[str]:
        """The feature columns, in order."""
        return list(self.at_time) + [w.name for w in self.windows]

    @property
    def signals(self) -> set[str]:
        """The per-epoch signals the windows need."""
        return {w.signal for w in self.windows}

    @classmethod
    def from_json(cls, path: Path | str) -> FeatureConfig:
        return cls.from_dict(json.loads(Path(path).read_text()), source=str(path))

    @classmethod
    def from_dict(cls, data: dict, source: str = DEFAULT_NAME) -> FeatureConfig:
        """``source`` names where ``data`` came from, in error messages."""
        _check_keys(data, EXPECTED_KEYS, source)
        if data["type"] != INPUTS_TYPE:
            raise ValueError(f"{source}: type must be {INPUTS_TYPE!r}, got {data['type']!r}")
        features = dict(data["features"])
        at_time = tuple(features.pop("at_time", ()))
        unknown = sorted(set(at_time) - AT_TIME)
        if unknown:
            raise ValueError(f"{source}: unknown at_time features {unknown}")
        windows = []
        for signal, spans in features.items():
            if signal not in SIGNALS:
                raise ValueError(f"{source}: unknown signal {signal!r}")
            _check_keys(spans, set(), f"{source} {signal}", optional=set(DIRECTIONS))
            for key, direction in DIRECTIONS.items():
                for stat, minutes in spans.get(key, {}).items():
                    if stat not in STATS:
                        raise ValueError(f"{source}: {signal} {key}: unknown statistic {stat!r}")
                    windows += [Window(signal, stat, direction, int(m)) for m in minutes]
        config = cls(at_time, tuple(windows), bool(data["resample"]), bool(data["uncropped"]))
        config.validate(source)
        return config

    def validate(self, source: str = DEFAULT_NAME) -> None:
        names = self.names
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"{source}: features listed twice {repeated}")
        for w in self.windows:
            if w.minutes < 1:
                raise ValueError(f"{source}: {w.name}: a window is at least 1 minute")
            if w.direction == "f" and w.minutes > C.FORWARD_MAX_MIN:
                raise ValueError(
                    f"{source}: {w.name}: forward windows reach at most "
                    f"{C.FORWARD_MAX_MIN} minutes"
                )


def _check_keys(data: dict, required: set[str], what: str, optional: set[str] = frozenset()) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{what}: missing keys {missing}")
    unknown = sorted(set(data) - required - optional)
    if unknown:
        raise ValueError(f"{what}: unknown keys {unknown}")

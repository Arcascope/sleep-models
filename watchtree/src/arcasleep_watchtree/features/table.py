"""The WatchTree feature table: definitions, per-epoch computation and on-watch cost.

One row per 30-s epoch. A feature at epoch ``k`` uses data up to ``FORWARD_MAX_MIN`` minutes
after the end of epoch ``k`` and nothing later; the stager scores epoch ``k`` once that much
has been recorded. No feature depends on the position of the epoch in the recording.

Names are ``<signal>[_<stat>_<b|f><minutes>]``: ``b`` is a backward window that ends at the
epoch, ``f`` a forward window that starts after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arcasleep_watchtree.features import constants as C
from arcasleep_watchtree.features.baseline import low_baseline
from arcasleep_watchtree.features.breathing import breathing_features
from arcasleep_watchtree.features.heartrate import epoch_heart_rate
from arcasleep_watchtree.features.preprocess import epoch_signals
from arcasleep_watchtree.features.windows import window_stat

EPOCHS_PER_MIN = 60 // C.EPOCH_S

#: (signal, stat, direction, minutes): the windowed features, in column order.
WINDOWS: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    ("jerk", "mean", "b", (1, 5, 15, 30, 60, 120)),
    ("jerk", "max", "b", (5, 30, 120)),
    ("jerk", "mean", "f", (5, 15, 45)),
    ("jerk", "max", "f", (5, 15, 45)),
    ("lfm", "mean", "b", (5, 15, 60)),
    ("lfm", "max", "b", (15,)),
    ("lfm", "mean", "f", (5, 15, 45)),
    ("act", "mean", "b", (5, 15, 60, 120)),
    ("act", "mean", "f", (5, 15, 45)),
    ("angle", "std", "b", (15, 60)),
    ("angle", "std", "f", (15, 45)),
    ("angle_step", "mean", "b", (5, 15, 30, 60, 120)),
    ("angle_step", "max", "b", (15, 60)),
    ("angle_step", "mean", "f", (5, 15, 45)),
    ("angle_step", "max", "f", (15, 45)),
    ("daz", "mean", "b", (15, 60)),
    ("daz", "mean", "f", (15, 45)),
    ("sib", "mean", "b", (5, 15, 60)),
    ("sib", "mean", "f", (15,)),
    ("hr", "mean", "b", (5, 15, 30, 60)),
    ("hr", "std", "b", (5, 15, 60)),
    ("hr", "mean", "f", (5, 15, 45)),
    ("hr", "std", "f", (15, 45)),
    ("hr_rel", "mean", "b", (5, 15, 60)),
    ("hr_rel", "mean", "f", (15, 45)),
    ("hr_valid", "mean", "b", (15,)),
    ("hr_valid", "mean", "f", (15,)),
    ("breath_bpm", "mean", "b", (5, 15)),
    ("breath_bpm", "std", "b", (15,)),
    ("breath_bpm", "mean", "f", (15,)),
    ("breath_bpm", "std", "f", (15,)),
    ("breath_peak_share", "mean", "b", (5, 15)),
    ("breath_peak_share", "mean", "f", (15,)),
    ("breath_valid", "mean", "b", (15,)),
    ("breath_valid", "mean", "f", (15,)),
)

#: Features of the epoch alone (no window), in column order.
CURRENT = (
    "jerk", "jerk_peak", "lfm", "angle", "angle_step", "daz", "sib",
    "hr", "hr_std", "hr_n", "hr_missing", "hr_base", "hr_rel",
    "breath_bpm", "breath_peak_share", "breath_entropy", "breath_log_power", "breath_valid",
    "tsb_min", "tnb_min",
)  # fmt: skip

#: Ops per epoch to update one window of each statistic: add the new epoch, drop the oldest,
#: keep the count, and divide (std adds the sum of squares and a square root). A maximum scans
#: the window.
COST_MEAN = 6
COST_STD = 14


def window_name(signal: str, stat: str, direction: str, minutes: int) -> str:
    return f"{signal}_{stat}_{direction}{minutes}"


def feature_names() -> list[str]:
    names = list(CURRENT)
    for signal, stat, direction, minutes in WINDOWS:
        names += [window_name(signal, stat, direction, m) for m in minutes]
    return names


def bout_timing(act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minutes since the last high-activity epoch (capped) and until the next (capped).

    ``act`` is 1 for a bout epoch. With no bout so far the first is the cap, so the value does
    not count time from the start of the recording. The second looks ahead at most
    ``FORWARD_MAX_MIN`` minutes.
    """
    n_epochs = act.size
    since = np.empty(n_epochs)
    until = np.empty(n_epochs)
    last = -(10**9)
    reach = C.FORWARD_MAX_MIN * EPOCHS_PER_MIN
    for k in range(n_epochs):
        if act[k] == 1:
            last = k
        since[k] = min((k - last) / EPOCHS_PER_MIN, C.TSB_CAP_MIN)
        ahead = np.flatnonzero(act[k + 1 : k + 1 + reach] == 1)
        until[k] = ahead[0] + 1 if ahead.size else reach
    return since, until / EPOCHS_PER_MIN


def compute_features(
    acc25: np.ndarray, valid25: np.ndarray, hr_t: np.ndarray, hr_bpm: np.ndarray
) -> pd.DataFrame:
    """The feature table of one night.

    ``acc25`` is ``(n_epochs * 750, 3)`` in g on the 25 Hz grid, ``valid25`` marks the samples
    that are not inside a gap, and ``hr_t`` (seconds from the start of epoch 0) and ``hr_bpm``
    are the heart-rate samples, which may be empty.
    """
    sig = epoch_signals(acc25, valid25)
    n_epochs = sig.valid.size
    present = sig.valid

    hr, hr_std, hr_n = epoch_heart_rate(hr_t, hr_bpm, n_epochs)
    hr_valid = np.where(present, (hr_n > 0).astype(float), np.nan)
    base = low_baseline(hr)
    hr_rel = hr - base
    breath = breathing_features(sig.breath_input, sig.breath_still)
    breath_valid = np.where(present, np.isfinite(breath["breath_bpm"]).astype(float), np.nan)
    act = np.where(
        present, (sig.jerk >= C.BOUT_RATIO * low_baseline(sig.jerk)).astype(float), np.nan
    )
    angle = sig.angle
    daz = np.abs(np.diff(angle, prepend=angle[:1]))
    since, until = bout_timing(np.nan_to_num(act, nan=0.0))

    series = {
        "jerk": sig.jerk, "jerk_peak": sig.jerk_peak, "lfm": sig.lfm, "act": act,
        "angle": angle, "angle_step": sig.angle_step, "daz": daz, "sib": sig.still_fraction,
        "hr": hr, "hr_rel": hr_rel, "hr_valid": hr_valid, **breath, "breath_valid": breath_valid,
    }  # fmt: skip
    columns: dict[str, np.ndarray] = {
        "jerk": sig.jerk, "jerk_peak": sig.jerk_peak, "lfm": sig.lfm, "angle": angle,
        "angle_step": sig.angle_step, "daz": daz, "sib": sig.still_fraction,
        "hr": hr, "hr_std": hr_std, "hr_n": hr_n.astype(float),
        "hr_missing": (hr_n == 0).astype(float), "hr_base": base, "hr_rel": hr_rel,
        **breath, "breath_valid": breath_valid, "tsb_min": since, "tnb_min": until,
    }  # fmt: skip
    for signal, stat, direction, minutes in WINDOWS:
        for m in minutes:
            n = m * EPOCHS_PER_MIN
            columns[window_name(signal, stat, direction, m)] = window_stat(
                series[signal], stat, direction, n
            )
    table = pd.DataFrame({name: columns[name] for name in feature_names()})
    table.insert(0, "epoch", np.arange(n_epochs))
    return table.astype({name: np.float32 for name in feature_names()})


@dataclass(frozen=True)
class CostLine:
    """One row of the on-watch cost table: what it computes and its ops per 30-s epoch."""

    family: str
    what: str
    ops_per_epoch: float


def stream_costs() -> list[CostLine]:
    """Per-sample work, converted to ops per epoch (a multiply-add, or an add/compare, is 1 op)."""
    sec = C.EPOCH_S
    return [
        CostLine(
            "preprocess",
            "jerk: 3 subtract, 3 multiply, 2 add, sqrt (4), 1 add; 25 Hz",
            13 * C.FS * sec,
        ),
        CostLine(
            "preprocess", "stage-1 biquads: 3 axes x 2 sections x 5 MAC; 25 Hz", 30 * C.FS * sec
        ),
        CostLine(
            "movement",
            "low-frequency motion on 5 Hz: 3 subtract, 3 multiply, 2 add, sqrt (4), add; sums (3)",
            16 * C.FS_1 * sec,
        ),
        CostLine(
            "angle",
            "6 block angles per epoch (3 multiply, 2 add, sqrt, atan2 about 20 ops) and steps",
            6 * 30,
        ),
        CostLine(
            "breathing", "stage-2 biquads: 3 axes x 2 sections x 5 MAC; 5 Hz", 30 * C.FS_1 * sec
        ),
        CostLine("breathing", "high-pass, 3 axes x 3 ops; 2.5 Hz", 9 * C.FS_2 * sec),
        CostLine(
            "breathing", "Goertzel: 16 bins x 3 axes x 3 ops; 2.5 Hz", 16 * 3 * 3 * C.FS_2 * sec
        ),
        CostLine(
            "breathing",
            "epoch end: bin powers (16 x 3 x 5), peak, parabola, 16 ln, entropy",
            16 * 3 * 5 + 16 * 12 + 40,
        ),
        CostLine(
            "heart rate",
            "per sample: sum, sum of squares, count (1 Hz); epoch mean and std",
            3 * sec + 12,
        ),
        CostLine(
            "heart rate",
            "baseline: block mean, lowest 3 of 12 (about 36 ops per 20 epochs), subtract",
            4,
        ),
        CostLine(
            "bouts",
            "jerk floor (4), bout test (2), time since last bout (3), time to next bout (scan, 90)",
            4 + 2 + 3 + 90,
        ),
    ]


def window_costs() -> list[CostLine]:
    lines = []
    for signal, stat, direction, minutes in WINDOWS:
        for m in minutes:
            n = m * EPOCHS_PER_MIN
            ops = n + 1 if stat == "max" else COST_STD if stat == "std" else COST_MEAN
            lines.append(CostLine("windows", window_name(signal, stat, direction, m), ops))
    return lines

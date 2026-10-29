"""The on-watch features of an ``inputs.json``, their integer units and the numbers the Monkey C app shares.

The watch has no breathing features. Every feature is an integer on the watch; ``SCALE[name]`` is how
many integer units one original unit (g, degree, bpm, minute, fraction) is. A model's bin edges
are the thresholds of its trees converted to these integers, so the watch only compares integers.

Signals (per 30-s epoch) and their integer units:

========== =================== ==================================
signal     unit                meaning
========== =================== ==================================
jerk       1/16 mg             mean of ``||a[n] - a[n-1]||``
jerk_peak  1/16 mg             largest 1-s mean of that
lfm        1/64 mg             mean distance of 5 Hz low-passed samples from their 5-s mean
angle      1/16 degree         mean of six 5-s z-angles
angle_step 1/1024 degree      mean absolute change of the 5-s z-angle
daz        1/1024 degree        absolute change of ``angle`` from the previous epoch
sib        1/1024              share of 5-s steps under 5 degrees
act        1/1024              1 when ``jerk`` is at least 3 times its low baseline
hr         1/16 bpm            mean valid heart rate
hr_rel     1/16 bpm            ``hr`` minus its low baseline
hr_valid   1/1024              1 when the epoch has a heart-rate sample
========== =================== ==================================
"""

from __future__ import annotations

from arcasleep_tree.features import constants as C
from arcasleep_tree.features.config import FeatureConfig
from arcasleep_tree.features.table import BREATH_PREFIX, EPOCHS_PER_MIN

#: Marks an epoch or window with no value.
NA = -(2**31)

#: Integer units per original unit of each signal (as a mean or maximum).
SIGNAL_SCALE = {
    "jerk": 16000, "jerk_peak": 16000, "lfm": 64000, "angle": 16, "angle_step": 1024, "daz": 1024,
    "sib": 1024, "act": 1024, "hr": 16, "hr_rel": 16, "hr_valid": 1024,
}  # fmt: skip
#: A standard deviation of a windowed signal is given with 16 times the signal's resolution.
STD_EXTRA = 16

CURRENT_SCALE = {
    "jerk": 16000, "jerk_peak": 16000, "lfm": 64000, "angle": 16, "angle_step": 1024, "daz": 1024,
    "sib": 1024, "hr": 16, "hr_std": 16, "hr_n": 1, "hr_missing": 1, "hr_base": 16, "hr_rel": 16,
    "tsb_min": EPOCHS_PER_MIN, "tnb_min": EPOCHS_PER_MIN,
}  # fmt: skip

#: Signal ids, in the order the Monkey C app numbers its rings.
SIGNALS = (
    "jerk", "jerk_peak", "lfm", "angle", "angle_step", "daz", "sib", "act",
    "hr", "hr_std", "hr_n", "hr_base", "hr_rel", "hr_valid", "tsb",
)  # fmt: skip
STATS = ("mean", "max", "std")


def spec(config: FeatureConfig) -> list[dict]:
    """One dict per feature of ``config``: name, kind (``cur`` or ``win``), signal, stat, direction, epochs, scale.

    The watch has no breathing path, so a breathing feature is refused.
    """
    breathing = [n for n in config.names if n.startswith(BREATH_PREFIX)]
    if breathing:
        raise ValueError(f"the watch does not compute breathing features: {breathing}")
    rows = []
    for name in config.at_time:
        signal = {"tsb_min": "tsb", "tnb_min": "tnb", "hr_missing": "hr_n"}.get(name, name)
        rows.append(dict(name=name, kind="cur", signal=signal, stat="", dir="", n=0,
                         scale=CURRENT_SCALE[name]))  # fmt: skip
    for w in config.windows:
        scale = SIGNAL_SCALE[w.signal] * (STD_EXTRA if w.stat == "std" else 1)
        rows.append(dict(name=w.name, kind="win", signal=w.signal, stat=w.stat, dir=w.direction,
                         n=w.epochs, scale=scale))  # fmt: skip
    return rows


def scales(config: FeatureConfig) -> dict[str, int]:
    """Integer units per original unit of each feature of ``config``."""
    return {r["name"]: r["scale"] for r in spec(config)}

#: Epochs of scoring delay: epoch ``k`` is scored when epoch ``k + DELAY`` has ended.
DELAY = C.FORWARD_MAX_MIN * EPOCHS_PER_MIN
#: Epochs before which a window is not complete (2 h).
WARM = 120 * EPOCHS_PER_MIN

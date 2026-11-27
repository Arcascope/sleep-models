"""Per-epoch features of a wrist recording, as the watch computes them, and their integer units.

One row per 30-s epoch. A feature at epoch ``k`` uses data up to ``FORWARD_MAX_MIN`` minutes
after the end of epoch ``k`` and nothing later; the watch scores epoch ``k`` once that much has
been recorded. ``inputs`` may set the reach of the features with the keys of ``REACH``. No
feature depends on the position of the epoch in the recording.

``INPUTS`` lists the columns. ``at_time`` names values of the epoch alone; every other key of
``features`` is a per-epoch signal whose ``b_spans`` and ``f_spans`` map a statistic to window
lengths in minutes. A ``b`` window ends at, and includes, the epoch; an ``f`` window starts at
the next epoch. A window feature is named ``<signal>_<stat>_<b|f><minutes>``. Columns come in the
order written: ``at_time``, then each signal's ``b_spans`` and ``f_spans``.

The Instinct delivers 25 Hz, 3-axis acceleration in g. Other cohorts are brought to that stream
by :func:`to_25hz` (offline only; the watch never resamples). Everything after that is the
arithmetic the watch runs:

* jerk, ``||a[n] - a[n-1]||``, on the 25 Hz stream;
* two cascaded biquads per axis (4-pole Butterworth, 1.5 Hz), state started at the first sample,
  then every 5th output (5 Hz);
* low-frequency motion (mean distance of the 5 Hz samples from their 5-s block mean) and 5-s
  block z-angles on the 5 Hz stream.

Every feature is an integer on the watch: ``scales(inputs)[name]`` integer units per original
unit (g, degree, bpm, minute, fraction). Signals and their integer units:

========== ============ ==================================================
signal     unit         meaning
========== ============ ==================================================
jerk       1/16 mg      mean of ``||a[n] - a[n-1]||``
jerk_peak  1/16 mg      largest 1-s mean of that
lfm        1/64 mg      mean distance of 5 Hz samples from their 5-s mean
angle      1/16 degree  mean of six 5-s z-angles
angle_step 1/1024 deg   mean absolute change of the 5-s z-angle
daz        1/1024 deg   absolute change of ``angle`` from the previous epoch
sib        1/1024       share of 5-s steps under 5 degrees
act        1/1024       1 when ``jerk`` is at least 3 times its low baseline
hr         1/16 bpm     mean valid heart rate
hr_rel     1/16 bpm     ``hr`` minus its low baseline
hr_valid   1/1024       1 when the epoch has a heart-rate sample
========== ============ ==================================================
"""

from __future__ import annotations

from math import gcd

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

EPOCH_S = 30
EPOCHS_PER_MIN = 60 // EPOCH_S
#: Accelerometer rate the Instinct delivers and every cohort is brought to.
FS = 25
SAMPLES_PER_EPOCH = EPOCH_S * FS
#: A gap in the raw stream longer than this is not interpolated; the last value is held and the
#: samples are marked invalid.
GAP_S = 1.0
#: An epoch with a smaller fraction of valid 25 Hz samples has no movement or angle features.
MIN_VALID_FRACTION = 0.5

#: 4-pole Butterworth low-pass at 1.5 Hz on the 25 Hz stream, then every 5th output.
DECIMATE = 5
FS_LOW = FS // DECIMATE
LP_HZ = 1.5
SOS = butter(4, LP_HZ, fs=FS, output="sos")

#: An epoch is a high-activity bout when its mean jerk is at least this multiple of its low
#: baseline.
BOUT_RATIO = 3.0
#: Angle: mean of each 5-s block of the 5 Hz stream, in degrees; a change under this is "still".
ANGLE_BLOCK_SAMPLES = 25
STILL_DEG = 5.0

#: Low baseline: blocks of 20 epochs (10 min); the mean of the lowest 3 of the last 12 completed
#: blocks (about the lowest quartile of the last 2 h). A block with under 5 epochs of data is
#: skipped.
BLOCK_EPOCHS = 20
BASELINE_BLOCKS = 12
BASELINE_LOWEST = 3
BLOCK_MIN_EPOCHS = 5

HR_MIN_BPM = 30.0
HR_MAX_BPM = 200.0

#: Time since the last bout is capped here; time to the next bout is capped at the forward reach.
TSB_CAP_MIN = 240
FORWARD_MAX_MIN = 45
#: Epochs of scoring delay on the watch: epoch ``k`` is scored when epoch ``k + DELAY`` has ended.
DELAY = FORWARD_MAX_MIN * EPOCHS_PER_MIN

#: Keys of ``inputs`` that set the reach of the features, and their defaults. ``back_min`` is the
#: recording a training epoch needs before it, ``baseline_blocks`` the blocks of the low baseline,
#: ``tsb_cap_min`` the cap of ``tsb_min`` and ``forward_max_min`` the longest forward window and
#: the scoring delay.
REACH = {
    "back_min": 120,
    "baseline_blocks": BASELINE_BLOCKS,
    "tsb_cap_min": TSB_CAP_MIN,
    "forward_max_min": FORWARD_MAX_MIN,
}


def reach(inputs: dict, key: str) -> int:
    """The value of the ``REACH`` key ``key`` in ``inputs``."""
    return int(inputs.get(key, REACH[key]))


def delay(inputs: dict) -> int:
    """Epochs of scoring delay of ``inputs``."""
    return reach(inputs, "forward_max_min") * EPOCHS_PER_MIN


INPUTS = {
    "type": "tree",
    "features": {
        "at_time": [
            "jerk", "jerk_peak", "lfm", "angle", "angle_step", "daz", "sib",
            "hr", "hr_std", "hr_n", "hr_missing", "hr_base", "hr_rel",
            "tsb_min", "tnb_min",
        ],
        "jerk": {
            "b_spans": {"mean": [1, 5, 15, 30, 60, 120], "max": [5, 30, 120]},
            "f_spans": {"mean": [5, 15, 45], "max": [5, 15, 45]},
        },
        "lfm": {"b_spans": {"mean": [5, 15, 60], "max": [15]}, "f_spans": {"mean": [5, 15, 45]}},
        "act": {"b_spans": {"mean": [5, 15, 60, 120]}, "f_spans": {"mean": [5, 15, 45]}},
        "angle": {"b_spans": {"std": [15, 60]}, "f_spans": {"std": [15, 45]}},
        "angle_step": {
            "b_spans": {"mean": [5, 15, 30, 60, 120], "max": [15, 60]},
            "f_spans": {"mean": [5, 15, 45], "max": [15, 45]},
        },
        "daz": {"b_spans": {"mean": [15, 60]}, "f_spans": {"mean": [15, 45]}},
        "sib": {"b_spans": {"mean": [5, 15, 60]}, "f_spans": {"mean": [15]}},
        "hr": {
            "b_spans": {"mean": [5, 15, 30, 60], "std": [5, 15, 60]},
            "f_spans": {"mean": [5, 15, 45], "std": [15, 45]},
        },
        "hr_rel": {"b_spans": {"mean": [5, 15, 60]}, "f_spans": {"mean": [15, 45]}},
        "hr_valid": {"b_spans": {"mean": [15]}, "f_spans": {"mean": [15]}},
    },
    "uncropped": True,
}  # fmt: skip

#: Values of the epoch alone that ``at_time`` may name, and their integer units.
AT_TIME_SCALE = {
    "jerk": 16000, "jerk_peak": 16000, "lfm": 64000, "angle": 16, "angle_step": 1024, "daz": 1024,
    "sib": 1024, "hr": 16, "hr_std": 16, "hr_n": 1, "hr_missing": 1, "hr_base": 16, "hr_rel": 16,
    "tsb_min": EPOCHS_PER_MIN, "tnb_min": EPOCHS_PER_MIN,
}  # fmt: skip
#: Per-epoch signals a window may summarise, and their integer units (as a mean or maximum).
SIGNAL_SCALE = {
    "jerk": 16000, "jerk_peak": 16000, "lfm": 64000, "angle": 16, "angle_step": 1024, "daz": 1024,
    "sib": 1024, "act": 1024, "hr": 16, "hr_rel": 16, "hr_valid": 1024,
}  # fmt: skip
#: A standard deviation of a windowed signal is given with 16 times the signal's resolution.
STD_EXTRA = 16
STATS = ("mean", "max", "std")
DIRECTIONS = {"b_spans": "b", "f_spans": "f"}


# ---------------------------------------------------------------------------------- the inputs


def windows(inputs: dict) -> list[dict]:
    """The window features of ``inputs``, in column order.

    Each is a dict of name, signal, stat, dir (``b`` or ``f``) and epochs.
    """
    out = []
    for signal, spans in inputs["features"].items():
        if signal == "at_time":
            continue
        if signal not in SIGNAL_SCALE:
            raise ValueError(f"unknown signal {signal!r}")
        for key, direction in DIRECTIONS.items():
            for stat, minutes in spans.get(key, {}).items():
                if stat not in STATS:
                    raise ValueError(f"{signal} {key}: unknown statistic {stat!r}")
                for m in minutes:
                    if direction == "f" and m > reach(inputs, "forward_max_min"):
                        raise ValueError(
                            f"{signal}: forward windows reach at most "
                            f"{reach(inputs, 'forward_max_min')} minutes"
                        )
                    name = f"{signal}_{stat}_{direction}{m}"
                    epochs = m * EPOCHS_PER_MIN
                    out.append(
                        dict(name=name, signal=signal, stat=stat, dir=direction, epochs=epochs)
                    )
    return out


def feature_names(inputs: dict = INPUTS) -> list[str]:
    """The feature columns, in order."""
    at_time = list(inputs["features"].get("at_time", []))
    unknown = sorted(set(at_time) - set(AT_TIME_SCALE))
    if unknown:
        raise ValueError(f"unknown at_time features {unknown}")
    return at_time + [w["name"] for w in windows(inputs)]


def scales(inputs: dict = INPUTS) -> dict[str, int]:
    """Integer units per original unit of each feature."""
    out = {name: AT_TIME_SCALE[name] for name in inputs["features"].get("at_time", [])}
    for w in windows(inputs):
        out[w["name"]] = SIGNAL_SCALE[w["signal"]] * (STD_EXTRA if w["stat"] == "std" else 1)
    return out


# ---------------------------------------------------------------------------------- the stream


def _grid_values(t: np.ndarray, xyz: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of ``xyz`` onto ``grid``; across a gap the last value is held.

    Returns the values and a mask that is False inside a gap (more than ``GAP_S`` between the
    samples either side) and outside the recording.
    """
    prev = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, t.size - 1)
    nxt = np.minimum(prev + 1, t.size - 1)
    span = t[nxt] - t[prev]
    inside = (grid >= t[0]) & (grid <= t[-1])
    ok = inside & (span <= GAP_S)
    frac = np.where(span > 0, (grid - t[prev]) / np.where(span > 0, span, 1.0), 0.0)
    frac = np.where(ok, np.clip(frac, 0.0, 1.0), 0.0)
    values = xyz[prev] + frac[:, None] * (xyz[nxt] - xyz[prev])
    return values, ok


def nominal_rate(t: np.ndarray) -> int:
    """The integer rate, in Hz, nearest to the stream's median sampling rate."""
    steps = np.diff(t)
    steps = steps[steps > 0]
    return max(1, int(round(1.0 / float(np.median(steps)))))


def to_25hz(
    t: np.ndarray, xyz: np.ndarray, t0: float, n_epochs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a stream to the 25 Hz grid ``t0 + n / 25``, ``n_epochs`` epochs long.

    The stream is placed on a uniform grid at its nominal rate (linear interpolation), then
    reduced to 25 Hz by a polyphase Kaiser FIR low-pass. A 25 Hz stream skips the second step.
    Returns ``(n, 3)`` values and the mask of samples that are not inside a gap.
    """
    n_out = n_epochs * SAMPLES_PER_EPOCH
    fs_in = nominal_rate(t)
    if fs_in == FS:
        values, _ = _grid_values(t, xyz, t0 + np.arange(n_out) / FS)
    else:
        n_in = n_epochs * EPOCH_S * fs_in
        dense, _ = _grid_values(t, xyz, t0 + np.arange(n_in) / fs_in)
        g = gcd(FS, fs_in)
        values = resample_poly(dense, FS // g, fs_in // g, axis=0, padtype="line")[:n_out]
    _, valid = _grid_values(t, xyz, t0 + np.arange(n_out) / FS)
    return values, valid


def _biquads(x: np.ndarray) -> np.ndarray:
    """Cascaded biquads along axis 0, state started at the steady state of the first sample."""
    zi = sosfilt_zi(SOS)[:, :, None] * x[0][None, None, :]
    return sosfilt(SOS, x, axis=0, zi=zi)[0]


def _angle_blocks(low5: np.ndarray) -> np.ndarray:
    """z-angle in degrees, ``atan(z / sqrt(x^2 + y^2))``, of each 5-s block mean."""
    n_blocks = low5.shape[0] // ANGLE_BLOCK_SAMPLES
    mean = (
        low5[: n_blocks * ANGLE_BLOCK_SAMPLES]
        .reshape(n_blocks, ANGLE_BLOCK_SAMPLES, 3)
        .mean(axis=1)
    )
    return np.degrees(np.arctan2(mean[:, 2], np.sqrt(mean[:, 0] ** 2 + mean[:, 1] ** 2)))


def epoch_signals(acc: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Per-epoch movement and angle signals of a 25 Hz stream.

    A signal is NaN where too few samples are valid.
    """
    n_epochs = acc.shape[0] // SAMPLES_PER_EPOCH
    acc = acc[: n_epochs * SAMPLES_PER_EPOCH]
    present = (
        valid[: n_epochs * SAMPLES_PER_EPOCH].reshape(n_epochs, -1).mean(axis=1)
        >= MIN_VALID_FRACTION
    )

    step = np.diff(acc, axis=0, prepend=acc[:1])
    jerk = np.sqrt(np.sum(step * step, axis=1)).reshape(n_epochs, -1)
    jerk_seconds = jerk.reshape(n_epochs, -1, FS).mean(axis=2)

    low5 = _biquads(acc)[DECIMATE - 1 :: DECIMATE][: n_epochs * EPOCH_S * FS_LOW]
    blocks = low5.reshape(-1, ANGLE_BLOCK_SAMPLES, 3)
    deviation = blocks - blocks.mean(axis=1, keepdims=True)
    lfm = np.sqrt(np.sum(deviation * deviation, axis=2)).mean(axis=1).reshape(n_epochs, -1)

    angles = _angle_blocks(low5)
    angle_step = np.abs(np.diff(angles, prepend=angles[:1])).reshape(n_epochs, -1)

    def masked(values: np.ndarray) -> np.ndarray:
        return np.where(present, values, np.nan)

    return {
        "present": present,
        "jerk": masked(jerk.mean(axis=1)),
        "jerk_peak": masked(jerk_seconds.max(axis=1)),
        "lfm": masked(lfm.mean(axis=1)),
        "angle": masked(angles.reshape(n_epochs, -1).mean(axis=1)),
        "angle_step": masked(angle_step.mean(axis=1)),
        "sib": masked((angle_step < STILL_DEG).mean(axis=1)),
    }


def epoch_heart_rate(
    t: np.ndarray, bpm: np.ndarray, n_epochs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, within-epoch standard deviation and number of valid samples per epoch.

    ``t`` is seconds from the start of epoch 0. Samples outside 30–200 bpm are dropped; an epoch
    with no valid sample has NaN heart rate.
    """
    t = np.asarray(t, dtype=np.float64)
    bpm = np.asarray(bpm, dtype=np.float64)
    keep = np.isfinite(bpm) & (bpm >= HR_MIN_BPM) & (bpm <= HR_MAX_BPM)
    keep &= (t >= 0) & (t < n_epochs * EPOCH_S)
    k = (t[keep] // EPOCH_S).astype(int)
    count = np.bincount(k, minlength=n_epochs).astype(np.float64)
    total = np.bincount(k, weights=bpm[keep], minlength=n_epochs)
    total_sq = np.bincount(k, weights=bpm[keep] ** 2, minlength=n_epochs)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        var = total_sq / count - (total / count) ** 2
    std = np.where(count > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return mean, std, count


def low_baseline(x: np.ndarray, n_blocks: int = BASELINE_BLOCKS) -> np.ndarray:
    """Mean of the lowest 3 of the last ``n_blocks`` (12) completed 10-min block means of ``x``.

    NaN epochs are skipped.

    Before the first block completes this is the mean of the epochs so far. A block completed at
    the end of epoch ``k`` first counts at epoch ``k + 1``.
    """
    base = np.full(x.size, np.nan)
    blocks: list[float] = []
    block_sum, block_count, partial_sum, partial_count = 0.0, 0, 0.0, 0
    for k in range(x.size):
        if np.isfinite(x[k]):
            block_sum += x[k]
            block_count += 1
            partial_sum += x[k]
            partial_count += 1
        if blocks:
            base[k] = float(np.mean(sorted(blocks[-n_blocks:])[:BASELINE_LOWEST]))
        elif partial_count:
            base[k] = partial_sum / partial_count
        if (k + 1) % BLOCK_EPOCHS == 0:
            if block_count >= BLOCK_MIN_EPOCHS:
                blocks.append(block_sum / block_count)
            block_sum, block_count = 0.0, 0
    return base


def bout_timing(
    act: np.ndarray, tsb_cap_min: int = TSB_CAP_MIN, ahead_epochs: int = DELAY
) -> tuple[np.ndarray, np.ndarray]:
    """Minutes since the last bout epoch and minutes until the next.

    The first is capped at ``tsb_cap_min``, the second at the forward reach ``ahead_epochs``.
    With no bout so far the first is the cap, so it does not count time from the start of the
    recording.
    """
    since = np.empty(act.size)
    until = np.empty(act.size)
    last = -(10**9)
    for k in range(act.size):
        if act[k] == 1:
            last = k
        since[k] = min((k - last) / EPOCHS_PER_MIN, tsb_cap_min)
        ahead = np.flatnonzero(act[k + 1 : k + 1 + ahead_epochs] == 1)
        until[k] = ahead[0] + 1 if ahead.size else ahead_epochs
    return since, until / EPOCHS_PER_MIN


def window_stat(values: np.ndarray, stat: str, direction: str, n: int) -> np.ndarray:
    """``stat`` of ``values`` over a backward (``b``) or forward (``f``) window of ``n`` epochs.

    NaN epochs are left out; a window with nothing in it is NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    n_epochs = values.size
    k = np.arange(n_epochs)
    if direction == "b":
        lo, hi = np.maximum(k - n + 1, 0), k + 1
    else:
        lo, hi = np.minimum(k + 1, n_epochs), np.minimum(k + n + 1, n_epochs)
    if stat == "max":
        pad = np.full(n, np.nan)
        if direction == "b":
            view = sliding_window_view(np.concatenate([pad[:-1], values]), n)
        else:
            view = sliding_window_view(np.concatenate([values, pad]), n)[1 : n_epochs + 1]
        return np.fmax.reduce(view, axis=1)
    present = np.isfinite(values)
    filled = np.where(present, values, 0.0)

    def total(x: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(x)])
        return c[hi] - c[lo]

    count = total(present.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total(filled) / count
        if stat == "mean":
            return np.where(count > 0, mean, np.nan)
        var = total(filled * filled) / count - mean * mean
    return np.where(count > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)


def compute_features(
    acc25: np.ndarray,
    valid25: np.ndarray,
    hr_t: np.ndarray,
    hr_bpm: np.ndarray,
    inputs: dict = INPUTS,
) -> pd.DataFrame:
    """The feature table of one recording: ``epoch`` and the columns of ``inputs``, in order.

    ``acc25`` is ``(n_epochs * 750, 3)`` in g on the 25 Hz grid, ``valid25`` marks the samples
    that are not inside a gap, and ``hr_t`` (seconds from the start of epoch 0) and ``hr_bpm``
    are the heart-rate samples, which may be empty.
    """
    sig = epoch_signals(acc25, valid25)
    present = sig.pop("present")
    n_epochs = present.size
    hr, hr_std, hr_n = epoch_heart_rate(hr_t, hr_bpm, n_epochs)
    blocks = reach(inputs, "baseline_blocks")
    base = low_baseline(hr, blocks)
    act = np.where(
        present,
        (sig["jerk"] >= BOUT_RATIO * low_baseline(sig["jerk"], blocks)).astype(float),
        np.nan,
    )
    since, until = bout_timing(
        np.nan_to_num(act, nan=0.0), reach(inputs, "tsb_cap_min"), delay(inputs)
    )
    series = {
        **sig, "act": act, "daz": np.abs(np.diff(sig["angle"], prepend=sig["angle"][:1])),
        "hr": hr, "hr_rel": hr - base,
        "hr_valid": np.where(present, (hr_n > 0).astype(float), np.nan),
    }  # fmt: skip
    at_time = {
        **series, "hr_std": hr_std, "hr_n": hr_n, "hr_missing": (hr_n == 0).astype(float),
        "hr_base": base, "tsb_min": since, "tnb_min": until,
    }  # fmt: skip
    columns = {name: at_time[name] for name in inputs["features"].get("at_time", [])}
    for w in windows(inputs):
        columns[w["name"]] = window_stat(series[w["signal"]], w["stat"], w["dir"], w["epochs"])
    table = pd.DataFrame(columns).astype(np.float32)
    table.insert(0, "epoch", np.arange(n_epochs))
    return table

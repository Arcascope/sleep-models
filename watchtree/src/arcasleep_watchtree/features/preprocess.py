"""From a raw wrist accelerometer stream to per-epoch movement and angle signals.

The Instinct delivers 25 Hz, 3-axis acceleration in g. Other cohorts are brought to that stream
by :func:`to_25hz` (offline only; the watch never resamples). Everything after that is the
arithmetic the watch runs, sample by sample:

* jerk, ``||a[n] - a[n-1]||``, on the 25 Hz stream (senpy's ``compute_jerk`` with ``use_diff``);
* two cascaded biquads per axis (4-pole Butterworth, 1.5 Hz), direct form II transposed, with the
  state initialised to the first sample, then every 5th output (5 Hz);
* low-frequency motion (mean distance of the 5 Hz samples from their 5-s block mean) and 5-s
  block angles on the 5 Hz stream;
* two more biquads and a decimation by 2 (2.5 Hz) for the breathing path.

The reference runs the same recursions as vectors in float64; the watch runs them in float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import lfilter, lfiltic, resample_poly, sosfilt, sosfilt_zi

from arcasleep_watchtree.features import constants as C


def _grid_values(t: np.ndarray, xyz: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of ``xyz`` onto ``grid``; across a gap the last value is held.

    Returns the values and a mask that is False inside a gap (more than ``GAP_S`` between the
    samples either side) and outside the recording.
    """
    prev = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, t.size - 1)
    nxt = np.minimum(prev + 1, t.size - 1)
    span = t[nxt] - t[prev]
    inside = (grid >= t[0]) & (grid <= t[-1])
    ok = inside & (span <= C.GAP_S)
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

    The stream is placed on a uniform grid at its nominal rate (linear interpolation, which also
    removes timestamp jitter), then reduced to 25 Hz by a polyphase Kaiser FIR low-pass at the
    new Nyquist frequency. A 25 Hz stream skips the second step. Returns ``(n, 3)`` values and
    the mask of samples that are not inside a gap.
    """
    n_out = n_epochs * C.SAMPLES_PER_EPOCH
    fs_in = nominal_rate(t)
    if fs_in == C.FS:
        values, _ = _grid_values(t, xyz, t0 + np.arange(n_out) / C.FS)
    else:
        n_in = n_epochs * C.EPOCH_S * fs_in
        dense, _ = _grid_values(t, xyz, t0 + np.arange(n_in) / fs_in)
        g = gcd(C.FS, fs_in)
        values = resample_poly(dense, C.FS // g, fs_in // g, axis=0, padtype="line")[:n_out]
    _, valid = _grid_values(t, xyz, t0 + np.arange(n_out) / C.FS)
    return values, valid


def biquads(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cascaded biquads along axis 0, state started at the steady state of the first sample."""
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    return sosfilt(sos, x, axis=0, zi=zi)[0]


@dataclass
class EpochSignals:
    """Per-epoch values from one night; a NaN epoch has too few valid samples."""

    valid: np.ndarray
    jerk: np.ndarray
    jerk_peak: np.ndarray
    lfm: np.ndarray
    angle: np.ndarray
    angle_step: np.ndarray
    still_fraction: np.ndarray
    breath_still: np.ndarray
    #: ``(n_epochs, 75, 3)``, the 2.5 Hz high-passed stream, for the breathing bins.
    breath_input: np.ndarray


def jerk_stream(acc: np.ndarray) -> np.ndarray:
    """``||a[n] - a[n-1]||`` per 25 Hz sample (0 for the first)."""
    step = np.diff(acc, axis=0, prepend=acc[:1])
    return np.sqrt(np.sum(step * step, axis=1))


def angle_blocks(low5: np.ndarray) -> np.ndarray:
    """z-angle in degrees of each 5-s block mean of the 5 Hz stream.

    ``atan(z / sqrt(x^2 + y^2))``, as GGIR defines it.
    """
    n_blocks = low5.shape[0] // C.ANGLE_BLOCK_SAMPLES
    mean = low5[: n_blocks * C.ANGLE_BLOCK_SAMPLES].reshape(n_blocks, C.ANGLE_BLOCK_SAMPLES, 3)
    mean = mean.mean(axis=1)
    horizontal = np.sqrt(mean[:, 0] ** 2 + mean[:, 1] ** 2)
    return np.degrees(np.arctan2(mean[:, 2], horizontal))


def epoch_signals(acc: np.ndarray, valid: np.ndarray) -> EpochSignals:
    """Per-epoch movement, angle and breathing-path signals from a 25 Hz stream."""
    n_epochs = acc.shape[0] // C.SAMPLES_PER_EPOCH
    acc = acc[: n_epochs * C.SAMPLES_PER_EPOCH]
    valid = valid[: n_epochs * C.SAMPLES_PER_EPOCH]
    epoch_valid = valid.reshape(n_epochs, -1).mean(axis=1) >= C.MIN_VALID_FRACTION

    jerk = jerk_stream(acc).reshape(n_epochs, -1)
    jerk_seconds = jerk.reshape(n_epochs, -1, C.JERK_SECOND).mean(axis=2)

    low1 = biquads(C.SOS_1, acc)
    low5 = low1[C.DECIMATE_1 - 1 :: C.DECIMATE_1][: n_epochs * C.EPOCH_S * C.FS_1]
    blocks5 = low5.reshape(-1, C.ANGLE_BLOCK_SAMPLES, 3)
    deviation = blocks5 - blocks5.mean(axis=1, keepdims=True)
    lfm = np.sqrt(np.sum(deviation * deviation, axis=2)).mean(axis=1).reshape(n_epochs, -1)

    angles = angle_blocks(low5)
    step = np.abs(np.diff(angles, prepend=angles[:1])).reshape(n_epochs, -1)
    angle = angles.reshape(n_epochs, -1)

    low2 = biquads(C.SOS_2, low5)[C.DECIMATE_2 - 1 :: C.DECIMATE_2]
    # y[n] = HP_A * (y[n-1] + x[n] - x[n-1]), started with y = 0 and x[-1] = x[0].
    hp_b, hp_a = [C.HP_A, -C.HP_A], [1.0, -C.HP_A]
    hp_zi = np.stack([lfiltic(hp_b, hp_a, [0.0], [low2[0, axis]]) for axis in range(3)], axis=1)
    high = lfilter(hp_b, hp_a, low2, axis=0, zi=hp_zi)[0]
    breath = high.reshape(n_epochs, C.BREATH_SAMPLES, 3)

    def masked(values: np.ndarray) -> np.ndarray:
        return np.where(epoch_valid, values, np.nan)

    return EpochSignals(
        valid=epoch_valid,
        jerk=masked(jerk.mean(axis=1)),
        jerk_peak=masked(jerk_seconds.max(axis=1)),
        lfm=masked(lfm.mean(axis=1)),
        angle=masked(angle.mean(axis=1)),
        angle_step=masked(step.mean(axis=1)),
        still_fraction=masked((step < C.STILL_DEG).mean(axis=1)),
        breath_still=epoch_valid & (step.max(axis=1) < C.BREATH_STILL_DEG),
        breath_input=breath,
    )

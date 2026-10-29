"""Breathing rate and regularity from Goertzel bins over one epoch of the 2.5 Hz stream.

Each axis of the high-passed 2.5 Hz stream (75 samples per epoch) goes through a Goertzel
recursion at 16 frequencies, 0.1 to 0.6 Hz in steps of 1/30 Hz. The three axes' powers add. The
peak bin, refined by a parabola through its neighbours, gives the rate; the share of the band's
power in the peak and the spectral entropy say how regular it is. Only epochs that are still
(:data:`~arcasleep_watchtree.features.constants.BREATH_STILL_DEG`) get values; the others are NaN.
"""

from __future__ import annotations

import numpy as np

from arcasleep_watchtree.features import constants as C

NAMES = ("breath_bpm", "breath_peak_share", "breath_entropy", "breath_log_power")


def goertzel_power(x: np.ndarray) -> np.ndarray:
    """Power at each breathing bin: ``x`` is ``(n_epochs, 75, 3)``; returns ``(n_epochs, 16)``.

    The recursion per sample is ``s = x + 2 cos(w) s1 - s2``; the power after the last sample is
    ``s1^2 + s2^2 - 2 cos(w) s1 s2``.
    """
    omega = 2.0 * np.pi * C.BREATH_BINS / C.BREATH_SAMPLES
    coeff = 2.0 * np.cos(omega)
    n_epochs = x.shape[0]
    s1 = np.zeros((n_epochs, 3, coeff.size))
    s2 = np.zeros_like(s1)
    for n in range(x.shape[1]):
        s0 = x[:, n, :, None] + coeff * s1 - s2
        s2, s1 = s1, s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return power.sum(axis=1)


def breathing_features(breath_input: np.ndarray, still: np.ndarray) -> dict[str, np.ndarray]:
    """The four per-epoch breathing values; NaN where the epoch is not still."""
    power = np.maximum(goertzel_power(breath_input), 1e-30)
    total = power.sum(axis=1)
    peak = power.argmax(axis=1)
    rows = np.arange(power.shape[0])
    inner = np.clip(peak, 1, power.shape[1] - 2)
    below, centre, above = power[rows, inner - 1], power[rows, peak], power[rows, inner + 1]
    curvature = below - 2.0 * centre + above
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(curvature < 0, 0.5 * (below - above) / curvature, 0.0)
    shift = np.where((peak == 0) | (peak == power.shape[1] - 1), 0.0, np.clip(shift, -0.5, 0.5))
    step_hz = C.FS_2 / C.BREATH_SAMPLES
    bpm = 60.0 * (C.BREATH_HZ[peak] + shift * step_hz)
    share = power[rows, peak] / total
    p = power / total[:, None]
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(power.shape[1])
    values = {
        "breath_bpm": bpm,
        "breath_peak_share": share,
        "breath_entropy": entropy,
        "breath_log_power": np.log10(total),
    }
    return {name: np.where(still, v, np.nan) for name, v in values.items()}

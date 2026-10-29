"""Synthetic signals as functions of time u (s since the night's t0); known content."""
import numpy as np
SIN_F = (0.7, 1.3, 2.0, 3.0, 4.2, 5.5, 7.0, 9.0, 11.5)
AX = np.array([1.0, 0.8, 0.6])


def sines(u, freqs):
    """(n, 3): one sinusoid per axis (unit amplitude)"""
    return np.stack([np.sin(2 * np.pi * f * u + 0.3 * i) for i, f in enumerate(freqs)], axis=1)


def pulse(u, bpm, burst_hz=5.5, sigma=0.06):
    """(n, 3): a burst_hz Gaussian-windowed burst once per beat plus a component at the beat rate (Hz), scaled per axis."""
    T = 60.0 / bpm; j = np.round(u / T); s = np.zeros_like(u)
    for dj in (-1, 0, 1):
        d = u - (j + dj) * T
        s += np.exp(-0.5 * (d / sigma) ** 2) * np.sin(2 * np.pi * burst_hz * d)
    s = s + 0.5 * np.sin(2 * np.pi * (bpm / 60.0) * u)
    return s[:, None] * AX[None, :]

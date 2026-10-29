"""Pulse rate from the wrist accelerometer in the heart-rate band (0.67-3.0 Hz, 40-180 bpm).

Causal: the band-pass is a forward-only Butterworth filter on the 25 Hz stream, and the estimate that ends at time e
uses only samples in [e - W, e). One window per 5 s (stride 5 s), so epoch k's estimate is the window that ends at the
epoch end 30 (k + 1) s (window index 6 k). Spectra are senpy NUFFT power spectra (Hann taper, window mean removed) of each
axis, summed over the three axes; no spectrum is averaged over windows and nothing is resampled.

  bpm     60 x the peak frequency in 0.67-3.0 Hz (parabolic interpolation over the three bins around the peak)
  quality peak-bin power / total power in 0.67-3.0 Hz (1 = all band power in one bin)
  A window is estimable when >= 80% of its samples are valid (not in an accelerometer gap); invalid samples are
  dropped from the NUFFT, not filled.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

FS = 25.0
BAND = (0.67, 3.0)
STRIDE_S = 5
WINDOWS_S = (10, 20, 30)
MIN_VALID = 0.8
EPOCH_S = 30
_SOS = butter(4, BAND, btype="bandpass", fs=FS, output="sos")


def accel_hr(acc25: np.ndarray, valid25: np.ndarray, n_epochs: int, window_s: int):
    """Window j ends at 30 + 5 j s (j = 0 .. 6 n_epochs - 6). Returns (bpm, quality) arrays of that length (NaN = no estimate)."""
    from senpy import compute_nufft_spectrogram

    n = n_epochs * EPOCH_S * int(FS)
    x = np.asarray(acc25[:n], float)
    ok = np.asarray(valid25[:n], bool)
    n_win = (n_epochs * EPOCH_S - EPOCH_S) // STRIDE_S + 1
    bpm = np.full(n_win, np.nan)
    qual = np.full(n_win, np.nan)
    if n_win <= 0 or ok.sum() < window_s * FS:
        return bpm, qual
    x = np.where(ok[:, None], x, np.nan)
    first = np.flatnonzero(ok)[0]
    x = x - x[first]
    xf = np.where(ok[:, None], x, 0.0)
    y = sosfilt(_SOS, xf, axis=0)  # causal; gaps carry zeros through the filter
    t = np.arange(n) / FS
    tv, yv = t[ok], y[ok]
    min_samples = int(MIN_VALID * window_s * FS)
    psd = None
    for a in range(3):
        r = compute_nufft_spectrogram(
            tv, yv[:, a], float(window_s), float(window_s - STRIDE_S), kind="psd",
            origin_s=float(EPOCH_S - window_s), empty_windows="keep", min_samples=min_samples,
        )
        sel_f = (r.frequencies >= BAND[0]) & (r.frequencies <= BAND[1])
        good = r.valid & (r.window_index < n_win)
        s = np.full((n_win, int(sel_f.sum())), np.nan)
        s[r.window_index[good]] = r.Sxx[good][:, sel_f]
        psd = s if psd is None else psd + s
        freqs = r.frequencies[sel_f]
    have = np.isfinite(psd).all(axis=1) & (psd.sum(axis=1) > 0)
    p = np.argmax(np.where(have[:, None], psd, -1.0), axis=1)
    rows = np.flatnonzero(have)
    pk = p[rows]
    f = freqs[pk].copy()
    inner = (pk > 0) & (pk < freqs.size - 1)
    a_, b_, c_ = psd[rows, np.clip(pk - 1, 0, None)], psd[rows, pk], psd[rows, np.clip(pk + 1, None, freqs.size - 1)]
    den = a_ - 2 * b_ + c_
    delta = np.where(inner & (den < 0), np.clip(0.5 * (a_ - c_) / np.where(den < 0, den, -1.0), -0.5, 0.5), 0.0)
    f = f + delta * (freqs[1] - freqs[0])
    bpm[rows] = 60.0 * f
    qual[rows] = b_ / psd[rows].sum(axis=1)
    return bpm, qual

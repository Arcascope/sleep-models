"""Accelerometer pulse-rate estimators, causal, one estimate per 30 s epoch from the W s window that ends at the epoch end.

Spectra: senpy NUFFT PSD (Hann taper, window mean removed) of the causally filtered (0.5-10 Hz, 4th-order Butterworth,
forward only) 25 Hz stream, per window; nothing is averaged over windows. Window k of length W covers [30 k, 30 k + W), so
epoch e (end 30 (e + 1) s) uses window k = e + 1 - W/30; the first W/30 - 1 epochs have no estimate. Windows need >= 80% valid
samples.

Spectral types: x, y, z (one axis), sum (sum of the three axis PSDs), mag (PSD of the norm of the acceleration vector).
Methods on a spectrum S over the candidate grid F = 40..180 bpm in 1 bpm steps:
  peak   argmax S(f)
  hsum   argmax S(f) + S(2f) + S(3f)
  hps    argmax S(f) S(2f) S(3f)   (log sum)
  sacf   spectral autocorrelation: autocorrelation over frequency (lag) of S on 0.5-9 Hz; argmax over lags in F
  acfX   (time domain) summed per-axis autocorrelation of the band-limited signal, lag 0.33-1.5 s (X = band, 3 or 8 Hz)
whiten: S is divided by its running median over +-1.5 Hz first (flattens the 1/f slope so harmonics are comparable).
Quality of a window = score at the chosen frequency / mean score over the grid (acf: the autocorrelation value).
vit_*: forward-only Viterbi (filtering, no backtracking) over the 1 bpm grid on the hsum score of the whitened sum spectrum.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.ndimage import median_filter

FS = 25.0
EPOCH_S = 30
MIN_VALID = 0.8
GRID_BPM = np.arange(40, 181, 1.0)
GRID_HZ = GRID_BPM / 60.0
_SOS_W = butter(4, (0.5, 10.0), btype="bandpass", fs=FS, output="sos")
_SOS_3 = butter(4, (0.67, 3.0), btype="bandpass", fs=FS, output="sos")
_SOS_8 = butter(4, (0.67, 8.0), btype="bandpass", fs=FS, output="sos")
TYPES = ("x", "y", "z", "sum", "mag")


def _filt(sos, acc, ok):
    x = np.where(ok[:, None], acc, 0.0)
    f = np.flatnonzero(ok)[0]
    x = np.where(ok[:, None], acc - acc[f], 0.0)
    return sosfilt(sos, x, axis=0)


def spectra(acc25, valid25, n_epochs, W):
    """dict type -> (n_win, n_freq) PSD on freqs (0.4..12 Hz), NaN rows where the window is not estimable; freqs."""
    from senpy import compute_nufft_spectrogram
    n = n_epochs * EPOCH_S * int(FS)
    acc, ok = np.asarray(acc25[:n], float), np.asarray(valid25[:n], bool)
    n_win = n_epochs - W // EPOCH_S + 1
    if n_win <= 0 or ok.sum() < W * FS:
        return None, None
    y = _filt(_SOS_W, acc, ok)
    mag = np.linalg.norm(acc, axis=1)
    f0 = np.flatnonzero(ok)[0]
    ym = sosfilt(_SOS_W, np.where(ok, mag - mag[f0], 0.0))
    t = np.arange(n) / FS
    tv = t[ok]
    sigs = {"x": y[ok, 0], "y": y[ok, 1], "z": y[ok, 2], "mag": ym[ok]}
    out, freqs = {}, None
    for k, s in sigs.items():
        r = compute_nufft_spectrogram(tv, s, float(W), float(W - EPOCH_S), kind="psd", origin_s=0.0,
                                      empty_windows="keep", min_samples=int(MIN_VALID * W * FS))
        sel = (r.frequencies >= 0.4) & (r.frequencies <= 12.0)
        good = r.valid & (r.window_index < n_win)
        S = np.full((n_win, int(sel.sum())), np.nan)
        S[r.window_index[good]] = r.Sxx[good][:, sel]
        out[k] = S
        freqs = r.frequencies[sel]
    out["sum"] = out["x"] + out["y"] + out["z"]
    return out, freqs


def whiten(S, freqs):
    df = freqs[1] - freqs[0]
    k = max(3, int(round(1.5 / df)) * 2 + 1)
    ok = np.isfinite(S).all(axis=1)
    out = np.full_like(S, np.nan)
    if ok.any():
        med = median_filter(S[ok], size=(1, k), mode="nearest")
        out[ok] = S[ok] / np.maximum(med, 1e-30)
    return out


def _interp(S, freqs, f):
    """S (n, nf) at frequencies f (any shape) per row -> (n, *f.shape); out-of-range -> NaN."""
    pos = (f - freqs[0]) / (freqs[1] - freqs[0])
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    valid = (i0 >= 0) & (i0 < freqs.size - 1)
    i0c = np.clip(i0, 0, freqs.size - 2)
    v = S[:, i0c] * (1 - fr) + S[:, i0c + 1] * fr
    return np.where(valid, v, np.nan)


def scores(S, freqs, method):
    """(n_win, n_grid) score of every candidate frequency; NaN rows stay NaN."""
    if method == "peak":
        return _interp(S, freqs, GRID_HZ)
    if method in ("hsum", "hps"):
        h = [_interp(S, freqs, GRID_HZ * m) for m in (1, 2, 3)]
        if method == "hsum":
            return h[0] + h[1] + h[2]
        return np.log(np.maximum(h[0], 1e-30)) + np.log(np.maximum(h[1], 1e-30)) + np.log(np.maximum(h[2], 1e-30))
    if method == "sacf":
        sel = (freqs >= 0.5) & (freqs <= 9.0)
        X = S[:, sel]
        X = X - np.nanmean(X, axis=1, keepdims=True)
        df = freqs[1] - freqs[0]
        lags = np.arange(1, int(round(3.2 / df)) + 1)
        R = np.full((S.shape[0], lags.size), np.nan)
        good = np.isfinite(X).all(axis=1)
        Xg = X[good]
        for j, L in enumerate(lags):
            R[good, j] = (Xg[:, :-L] * Xg[:, L:]).mean(axis=1)
        R[good] = R[good] / np.maximum((Xg**2).mean(axis=1, keepdims=True), 1e-30)
        out = np.full((S.shape[0], GRID_HZ.size), np.nan)
        pos = GRID_HZ / df - 1  # lag index (lag L at index L-1)
        i0 = np.clip(np.floor(pos).astype(int), 0, lags.size - 2)
        fr = pos - i0
        out[good] = R[good][:, i0] * (1 - fr) + R[good][:, i0 + 1] * fr
        return out
    raise ValueError(method)


def pick(score, lin=True):
    """Peak bin with parabolic refinement (bpm) and quality = peak / mean over grid (linear scores) or peak - mean (log)."""
    have = np.isfinite(score).all(axis=1)
    bpm = np.full(score.shape[0], np.nan); q = np.full(score.shape[0], np.nan)
    if not have.any():
        return bpm, q
    s = score[have]
    p = np.argmax(s, axis=1)
    r = np.arange(s.shape[0])
    a, b, c = s[r, np.clip(p - 1, 0, None)], s[r, p], s[r, np.clip(p + 1, None, s.shape[1] - 1)]
    den = a - 2 * b + c
    inner = (p > 0) & (p < s.shape[1] - 1) & (den < 0)
    d = np.where(inner, np.clip(0.5 * (a - c) / np.where(den < 0, den, -1.0), -0.5, 0.5), 0.0)
    bpm[have] = GRID_BPM[p] + d * (GRID_BPM[1] - GRID_BPM[0])
    m = s.mean(axis=1)
    q[have] = (b / np.maximum(m, 1e-30)) if lin else (b - m)
    return bpm, q


def viterbi_filter(emis, lam, raw_quality=False):
    """Forward-only tracking: delta_t(f) = emis_t(f) + max_f' [delta_{t-1}(f') - lam |f - f'|] (bins of 1 bpm).
    The estimate at t is argmax delta_t, which uses windows up to t only. NaN rows (no window) carry delta forward, estimate NaN.
    Returns bpm (n,), quality = emission at the chosen bin / mean emission (raw_quality: the emission at the chosen bin)."""
    n, nb = emis.shape
    est = np.full(n, np.nan); q = np.full(n, np.nan)
    delta = None
    idx = np.arange(nb)
    for t in range(n):
        e = emis[t]
        if not np.isfinite(e).all():
            continue
        if delta is None:
            delta = e.copy()
        else:
            fw = np.maximum.accumulate(delta + lam * idx) - lam * idx  # max over f' <= f of delta(f') - lam (f - f')
            bw = np.maximum.accumulate((delta - lam * idx)[::-1])[::-1] + lam * idx
            delta = np.maximum(fw, bw) + e
            delta -= delta.max()
        p = int(np.argmax(delta))
        est[t] = GRID_BPM[p]
        q[t] = e[p] if raw_quality else e[p] / max(e.mean(), 1e-30)
    return est, q


def acf_est(acc25, valid25, n_epochs, W, sos):
    """Time-domain autocorrelation pulse estimate on the filtered stream (sum over axes); lags 0.33-1.5 s (40-180 bpm)."""
    n = n_epochs * EPOCH_S * int(FS)
    acc, ok = np.asarray(acc25[:n], float), np.asarray(valid25[:n], bool)
    n_win = n_epochs - W // EPOCH_S + 1
    bpm = np.full(max(n_win, 0), np.nan); q = np.full(max(n_win, 0), np.nan)
    if n_win <= 0 or ok.sum() < W * FS:
        return bpm, q
    y = _filt(sos, acc, ok)
    wl = W * int(FS)
    lags = np.arange(int(np.floor(FS * 60 / 180)) - 1, int(np.ceil(FS * 60 / 40)) + 2)
    for k in range(n_win):
        a, b = k * 750, k * 750 + wl
        if ok[a:b].mean() < MIN_VALID:
            continue
        seg = y[a:b]
        seg = seg - seg.mean(axis=0)
        v = (seg**2).sum()
        if v <= 0:
            continue
        r = np.array([(seg[:-L] * seg[L:]).sum() / (wl - L) for L in lags]) / (v / wl)
        lag_s = lags / FS
        pk = np.flatnonzero((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:])) + 1
        pk = pk[(lag_s[pk] >= 60 / 180) & (lag_s[pk] <= 60 / 40)]
        if pk.size == 0:
            continue
        j = pk[np.argmax(r[pk])]
        a_, b_, c_ = r[j - 1], r[j], r[j + 1]
        den = a_ - 2 * b_ + c_
        d = np.clip(0.5 * (a_ - c_) / den, -0.5, 0.5) if den < 0 else 0.0
        bpm[k] = 60.0 * FS / (lags[j] + d)
        q[k] = b_
    return bpm, q


def acf_curves(acc25, valid25, n_epochs, W, sos):
    """(n_win, n_grid) normalised autocorrelation of the filtered stream (sum over axes) at the lag of every grid bpm; NaN = no window."""
    n = n_epochs * EPOCH_S * int(FS)
    acc, ok = np.asarray(acc25[:n], float), np.asarray(valid25[:n], bool)
    n_win = n_epochs - W // EPOCH_S + 1
    out = np.full((max(n_win, 0), GRID_BPM.size), np.nan)
    if n_win <= 0 or ok.sum() < W * FS:
        return out
    y = _filt(sos, acc, ok)
    wl = W * int(FS)
    lag_f = FS * 60.0 / GRID_BPM
    l0 = np.floor(lag_f).astype(int); fr = lag_f - l0
    for k in range(n_win):
        a, b = k * 750, k * 750 + wl
        if ok[a:b].mean() < MIN_VALID:
            continue
        seg = y[a:b]; seg = seg - seg.mean(axis=0)
        v = (seg**2).sum()
        if v <= 0:
            continue
        r = np.zeros(l0.max() + 2)
        for L in range(1, r.size):
            r[L] = (seg[:-L] * seg[L:]).sum() / (wl - L)
        r /= v / wl
        out[k] = r[l0] * (1 - fr) + r[l0 + 1] * fr
    return out

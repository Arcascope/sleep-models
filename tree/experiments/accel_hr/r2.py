"""Accelerometer heart rate from the recorded accelerometer samples, without resampling (estimator R2, band 0.67-8 Hz).

One estimate per 30 s epoch window k = [t0 + 30 k, t0 + 30 (k + 1)):
  1. senpy NUFFT coefficients of each axis on the recorded timestamps (bins j / 30 Hz, Hann taper, window mean removed);
  2. in-band coefficients, 0.67 Hz to min(8 Hz, 0.9 x the window's Nyquist frequency from its median recorded step);
  3. autocorrelation of the band-limited window (axes pooled) at the exact lag tau = 60 / bpm for every bpm in GRID_BPM,
     the band-limited signal evaluated at 25 Hz points and at the same points shifted by tau (no interpolation of samples);
  4. forward-only tracking: score(k, f) = 10 r(k, f) + max over f' of [score(k - 1, f') - |f - f'|]; the estimate is argmax at k.
A window needs >= 80 % of its 30 s covered by recorded steps no longer than GAP_S; other windows give no estimate.
"""
import numpy as np
from senpy import compute_nustft

EPOCH_S = 30
GAP_S = 1.0
MIN_COVERAGE = 0.8
GRID_BPM = np.arange(40, 181, 1.0)
TAU = 60.0 / GRID_BPM
FMAX = 10.0
EDGE = 8.0
LOW = 0.67
WL = 750
U = np.arange(WL) / 25.0
N_TAU = np.array([int(np.sum(U + tau < 30.0)) for tau in TAU])
LAMBDA = 1.0
EMISSION_SCALE = 10.0


def coverage(t, t0, n_epochs):
    """Fraction of each epoch window covered by recorded steps no longer than GAP_S."""
    edges = np.searchsorted(t, t0 + EPOCH_S * np.arange(n_epochs + 1))
    out = np.zeros(n_epochs)
    for k in range(n_epochs):
        s = np.diff(t[edges[k]:edges[k + 1]])
        out[k] = s[s <= GAP_S].sum() / EPOCH_S
    return out


def window_nyquist(t, t0, n_epochs):
    edges = np.searchsorted(t, t0 + EPOCH_S * np.arange(n_epochs + 1))
    nyq = np.full(n_epochs, np.nan)
    for k in range(n_epochs):
        if edges[k + 1] - edges[k] > 2:
            s = np.diff(t[edges[k]:edges[k + 1]])
            s = s[s > 0]
            if s.size:
                nyq[k] = 0.5 / np.median(s)
    return nyq


def coefficients(t, x, t0, n_epochs):
    """Complex NUFFT coefficients (n_epochs, bins, 3) on bins j / 30 Hz up to FMAX; NaN where the window is empty."""
    out = np.full((n_epochs, int(FMAX * 30) + 1, 3), np.nan + 0j)
    for ax in range(3):
        r = compute_nustft(t, x[:, ax], float(EPOCH_S), 0.0, origin_s=t0, target_fs=2 * FMAX, empty_windows="keep", min_samples=100)
        good = r.valid & (r.window_index < n_epochs)
        out[r.window_index[good], :, ax] = r.coefficients[good][:, :out.shape[1]]
    return out, r.frequencies[:out.shape[1]]


def acf_exact(Cb, fb):
    """Autocorrelation of the band-limited window s(u) = Re sum_f Cb[f] e^{-i 2 pi f u} (Cb (nb, 3)) at the lags TAU, axes pooled.

    r(tau) = [sum_{n < n_tau} (s(u_n) - m)(s(u_n + tau) - m) / n_tau] / [sum_n (s(u_n) - m)^2 / WL], u_n on the 25 Hz grid with u_n + tau < 30 s,
    m the mean of s over all u_n. The shifted window is s(u + tau) = Re sum_f Cb[f] e^{-i 2 pi f tau} e^{-i 2 pi f u}; the sums over n are
    evaluated through the cumulative sums G[f, n] = sum_{k < n} (s(u_k) - m) e^{-i 2 pi f u_k}.
    """
    E = np.exp(-2j * np.pi * fb[:, None] * U[None, :])
    s0 = np.real(Cb.T @ E).T
    m = s0.mean(axis=0)
    a = s0 - m
    v = (a ** 2).sum()
    if v <= 0:
        return np.full(GRID_BPM.size, np.nan)
    G = np.concatenate([np.zeros((fb.size, 1, 3), complex), np.cumsum(E[:, :, None] * a[None, :, :], axis=1)], axis=1)
    cum_a = np.concatenate([np.zeros((1, 3)), np.cumsum(a, axis=0)])
    ph = np.exp(-2j * np.pi * fb[:, None] * TAU[None, :])
    cross = np.real(np.einsum("fa,ft,fta->t", Cb, ph, G[:, N_TAU, :]))
    cross -= (m[None, :] * cum_a[N_TAU]).sum(axis=1)
    return cross / N_TAU / (v / WL)


def curves(t, x, t0, n_epochs):
    """(n_epochs, len(GRID_BPM)) autocorrelation curves, NaN rows for windows without an estimate."""
    valid = coverage(t, t0, n_epochs) >= MIN_COVERAGE
    nyq = window_nyquist(t, t0, n_epochs)
    C, freqs = coefficients(t, x, t0, n_epochs)
    cap = np.where(valid, np.minimum(EDGE, 0.9 * nyq), np.nan)
    out = np.full((n_epochs, GRID_BPM.size), np.nan)
    for k in range(n_epochs):
        if np.isfinite(C[k]).all() and np.isfinite(cap[k]):
            b = (freqs >= LOW) & (freqs <= cap[k])
            out[k] = acf_exact(C[k, b], freqs[b])
    return out


def track(emis, lam=LAMBDA):
    """Forward-only tracking over the 1 bpm grid; NaN rows carry the state forward and give a NaN estimate."""
    n, nb = emis.shape
    est = np.full(n, np.nan)
    delta = None
    idx = np.arange(nb)
    for k in range(n):
        e = emis[k]
        if not np.isfinite(e).all():
            continue
        if delta is None:
            delta = e.copy()
        else:
            fw = np.maximum.accumulate(delta + lam * idx) - lam * idx
            bw = np.maximum.accumulate((delta - lam * idx)[::-1])[::-1] + lam * idx
            delta = np.maximum(fw, bw) + e
            delta -= delta.max()
        est[k] = GRID_BPM[int(np.argmax(delta))]
    return est


def accel_heart_rate(t, xyz, t0, n_epochs):
    """Tracked heart rate (bpm) per epoch window from raw samples ``t`` (s) and ``xyz`` (g); NaN where there is no estimate."""
    return track(EMISSION_SCALE * curves(np.asarray(t, float), np.asarray(xyz, float), float(t0), int(n_epochs)))

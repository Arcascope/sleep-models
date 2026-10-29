"""Autocorrelation curves that do not resample the raw samples to a uniform rate.
Windows: 30 s, window k = epoch k, [t0 + 30 k, t0 + 30 (k + 1)); a window needs >= 80 % valid 25 Hz samples (valid25, as for estimator A).
Spectral bins are j / 30 Hz. Band: 0.67 Hz to min(edge, 0.9 x the window's Nyquist frequency), Nyquist from the window's median step.
  R1   r(tau) = sum_f P(f) cos(2 pi f tau) / sum_f P(f), P = sum over axes of |senpy NUFFT coefficient|^2 on the raw timestamps
       (Hann taper, window mean removed), tau = 60 / bpm
  R2   the NUFFT coefficients inside the band evaluated (no interpolation) at the 25 Hz points of the window and at the same points shifted by tau;
       the autocorrelation of ahr2.acf_curves with the lag tau = 60 / bpm taken exactly instead of interpolated between integer-sample lags
  R2L  least-squares harmonic fit (cos and sin at j / 30 Hz, 0 Hz to the 8 Hz cap, ridge 1e-6) to the raw samples of the window, the in-band
       part evaluated on the 25 Hz grid, then the same autocorrelation
  R1c  R1 computed from the valid samples of the to_25hz stream (uniform timestamps): separates the NUFFT / Hann form from the irregular timestamps
Edges: 8 (0.67-8 Hz) and 3 (0.67-3 Hz)."""
import numpy as np
import ahr2 as A
from senpy import compute_nustft

EDGES = (8.0, 3.0)
FMAX = 10.0
WL = 750
U = np.arange(WL) / 25.0
LAG_F = 25.0 * 60.0 / A.GRID_BPM
L0 = np.floor(LAG_F).astype(int); LFR = LAG_F - L0
TAU = 60.0 / A.GRID_BPM


def acf_seg(seg):
    """ahr2.acf_curves' autocorrelation of one band-limited (750, 3) window; NaN if it has no power."""
    seg = seg - seg.mean(axis=0)
    v = (seg**2).sum()
    if v <= 0:
        return np.full(A.GRID_BPM.size, np.nan)
    r = np.zeros(L0.max() + 2)
    for L in range(1, r.size):
        r[L] = (seg[:-L] * seg[L:]).sum() / (WL - L)
    r /= v / WL
    return r[L0] * (1 - LFR) + r[L0 + 1] * LFR


def coverage(t, t0, ne):
    """fraction of each 30 s window covered by recorded steps no longer than GAP_S (recorded timestamps only)."""
    from arcasleep_watchtree import constants as K
    edges = np.searchsorted(t, t0 + 30.0 * np.arange(ne + 1)); out = np.zeros(ne)
    for k in range(ne):
        s = np.diff(t[edges[k]:edges[k + 1]]); out[k] = s[s <= K.GAP_S].sum() / 30.0
    return out


def acf_exact(Cb, fb):
    """Autocorrelation of the band-limited window s(u) = Re sum_f Cb[f] e^{-i 2 pi f u} (Cb (nb, 3)), at lag tau = 60 / bpm seconds exactly (the fit is
    evaluated at u + tau; no interpolation): r(tau) = [sum_n s(u_n) s(u_n + tau) / n(tau)] / [sum_n s(u_n)^2 / 750] over the 25 Hz evaluation points u_n
    with u_n + tau < 30 s, axes pooled, mean over the points removed from both factors."""
    E = np.exp(-2j * np.pi * fb[:, None] * U[None, :])
    s0 = np.real(Cb.T @ E).T
    m = s0.mean(axis=0); v = ((s0 - m) ** 2).sum()
    if v <= 0:
        return np.full(A.GRID_BPM.size, np.nan)
    out = np.empty(A.GRID_BPM.size)
    for i, tau in enumerate(TAU):
        n = int(np.sum(U + tau < 30.0))
        ph = np.exp(-2j * np.pi * fb * tau)[:, None]
        st = np.real((Cb * ph).T @ E[:, :n]).T
        out[i] = ((s0[:n] - m) * (st - m)).sum() / n / (v / WL)
    return out


def window_nyquist(t, t0, ne):
    edges = np.searchsorted(t, t0 + 30.0 * np.arange(ne + 1))
    nyq = np.full(ne, np.nan); cnt = np.diff(edges)
    for k in range(ne):
        if cnt[k] > 2:
            s = np.diff(t[edges[k]:edges[k + 1]]); s = s[s > 0]
            if s.size:
                nyq[k] = 0.5 / np.median(s)
    return nyq, edges


def coeffs(t, x, t0, ne):
    """complex NUFFT coefficients (ne, nf, 3) on bins j / 30 Hz up to FMAX (convention: sum w x e^{+i 2 pi f t}), NaN where the window is empty."""
    out = np.full((ne, int(FMAX * 30) + 1, 3), np.nan + 0j)
    for ax in range(3):
        r = compute_nustft(t, x[:, ax], 30.0, 0.0, origin_s=t0, target_fs=2 * FMAX, empty_windows="keep", min_samples=100)
        good = r.valid & (r.window_index < ne)
        out[r.window_index[good], :, ax] = r.coefficients[good][:, :out.shape[1]]
    return out, r.frequencies[:out.shape[1]]


def r1_curves(C, freqs, fhi_edge):
    """C (ne, nf, 3), fhi_edge (ne,) upper edge per window -> (ne, 141)."""
    P = (np.abs(C) ** 2).sum(axis=2)
    out = np.full((C.shape[0], A.GRID_BPM.size), np.nan)
    cosm = np.cos(2 * np.pi * freqs[:, None] * TAU[None, :])
    for k in range(C.shape[0]):
        if not np.isfinite(P[k]).all() or not np.isfinite(fhi_edge[k]):
            continue
        b = (freqs >= 0.67) & (freqs <= fhi_edge[k])
        s = P[k, b].sum()
        if s > 0:
            out[k] = (P[k, b] @ cosm[b]) / s
    return out


def recon_nufft(C, freqs, fhi_edge):
    """list of (ne) band-limited windows (750, 3) evaluated on the 25 Hz grid from NUFFT coefficients, or None."""
    E = np.exp(-2j * np.pi * freqs[:, None] * U[None, :])
    segs = []
    for k in range(C.shape[0]):
        if not np.isfinite(C[k]).all() or not np.isfinite(fhi_edge[k]):
            segs.append(None); continue
        b = (freqs >= 0.67) & (freqs <= fhi_edge[k])
        segs.append(np.real(np.einsum("fa,ft->ta", C[k, b], E[b])))
    return segs


def ls_fit(tw, xw, fmax):
    """ridge least-squares harmonic fit on bins j / 30 Hz, j = 0..fmax*30; returns coefficient (nb, 3) over [cos_0..cos_J, sin_1..sin_J] and the freqs."""
    J = int(np.floor(fmax * 30)); f = np.arange(J + 1) / 30.0
    arg = 2 * np.pi * tw[:, None] * f[None, :]
    D = np.concatenate([np.cos(arg), np.sin(arg[:, 1:])], axis=1)
    G = D.T @ D; G[np.diag_indices_from(G)] += 1e-6 * np.trace(G) / G.shape[0]
    return np.linalg.solve(G, D.T @ xw), f


def ls_recon(coef, f, lo, hi):
    """in-band part (lo <= f <= hi) of the fit evaluated on the 25 Hz grid -> (750, 3)"""
    J = f.size - 1
    sel = (f >= lo) & (f <= hi)
    arg = 2 * np.pi * U[:, None] * f[None, sel]
    return np.cos(arg) @ coef[:J + 1][sel] + np.sin(arg) @ coef[J + 1:][sel[1:]]


def curves_all(t, x, t0, ne, valid_ep, with_ls=True, with_control=None):
    """t (n,) raw timestamps, x (n, 3) signal on them, valid_ep (ne,) bool. Returns {arm: (ne, 141)}, window Nyquist (ne,).
    with_control = (t25, x25) valid-sample stream for R1c."""
    nyq, edges = window_nyquist(t, t0, ne)
    C, freqs = coeffs(t, x, t0, ne)
    out = {}; hi8 = 0.9 * nyq
    caps = {8: np.minimum(8.0, hi8), 3: np.minimum(3.0, hi8)}
    for e in (8, 3):
        cap = np.where(valid_ep, caps[e], np.nan)
        out[f"R1_{e}"] = r1_curves(C, freqs, cap)
        out[f"R2_{e}"] = np.full((ne, A.GRID_BPM.size), np.nan)
        for k in range(ne):
            if np.isfinite(C[k]).all() and np.isfinite(cap[k]):
                b = (freqs >= 0.67) & (freqs <= cap[k]); out[f"R2_{e}"][k] = acf_exact(C[k, b], freqs[b])
    if with_ls:
        for e in (8, 3):
            out[f"R2L_{e}"] = np.full((ne, A.GRID_BPM.size), np.nan)
        for k in np.flatnonzero(valid_ep & np.isfinite(nyq)):
            a, b = edges[k], edges[k + 1]
            if b - a < 100:
                continue
            fm = min(8.0, 0.9 * nyq[k])
            coef, f = ls_fit(t[a:b] - (t0 + 30.0 * k), x[a:b], fm)
            for e in (8, 3):
                sel = (f >= 0.67) & (f <= min(float(e), fm)); J = f.size - 1
                Cb = coef[:J + 1][sel] + 1j * np.concatenate([np.zeros((1, 3)), coef[J + 1:]])[sel]
                out[f"R2L_{e}"][k] = acf_exact(Cb, f[sel])
    if with_control is not None:
        t25, x25 = with_control
        C25, fr = coeffs(t25, x25, t0, ne)
        for e in (8, 3):
            out[f"R1c_{e}"] = r1_curves(C25, fr, np.where(valid_ep, float(e), np.nan))
    return out, nyq

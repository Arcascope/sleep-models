"""Autocorrelation curves for the input comparison (30 s windows ending at the epoch end, window k = epoch k).
  A  0.67-8 Hz band-pass on the 25 Hz stream (ahr2.acf_curves, the current estimator)
  B  0.67-3 Hz band-pass on the 25 Hz stream
  C  causal 4th-order 3 Hz low-pass on the 25 Hz stream (no high-pass)
  D  per-axis derivative of the raw recorded samples, (a[i+1] - a[i]) / (t[i+1] - t[i]), lags in native samples
  E  0.67-8 Hz band-pass designed at the nominal native rate, applied to the raw samples, lags in native samples
Native-rate windows hold the raw samples with timestamps in [t0 + 30 k, t0 + 30 (k + 1)); a window needs >= 80 % of fs_nominal * 30 valid samples;
products across a gap (step > GAP_S) are left out; the window mean is removed per axis; normalised as ahr2.acf_curves."""
import numpy as np
from scipy.signal import butter, sosfilt
import ahr2 as A
from arcasleep_watchtree import constants as C
from arcasleep_watchtree.preprocess import nominal_rate

SOS_C = butter(4, 3.0, btype="lowpass", fs=A.FS, output="sos")
ARMS = ("A", "B", "C", "D", "E")
LAMS = (0.5, 1.0, 2.0)


def lam_tag(l): return str(l).replace(".", "p")


def curves_25(acc25, valid25, ne):
    return {"A": A.acf_curves(acc25, valid25, ne, 30, A._SOS_8), "B": A.acf_curves(acc25, valid25, ne, 30, A._SOS_3),
            "C": A.acf_curves(acc25, valid25, ne, 30, SOS_C)}


def segments(t):
    """segment id per sample: increments where the step from the previous sample exceeds GAP_S"""
    return np.concatenate([[0], np.cumsum(np.diff(t) > C.GAP_S)])


def _native_curves(t, y, valid, sid, t0, ne, fs):
    """y (n, 3) signal, valid (n,) bool, sid (n,) segment ids -> (ne, 141) curves"""
    out = np.full((ne, A.GRID_BPM.size), np.nan)
    lag_f = fs * 60.0 / A.GRID_BPM
    l0 = np.floor(lag_f).astype(int); fr = lag_f - l0
    Lmax = l0.max() + 1
    edges = np.searchsorted(t, t0 + 30.0 * np.arange(ne + 1))
    need = C.MIN_VALID_FRACTION * 30 * fs
    for k in range(ne):
        a, b = edges[k], edges[k + 1]
        v = valid[a:b]
        if v.sum() < need:
            continue
        seg = np.where(v[:, None], y[a:b], 0.0)
        seg = np.where(v[:, None], seg - seg[v].mean(axis=0), 0.0)
        e = (seg**2).sum()
        if e <= 0:
            continue
        s = sid[a:b]; n = seg.shape[0]
        r = np.zeros(Lmax + 1)
        for L in range(1, Lmax + 1):
            if L >= n:
                break
            pw = v[:-L] & v[L:] & (s[:-L] == s[L:])
            cnt = pw.sum()
            if cnt:
                r[L] = (seg[:-L] * seg[L:]).sum() / cnt
        r /= e / v.sum()
        out[k] = r[l0] * (1 - fr) + r[l0 + 1] * fr
    return out


def curves_native(rows, t0, ne):
    """rows (n, 4): t, x, y, z (g). Returns {"D": ..., "E": ...}, nominal rate."""
    t = rows[:, 0]; xyz = rows[:, 1:4]; fs = nominal_rate(t)
    # D: per-axis derivative, assigned to the later sample of each pair; invalid across a gap
    dt = np.diff(t)
    dv = dt <= C.GAP_S
    safe = np.where(dt > 0, dt, np.nan)
    d = np.diff(xyz, axis=0) / safe[:, None]
    dv = dv & np.isfinite(d).all(axis=1)
    d = np.where(dv[:, None], d, 0.0)
    D = _native_curves(t[1:], d, dv, np.cumsum(~dv), t0, ne, fs)
    # E: band-pass at the nominal rate on the raw samples, filtered per gap-free segment (zero state at the segment's first sample)
    hi = min(8.0, 0.45 * fs)   # nominal rates below 17.8 Hz cannot hold an 8 Hz edge; the upper edge is then 0.9 of Nyquist
    sos = butter(4, (0.67, hi), btype="bandpass", fs=fs, output="sos")
    sid = segments(t); y = np.zeros_like(xyz)
    cuts = np.flatnonzero(np.diff(sid)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, t.size]):
        y[a:b] = sosfilt(sos, xyz[a:b] - xyz[a], axis=0)
    E = _native_curves(t, y, np.ones(t.size, bool), sid, t0, ne, fs)
    return {"D": D, "E": E}, fs

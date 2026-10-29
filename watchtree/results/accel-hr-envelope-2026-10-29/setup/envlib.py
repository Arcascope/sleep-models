"""Envelope-based autocorrelation curves, one per 30 s window (window k = epoch k, ends at the epoch end), 25 Hz stream.
  hil_<band>        causal Butterworth band-pass per axis (4th order, forward only), envelope |analytic signal| per axis over the window,
                    summed over axes
  lp_<band>_<fc>    same band-pass, squared per axis, causal 2nd-order Butterworth low-pass (cutoff fc Hz) over the whole night, summed over axes
  cur               the current estimator: 0.67-8 Hz band-pass, summed per-axis autocorrelation (ahr2.acf_curves)
Curve = autocorrelation (window mean removed, divided by the window's energy) at the lag of every candidate rate 40-180 bpm."""
import numpy as np
from scipy.signal import butter, sosfilt, hilbert
import ahr2 as A

FS, WL = A.FS, 750
BANDS = {"b38": (3.0, 8.0), "b48": (4.0, 8.0)}
CUTS = (2.0, 3.0, 4.0)
LAMS = (0.5, 1.0, 2.0)
SOS_BP = {k: butter(4, v, btype="bandpass", fs=FS, output="sos") for k, v in BANDS.items()}
SOS_HP = butter(2, 0.5, btype="highpass", fs=FS, output="sos")   # extra variant only: removes the envelope's slow component
SOS_LP = {c: butter(2, c, btype="lowpass", fs=FS, output="sos") for c in CUTS}
_lag_f = FS * 60.0 / A.GRID_BPM
L0 = np.floor(_lag_f).astype(int); FR = _lag_f - L0


def lp_name(b, c, hp=False): return f"lp{'hp' if hp else ''}_{b}_f{int(c)}"
def hil_name(b, hp=False): return f"hil{'hp' if hp else ''}_{b}"
def lam_tag(l): return str(l).replace(".", "p")
def cand_names(hp=False):
    """all candidate (curve, lam) names"""
    cs = [hil_name(b, hp) for b in BANDS] + [lp_name(b, c, hp) for b in BANDS for c in CUTS]
    return [(c, l) for c in cs for l in LAMS]


def curves_1d(s, okfrac):
    """s: (n_win, 750) per-window 1-D signal -> (n_win, 141) normalised autocorrelation at the candidate lags; NaN rows when < 80 % valid or no energy."""
    n_win = s.shape[0]
    out = np.full((n_win, A.GRID_BPM.size), np.nan)
    x = s - s.mean(axis=1, keepdims=True)
    v = (x**2).sum(axis=1)
    good = (okfrac >= A.MIN_VALID) & (v > 0)
    r = np.zeros((n_win, L0.max() + 2))
    for L in range(1, r.shape[1]):
        r[:, L] = (x[:, :-L] * x[:, L:]).sum(axis=1) / (WL - L)
    r /= np.maximum(v / WL, 1e-300)[:, None]
    out[good] = (r[:, L0] * (1 - FR) + r[:, L0 + 1] * FR)[good]
    return out


def all_curves(acc25, valid25, ne, hp=False):
    """hp=False: the specified estimators (+ cur). hp=True: the extra variant, the envelope high-passed at 0.5 Hz (2nd order, causal) before the autocorrelation."""
    n = ne * 750
    acc, ok = np.asarray(acc25[:n], float), np.asarray(valid25[:n], bool)
    okfrac = ok.reshape(ne, WL).mean(axis=1)
    out = {} if hp else {"cur": A.acf_curves(acc25, valid25, ne, 30, A._SOS_8)}
    for b, sos in SOS_BP.items():
        y = A._filt(sos, acc, ok)                      # (n, 3)
        env = np.abs(hilbert(y.reshape(ne, WL, 3), axis=1)).sum(axis=2)
        if hp:
            env = sosfilt(SOS_HP, env - env.mean(axis=1, keepdims=True), axis=1)
        out[hil_name(b, hp)] = curves_1d(env, okfrac)
        sq = y**2
        for c, sl in SOS_LP.items():
            e = sosfilt(sl, sq, axis=0).sum(axis=1)
            if hp:
                e = sosfilt(SOS_HP, e)
            out[lp_name(b, c, hp)] = curves_1d(e.reshape(ne, WL), okfrac)
    return out

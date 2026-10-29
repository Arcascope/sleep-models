"""BPM variability feature group (Experiment 4): causal per-epoch features from the heart-rate stream.

For the epoch k that ends at e = 30 (k + 1) s and each window length W in {2, 5, 10} min, the window is the
half-open interval [e - W, e): the same samples the 30-s epoch assignment (t // 30 == k) would give, nothing at or
after e. Spectra come from senpy's NUFFT spectrogram on the raw (non-uniform) samples, one PSD per window;
no spectra are averaged over windows and nothing is resampled.

Definitions (see COLUMNS.md for the column list):
  usable window   >= 80% coverage (n * dt / W) and no gap > max(15 s, 3 dt), gaps counted to both window edges
  bpm_lowlogp_W   ln(P(0.01-0.04 Hz) + 1e-3); NaN unless the window is usable, W >= 5 min (3 cycles of 0.01 Hz)
                  and 0.04 Hz <= Nyquist
  bpm_midlogp_W   ln(P(0.04-0.15 Hz) + 1e-3); NaN unless usable, W >= 3 / 0.04 s and 0.15 Hz <= Nyquist
  bpm_lowratio_W  P_low / (P_low + P_mid + 1e-3); NaN unless both bands are defined
  bpm_rmssd_W     RMS of successive differences over consecutive samples (dt <= 3 s); NaN unless usable
  bpm_npairs_W    number of such pairs
  bpm_sampen_W    sample entropy (m = 2, r = 0.2 SD) of the longest run of consecutive samples (dt <= 3 s) in
                  the window; NaN unless the window is usable and that run has >= 100 samples
  bpm_slope_W     least-squares slope of BPM over the window, bpm per minute; NaN unless usable
  bpm_level_W     mean BPM over the window minus the lowest usable 10-min mean among the 10-min windows that end
                  at the last 240 epoch ends (the preceding 2 h, this epoch included); NaN unless both exist
  bpm_valid_W     1 if the window is usable, else 0
Cohort rules: Empatica E4 cohorts (wearanize, aauwss) are low-passed above 0.04 Hz (measured, log section 14), so the
mid band, ratio, successive differences and sample entropy are NaN there. Every other stream is used at its native
rate: a cohort whose median sampling interval is dt has Nyquist 1 / (2 dt); the mid band is 0.04 Hz to MID_HI
(0.15 Hz; 0.10 Hz for SleepAccel and the accelerometer-derived 5 s series, whose Nyquist is 0.10 Hz); successive
differences use consecutive samples at most max(3 s, 1.5 dt) apart; sample entropy uses the longest run of consecutive
samples with at least max(30, ceil(100 s / dt)) samples (100 samples at 1 s, 30 samples at 5 s).
The same function builds the accelerometer-derived columns with prefix "ahr" (cohort "accelhr", dt 5 s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH_S = 30
WINDOWS_MIN = (2, 5, 10)
LOW = (0.01, 0.04)
MID = (0.04, 0.15)
MIN_COVERAGE = 0.8
MAX_PAIR_DT = 3.0
SAMPEN_MIN_RUN = 100
LOOKBACK_EPOCHS = 240
POWER_EPS = 1e-3
HR_MIN, HR_MAX = 30.0, 200.0
LOW_PASSED = ("wearanize", "aauwss")
MID_HI_DEFAULT = 0.15
MID_HI = {"sleepaccel": 0.10, "accelhr": 0.10}


def columns_for(w: int, p: str = "bpm") -> list[str]:
    cols = []
    if w * 60 >= 3 / LOW[0]:
        cols.append(f"{p}_lowlogp_{w}m")
    cols.append(f"{p}_midlogp_{w}m")
    if w * 60 >= 3 / LOW[0]:
        cols.append(f"{p}_lowratio_{w}m")
    cols += [f"{p}_{n}_{w}m" for n in ("rmssd", "npairs", "sampen", "slope", "level", "valid")]
    return cols


ALL_COLS = [c for w in WINDOWS_MIN for c in columns_for(w)]
COLS_5_10 = [c for w in (5, 10) for c in columns_for(w)]
AHR_COLS_5_10 = [c for w in (5, 10) for c in columns_for(w, "ahr")]


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Sample entropy with the template count of the audit's version (n - m templates for m and m + 1)."""
    n = x.size
    sd = x.std()
    if n < m + 2 or sd == 0:
        return np.nan
    r = r_frac * sd
    d1 = np.abs(x[:, None] - x[None, :])
    k = n - m
    bm = np.maximum(d1[:k, :k], d1[1 : k + 1, 1 : k + 1])
    am = np.maximum(bm, d1[2 : k + 2, 2 : k + 2])
    b = (np.count_nonzero(bm <= r) - k) / 2.0
    a = (np.count_nonzero(am <= r) - k) / 2.0
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def _band(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    return psd[:, sel].sum(axis=1) * (freqs[1] - freqs[0])


def _empty(epochs: np.ndarray, p: str = "bpm") -> pd.DataFrame:
    out = pd.DataFrame({c: np.full(epochs.size, np.nan, np.float32) for c in [x.replace("bpm_", p + "_", 1) for x in ALL_COLS]})
    for w in WINDOWS_MIN:
        out[f"{p}_valid_{w}m"] = np.float32(0.0)
    out.insert(0, "epoch", epochs)
    return out


def night_features(hr_t, hr_bpm, n_epochs: int, epochs: np.ndarray, cohort: str, p: str = "bpm") -> pd.DataFrame:
    """Features of the given epochs of one night; ``hr_t`` is seconds from the start of epoch 0."""
    from senpy import compute_nufft_spectrogram

    epochs = np.asarray(epochs, dtype=int)
    t = np.asarray(hr_t, float)
    b = np.asarray(hr_bpm, float)
    keep = np.isfinite(t) & np.isfinite(b) & (b >= HR_MIN) & (b <= HR_MAX) & (t >= 0) & (t < n_epochs * EPOCH_S)
    t, b = t[keep], b[keep]
    order = np.argsort(t, kind="stable")
    t, b = t[order], b[order]
    if t.size:
        first = np.r_[True, np.diff(t) > 0]  # one value per timestamp
        t, b = t[first], b[first]
    if t.size < 20:
        return _empty(epochs, p)
    dt_nom = float(np.median(np.diff(t)))
    nyq = 1.0 / (2.0 * dt_nom)
    low_passed = cohort in LOW_PASSED
    mid_hi = MID_HI.get(cohort, MID_HI_DEFAULT)
    mid_ok = (not low_passed) and mid_hi <= nyq + 1e-9
    low_band_ok = LOW[1] <= nyq
    diff_ok = not low_passed
    max_gap = max(15.0, 3.0 * dt_nom)
    gaps = np.diff(t)
    max_pair = max(MAX_PAIR_DT, 1.5 * dt_nom)
    sampen_min = max(30, int(np.ceil(SAMPEN_MIN_RUN / dt_nom)))
    pair_ok = gaps <= max_pair  # pair (i, i + 1)
    dsq = np.where(pair_ok, np.diff(b) ** 2, 0.0)
    cs_sq = np.r_[0.0, np.cumsum(dsq)]  # cs[i] = sum over pairs < i
    cs_n = np.r_[0, np.cumsum(pair_ok)]
    cs_b = np.r_[0.0, np.cumsum(b)]
    all_ep = np.arange(n_epochs)
    ends_all = (all_ep + 1) * float(EPOCH_S)

    def basics(w: int):
        W = w * 60.0
        lo = np.searchsorted(t, ends_all - W, "left")
        hi = np.searchsorted(t, ends_all, "left")
        n = hi - lo
        valid = np.zeros(n_epochs, bool)
        for k in np.flatnonzero(n > 0):
            inner = gaps[lo[k] : hi[k] - 1].max() if n[k] > 1 else 0.0
            g = max(inner, t[lo[k]] - (ends_all[k] - W), ends_all[k] - t[hi[k] - 1])
            valid[k] = (n[k] * dt_nom / W >= MIN_COVERAGE) and g <= max_gap
        mean = np.where(n > 0, (cs_b[hi] - cs_b[lo]) / np.maximum(n, 1), np.nan)
        return lo, hi, n, valid, mean

    out = {"epoch": epochs}
    tmin = None
    lo10, hi10, n10, valid10, mean10 = basics(10)
    m10 = np.where(valid10, mean10, np.nan)
    floor10 = pd.Series(m10).rolling(LOOKBACK_EPOCHS, min_periods=1).min().to_numpy()
    for w in WINDOWS_MIN:
        lo, hi, n, valid, mean = (lo10, hi10, n10, valid10, mean10) if w == 10 else basics(w)
        W = w * 60.0
        sel = epochs
        v = valid[sel]
        out[f"{p}_valid_{w}m"] = v.astype(np.float32)
        out[f"{p}_level_{w}m"] = np.where(v, mean[sel] - floor10[sel], np.nan).astype(np.float32)
        slope = np.full(sel.size, np.nan)
        rms = np.full(sel.size, np.nan)
        npairs = np.full(sel.size, np.nan)
        sampen = np.full(sel.size, np.nan)
        for i, k in enumerate(sel):
            if not valid[k]:
                continue
            a, c = lo[k], hi[k]
            tw, bw = t[a:c], b[a:c]
            slope[i] = np.polyfit(tw / 60.0 - tw.mean() / 60.0, bw, 1)[0]
            if diff_ok:
                np_ = cs_n[c - 1] - cs_n[a]
                npairs[i] = np_
                if np_ > 0:
                    rms[i] = np.sqrt((cs_sq[c - 1] - cs_sq[a]) / np_)
                brk = np.flatnonzero(~pair_ok[a : c - 1])
                seg = np.diff(np.r_[-1, brk, c - a - 1])
                j = int(np.argmax(seg))
                if seg[j] >= sampen_min:
                    start = np.r_[-1, brk][j] + 1
                    sampen[i] = sample_entropy(bw[start : start + seg[j]])
        out[f"{p}_slope_{w}m"] = slope.astype(np.float32)
        out[f"{p}_rmssd_{w}m"] = rms.astype(np.float32)
        out[f"{p}_npairs_{w}m"] = npairs.astype(np.float32)
        out[f"{p}_sampen_{w}m"] = sampen.astype(np.float32)
        # spectra: one NUFFT PSD per window, window j ends at 30 (j + 1) s
        plow = np.full(sel.size, np.nan)
        pmid = np.full(sel.size, np.nan)
        want_low = low_band_ok and W >= 3 / LOW[0]
        if want_low or mid_ok:
            try:
                r = compute_nufft_spectrogram(
                    t, b, W, W - EPOCH_S, kind="psd", origin_s=float(EPOCH_S - W), empty_windows="keep", min_samples=8
                )
                ok = r.valid & (r.window_index < n_epochs)
                full = np.full((n_epochs, r.Sxx.shape[1]), np.nan)
                full[r.window_index[ok]] = r.Sxx[ok]
                if want_low:
                    plow = _band(r.frequencies, full[sel], *LOW)
                if mid_ok:
                    pmid = _band(r.frequencies, full[sel], MID[0], mid_hi)
            except Exception:  # too few samples overall
                pass
        plow = np.where(v, plow, np.nan)
        pmid = np.where(v, pmid, np.nan)
        if want_low:
            out[f"{p}_lowlogp_{w}m"] = np.log(plow + POWER_EPS).astype(np.float32)
        out[f"{p}_midlogp_{w}m"] = np.log(pmid + POWER_EPS).astype(np.float32)
        if want_low:
            out[f"{p}_lowratio_{w}m"] = (plow / (plow + pmid + POWER_EPS)).astype(np.float32)
    return pd.DataFrame({"epoch": epochs, **{c.replace("bpm_", p + "_", 1): out[c.replace("bpm_", p + "_", 1)] for c in ALL_COLS}})

"""Explainer figures for the accelerometer pulse-rate tracker (2026-10-28 §23):
  1. autocorrelation_idea.png   synthetic 30 s window: pulse + noise, lagged copies, autocorrelation vs candidate rate
  2. tracking_idea.png          synthetic 30 min: per-window peaks jump, the forward tracker follows the true rate
  3. sleepaccel_example.png     one SleepAccel stretch of still sleep where the method works (selected by find_window.py)
Uses ahr2 (the run's estimator code) for the real-data panels and the same autocorrelation definition for the synthetic ones."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import sosfilt
sys.path.insert(0, os.path.join(os.environ["WT"], "watchtree/results/accel-hr-v2-2026-10-28/setup"))
import ahr2 as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID = "#1f1f1e", "#5f5e58", "#9b9a93", "#e4e3dd"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
                     "axes.titleweight": "normal", "axes.titlelocation": "left", "legend.frameon": False})
FS = A.FS
rng = np.random.default_rng(7)


def acf_curve(seg):
    """Normalised autocorrelation of a (n, 3) window, summed over axes, at the lag of every candidate rate (as ahr2.acf_curves)."""
    seg = seg - seg.mean(axis=0); wl = seg.shape[0]; v = (seg**2).sum()
    lag_f = FS * 60.0 / A.GRID_BPM; l0 = np.floor(lag_f).astype(int); fr = lag_f - l0
    r = np.zeros(l0.max() + 2)
    for L in range(1, r.size):
        r[L] = (seg[:-L] * seg[L:]).sum() / (wl - L)
    r /= v / wl
    return r[l0] * (1 - fr) + r[l0 + 1] * fr


def r_at_lag(seg, lag_s):
    seg = seg - seg.mean(axis=0); L = int(round(lag_s * FS)); wl = seg.shape[0]
    return float((seg[:-L] * seg[L:]).sum() / (wl - L) / ((seg**2).sum() / wl))


def pulse(t, bpm_of_t, amp_mg=3.0):
    """BCG-like beats: a sharp J wave flanked by smaller I and K waves, beat times from the integrated rate; three axes."""
    phase = np.cumsum(bpm_of_t / 60.0) / FS
    beats = np.flatnonzero(np.diff(np.floor(phase)) > 0) / FS
    beats = beats + rng.normal(0, 0.015, beats.size)
    x = np.zeros_like(t)
    for b in beats:
        for dt, a, w in ((-0.06, -0.35, 0.025), (0.0, 1.0, 0.03), (0.07, -0.55, 0.035)):
            x += a * np.exp(-0.5 * ((t - b - dt) / w) ** 2)
    gains = np.array([1.0, 0.6, -0.4])
    return amp_mg * x[:, None] * gains[None, :]


def noise(n, sd_mg):
    return sosfilt(A._SOS_8, rng.normal(0, sd_mg, (n, 3)) * 2.2, axis=0)


# ---------- 1. one window ----------
def fig1():
    W = 30.0; t = np.arange(int(W * FS)) / FS; hr = 66.0; beat = 60 / hr
    clean = pulse(t, np.full(t.size, hr)); obs = clean + noise(t.size, 0.55)
    obs = sosfilt(A._SOS_8, obs, axis=0)
    curve = acf_curve(obs)
    good, bad = beat, 0.62
    fig = plt.figure(figsize=(11, 7.2), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1])
    a = fig.add_subplot(gs[0, 0]); show = t <= 8
    off = obs[show, 0].max() - clean[show, 0].min() + 2.5
    a.plot(t[show], clean[show, 0] + off, color=MUTED, lw=1.2)
    a.plot(t[show], obs[show, 0], color=INK, lw=1.0)
    a.text(8.1, off, "pulse alone", color=INK2, fontsize=9, va="center")
    a.text(8.1, 0, "pulse + noise,\nfiltered 0.67–8 Hz", color=INK2, fontsize=9, va="center")
    a.set_yticks([]); a.spines["left"].set_visible(False); a.set_xlim(0, 8)
    a.set_title("A. A 30 s window of wrist acceleration (first 8 s shown)")
    for k, (lag, name) in enumerate(((good, f"shifted by one beat ({good:.2f} s)"), (bad, f"shifted by {bad:.2f} s"))):
        ax = fig.add_subplot(gs[k + 1, 0], sharex=a)
        L = int(round(lag * FS)); tt = t[:-L]; m = tt <= 8
        ax.plot(tt[m], obs[:-L, 0][m], color=INK, lw=1.0, label="window")
        ax.plot(tt[m], obs[L:, 0][m], color=BLUE if k == 0 else ORANGE, lw=1.0, alpha=0.9, label="the same window, " + name)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        r = r_at_lag(obs, lag)
        ax.set_title(("B" if k == 0 else "C") + f". Multiply the window by a shifted copy and average: r = {r:+.2f}".replace("-", "−"))
        if k == 1:
            ax.set_xlabel("Time (s)")
    c = fig.add_subplot(gs[:, 1])
    c.axhline(0, color=GRID, lw=1)
    c.plot(A.GRID_BPM, curve, color=INK, lw=2)
    for lag, col in ((good, BLUE), (bad, ORANGE)):
        b = 60 / lag; c.plot([b], [np.interp(b, A.GRID_BPM, curve)], "o", ms=9, color=col, mec="white", mew=2)
    c.annotate(f"peak at {A.GRID_BPM[np.argmax(curve)]:.0f} bpm\n(true rate {hr:.0f} bpm)", (A.GRID_BPM[np.argmax(curve)], curve.max()),
               xytext=(95, curve.max() * 0.95), color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    c.set_xlabel("Candidate pulse rate (bpm) = 60 / shift (s)")
    c.set_ylabel("Autocorrelation r")
    c.set_title("D. r for every candidate rate, 40–180 bpm")
    c.set_xlim(40, 180); c.grid(axis="y", color=GRID, lw=0.6)
    fig.savefig(os.path.join(OUT, "autocorrelation_idea.png"), dpi=150); plt.close(fig)


# ---------- 2. tracking ----------
def fig2():
    n = 60; W = 30.0; tw = np.arange(int(W * FS)) / FS
    true = 64 + 5 * np.sin(np.linspace(0, 1.6 * np.pi, n)) + np.linspace(0, 3, n)
    burst = np.zeros(n, bool); burst[[6, 7, 15, 22, 23, 24, 31, 38, 39, 44, 51, 52]] = True
    C = np.empty((n, A.GRID_BPM.size))
    for k in range(n):
        x = pulse(tw, np.full(tw.size, true[k]), amp_mg=2.2) + noise(tw.size, 0.8 if not burst[k] else 6.0)
        x = sosfilt(A._SOS_8, x, axis=0)
        C[k] = acf_curve(x)
    peak = A.GRID_BPM[np.argmax(C, axis=1)]
    est, _ = A.viterbi_filter(10.0 * C, 1.0)
    tmin = (np.arange(n) + 1) * 0.5
    fig, (h, e) = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", sharex=True, height_ratios=[1.5, 1])
    im = h.imshow(C.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=np.percentile(C, 99.5),
                  extent=(tmin[0] - 0.25, tmin[-1] + 0.25, A.GRID_BPM[0] - 0.5, A.GRID_BPM[-1] + 0.5))
    h.plot(tmin, true, color=ORANGE, lw=2, label="true rate")
    h.plot(tmin, peak, "o", ms=5, color=INK, mfc="white", mew=1.2, label="highest r in each window alone")
    h.set_ylabel("Candidate rate (bpm)"); h.set_ylim(40, 180)
    h.set_title("A. Autocorrelation r of each 30 s window (one column per window; darker = higher r)")
    h.legend(loc="upper right", ncol=2, fontsize=9, facecolor="white", frameon=True, edgecolor="none")
    fig.colorbar(im, ax=h, label="r", shrink=0.8)
    e.plot(tmin, true, color=ORANGE, lw=2, label="true rate")
    e.plot(tmin, peak, "o", ms=5, color=INK, mfc="white", mew=1.2, label="highest r in each window alone")
    e.plot(tmin, est, color=BLUE, lw=2, label="tracked estimate")
    e.set_ylim(40, 180); e.set_ylabel("Rate (bpm)"); e.set_xlabel("Time (min)")
    e.grid(axis="y", color=GRID, lw=0.6)
    e.set_title("B. Tracking: score(t, f) = 10 r(t, f) + max over f′ of [score(t − 1, f′) − λ |f − f′|],  λ = 1 per bpm;  estimate = argmax")
    e.legend(loc="upper right", ncol=3, fontsize=9)
    fig.savefig(os.path.join(OUT, "tracking_idea.png"), dpi=150); plt.close(fig)


# ---------- 3. SleepAccel ----------
def fig3(rep="781756", e0=857, L=40):
    import warnings; warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
    from loadnight import load_with_rows
    c = COHORTS["sleepaccel"]; ids, reps = subject_ids(c); sub = next(s for s in ids if reps[s] == rep)
    night, _ = load_with_rows("sleepaccel", sub, rep, load_dataset(DATA_ROOT / c.directory), None)
    ne = night.labels.size
    C = A.acf_curves(night.acc25, night.valid25, ne, 30, A._SOS_8)
    est, _ = A.viterbi_filter(10.0 * C, 1.0)
    pk, _ = A.pick(C, lin=False)
    n = ne * 30 * int(FS); acc = np.asarray(night.acc25[:n], float); ok = np.asarray(night.valid25[:n], bool)
    y = A._filt(A._SOS_8, acc, ok) * 1000.0
    ht, hb = np.asarray(night.hr_t, float), np.asarray(night.hr_bpm, float)
    ks = np.arange(e0, e0 + L)
    dev = np.array([np.nanmean(hb[(ht >= 30 * k) & (ht < 30 * (k + 1))]) for k in ks])
    # the example window: highest r at the device rate relative to the curve maximum
    rel = [np.interp(dev[i], A.GRID_BPM, C[k]) / np.nanmax(C[k]) for i, k in enumerate(ks)]
    i0 = int(np.argmax(rel)); k0 = ks[i0]
    tmin = (ks - e0 + 1) * 0.5

    fig = plt.figure(figsize=(11, 8.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1.15])
    a = fig.add_subplot(gs[0, 0])
    seg = y[k0 * 750: k0 * 750 + 750]; t = np.arange(seg.shape[0]) / FS
    sh = 0.0
    for j, name in enumerate("xyz"):
        a.plot(t, seg[:, j] - sh, color=INK, lw=0.8); a.text(30.3, -sh, name, color=INK2, va="center")
        sh += 4 * np.std(seg[:, j]) + 1e-9
    a.set_yticks([]); a.spines["left"].set_visible(False); a.set_xlim(0, 30); a.set_xlabel("Time in window (s)")
    a.set_title("A. One 30 s window of still sleep, filtered 0.67–8 Hz (bar: 1 mg)")
    a.plot([1, 1], [-sh + 0.5, -sh + 1.5], color=INK, lw=2)
    b = fig.add_subplot(gs[0, 1])
    b.axhline(0, color=GRID, lw=1)
    b.plot(A.GRID_BPM, C[k0], color=INK, lw=2)
    b.axvline(dev[i0], color=ORANGE, lw=2, label=f"Apple Watch HR, {dev[i0]:.0f} bpm")
    b.axvline(est[k0], color=BLUE, lw=2, ls=(0, (4, 2)), label=f"tracked estimate, {est[k0]:.0f} bpm")
    b.plot([pk[k0]], [np.interp(pk[k0], A.GRID_BPM, C[k0])], "o", ms=8, color=INK, mfc="white", mew=1.5,
           label=f"highest r in this window, {pk[k0]:.0f} bpm")
    b.set_xlim(40, 180); b.set_xlabel("Candidate rate (bpm)"); b.set_ylabel("Autocorrelation r")
    b.set_title("B. r of that window"); b.legend(fontsize=8.5, loc="upper right"); b.grid(axis="y", color=GRID, lw=0.6)
    h = fig.add_subplot(gs[1, :])
    Cs = C[ks]
    im = h.imshow(Cs.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=np.nanpercentile(Cs, 99.5),
                  extent=(tmin[0] - 0.25, tmin[-1] + 0.25, A.GRID_BPM[0] - 0.5, A.GRID_BPM[-1] + 0.5))
    h.plot(tmin, dev, color=ORANGE, lw=2, label="Apple Watch HR (mean of the window's ~6 samples)")
    h.plot(tmin, pk[ks], "o", ms=5, color=INK, mfc="white", mew=1.2, label="highest r in each window alone")
    h.plot(tmin, est[ks], color=BLUE, lw=2, label="tracked estimate (λ = 1)")
    h.axvline(tmin[i0], color=MUTED, lw=1, ls=":")
    h.set_ylim(40, 180); h.set_xlabel("Time (min)"); h.set_ylabel("Candidate rate (bpm)")
    h.set_title("C. 20 min of still sleep, one column per 30 s window (dotted: the window in A and B)")
    h.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    fig.colorbar(im, ax=h, label="r", shrink=0.8)
    fig.savefig(os.path.join(OUT, "sleepaccel_example.png"), dpi=150); plt.close(fig)
    err_t = np.abs(est[ks] - dev); err_p = np.abs(pk[ks] - dev)
    print(f"window {i0}: dev {dev[i0]:.1f} est {est[k0]:.1f} peak {pk[k0]:.1f}; stretch: tracked MAE {np.nanmean(err_t):.2f}, "
          f"per-window peak MAE {np.nanmean(err_p):.2f}, peak within 3 bpm {np.nanmean(err_p <= 3):.0%}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: fig1()
    if "2" in which: fig2()
    if "3" in which: fig3()

"""Figures: (a) individual stillest-epoch spectra with the device HR marked; (b) estimator error vs coverage (test participants)."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ahr2 as A
from common2 import R, COH, load_est
from splits import split0
LAB = {"sleepaccel": "SleepAccel (Apple Watch)", "wearanize": "Wearanize+ (Empatica E4)", "weaver_fitbit": "Weaver Fitbit", "weaver_garmin": "Weaver Garmin"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
(R / "figures").mkdir(exist_ok=True)
sp = split0()
# (a) four random individual 60 s windows per cohort from test participants (seeded; not selected by quality)
rng = np.random.default_rng(1)
fig, axes = plt.subplots(4, 4, figsize=(13, 10), constrained_layout=True)
for r, n in enumerate(("sleepaccel", "weaver_fitbit", "wearanize", "weaver_garmin")):
    files = [f for f in sorted(glob.glob(str(R / "still" / n / "*.npz"))) if f.split("/")[-1][:-4] in sp[n][1]]
    pool = []
    for f in files:
        z = np.load(f); S = z["S_sum_60"]; dev = z["dev60"]
        ok = np.flatnonzero(np.isfinite(S).all(axis=1) & np.isfinite(dev))
        pool += [(f, i) for i in ok]
    pick = rng.choice(len(pool), 4, replace=False)
    for c, k in enumerate(pick):
        f, i = pool[k]; z = np.load(f); S = z["S_sum_60"][i].astype(float); dev = float(z["dev60"][i])
        fr = 0.4 + np.arange(S.size) / 60.0; m = fr <= 6.0
        ax = axes[r, c]
        ax.semilogy(fr[m], S[m], color="#222222", lw=0.9)
        for h, ls in ((1, "-"), (2, "--")):
            ax.axvline(h * dev / 60, color="#D55E00" if h == 1 else "#0072B2", ls=ls, lw=1.2, alpha=0.9)
        ax.axvspan(0.67, 3.0, color="#999999", alpha=0.08, lw=0)
        ax.set_xlim(0.4, 6)
        if c == 0: ax.set_ylabel(f"{LAB[n]}\nPSD, sum of axes (g²/Hz)")
        if r == 3: ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"device HR {dev:.0f} bpm", fontsize=9)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color="#D55E00"), Line2D([], [], color="#0072B2", ls="--"), plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.2)],
           ["device HR", "2 × device HR", "0.67–3 Hz pulse band"], loc="outside lower center", ncol=3, frameon=False)
fig.savefig(R / "figures" / "stillest_epoch_spectra_with_device_hr.png", dpi=150); plt.close(fig)

# (b) error vs coverage
cand = pd.read_csv(R / "step3_candidates.csv", index_col=0).iloc[:, 0]
d = load_est()
FAMCOL = {"peak": "#999999", "hsum": "#E69F00", "hps": "#56B4E9", "sacf": "#009E73", "acf": "#F0E442", "acf_curve": "#CC79A7",
          "viterbi_spec": "#0072B2", "viterbi_acf": "#D55E00", "chosen": "#000000"}
FAMLAB = {"peak": "spectral peak", "hsum": "harmonic sum", "hps": "harmonic product", "sacf": "spectral autocorrelation", "acf": "autocorrelation peak",
          "acf_curve": "autocorrelation curve peak", "viterbi_spec": "tracked harmonic sum", "viterbi_acf": "tracked autocorrelation", "chosen": "chosen (tracked autocorrelation, 30 s)"}
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
for ax, n in zip(axes, ("sleepaccel", "wearanize", "weaver_fitbit", "weaver_garmin")):
    g = d[(d.cohort == n) & d.sleep & (d.split == "test")]
    for fam, col in FAMCOL.items():
        c = cand[f"{n}|{fam}"] if f"{n}|{fam}" in cand.index else None
        if c is None: continue
        W = int(c.split(".")[-1])
        s = g[[f"est.{c}", f"q.{c}", f"dev{W}"]].dropna()
        e = (s[f"est.{c}"] - s[f"dev{W}"]).abs().to_numpy()[np.argsort(-s[f"q.{c}"].to_numpy())]
        cum = np.cumsum(e) / np.arange(1, e.size + 1)
        share = np.arange(1, e.size + 1) / len(g)
        ax.plot(share, cum, color=col, lw=2.2 if fam == "chosen" else 1.0, zorder=3 if fam == "chosen" else 2)
        if fam == "chosen":
            const = d[(d.cohort == n) & d.sleep & (d.split == "train")].dev30.median()
            ref = s[f"dev{W}"].to_numpy()[np.argsort(-s[f"q.{c}"].to_numpy())]
            ax.plot(share, np.cumsum(np.abs(ref - const)) / np.arange(1, e.size + 1), color="#444444", ls="--", lw=1.4)
    ax.set_xscale("log"); ax.set_xlim(0.01, 1.0); ax.set_ylim(0, 45)
    ax.set_xlabel("Share of PSG-sleep epochs kept (highest quality first)"); ax.set_title(LAB[n], fontsize=10)
    if n == "sleepaccel": ax.set_ylabel("Mean absolute error vs device HR (bpm)")
handles = [Line2D([], [], color=FAMCOL[f], lw=2.2 if f == "chosen" else 1.0) for f in FAMCOL] + [Line2D([], [], color="#444444", ls="--", lw=1.4)]
fig.legend(handles, [FAMLAB[f] for f in FAMCOL] + ["constant (training median HR), same epochs as chosen"], loc="outside lower center", ncol=4, frameon=False)
fig.savefig(R / "figures" / "error_vs_coverage.png", dpi=150); plt.close(fig)

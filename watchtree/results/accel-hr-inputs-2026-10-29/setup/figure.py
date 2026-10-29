"""r of every 30 s window over the explainer's 20 min SleepAccel stretch, one panel per input arm A-E; lambda = the one selected in the CV fold that holds
the night (training-fold all-sleep MAE). Same style as accel-hr-envelope-2026-10-29/setup/figure.py (the night id is read from that script)."""
import os, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
import ahr2 as A, nativelib as NL
from splits import night_folds
N = os.environ["N"]; WT = os.environ["WT"]
src = open(f"{WT}/watchtree/results/accel-hr-envelope-2026-10-29/setup/figure.py").read()
rep, e0, L = re.search(r'rep, e0, L = "(\d+)", (\d+), (\d+)', src).groups(); e0, L = int(e0), int(L)
INK, INK2, MUTED = "#1f1f1e", "#5f5e58", "#9b9a93"; BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "normal",
                     "axes.titlelocation": "left", "legend.frameon": False})
from arcasleep.data.prepare import load_dataset
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from loadnight import load_with_rows
c = COHORTS["sleepaccel"]; ids, reps = subject_ids(c); sub = next(s for s in ids if reps[s] == rep)
night, rows = load_with_rows("sleepaccel", sub, rep, load_dataset(DATA_ROOT / c.directory), None)
f = night_folds(); fold = int(f[(f.cohort == "sleepaccel") & (f.subject == rep)].fold.iloc[0])
sel = pd.read_csv(os.path.join(N, "sleepaccel_selection.csv")).set_index(["arm", "fold"])
ne = night.labels.size; C = NL.curves_25(night.acc25, night.valid25, ne); Cn, fs = NL.curves_native(rows, night.t0, ne); C.update(Cn)
TITLES = {"A": "A. 0.67–8 Hz band-pass, 25 Hz stream (current)", "B": "B. 0.67–3 Hz band-pass, 25 Hz stream", "C": "C. 3 Hz low-pass, 25 Hz stream",
          "D": "D. per-axis derivative, raw samples", "E": "E. 0.67–8 Hz band-pass, raw samples at the nominal rate"}
ht, hb = np.asarray(night.hr_t, float), np.asarray(night.hr_bpm, float)
ks = np.arange(e0, e0 + L); tmin = (ks - e0 + 1) * 0.5
dev = np.array([np.nanmean(hb[(ht >= 30 * k) & (ht < 30 * (k + 1))]) for k in ks])
fig, axs = plt.subplots(5, 1, figsize=(11, 17), layout="constrained", sharex=True)
vmax = max(np.nanpercentile(C[k][ks], 99.5) for k in TITLES)
for ax, (key, title) in zip(axs, TITLES.items()):
    lam = float(sel.loc[(key, fold), "lam_selected"]); Ck = C[key]; est, _ = A.viterbi_filter(10.0 * Ck, lam); pk, _ = A.pick(Ck, lin=False)
    im = ax.imshow(Ck[ks].T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                   extent=(tmin[0] - 0.25, tmin[-1] + 0.25, A.GRID_BPM[0] - 0.5, A.GRID_BPM[-1] + 0.5))
    ax.plot(tmin, dev, color=ORANGE, lw=2, label="Apple Watch HR (mean of the window's ~6 samples)")
    ax.plot(tmin, pk[ks], "o", ms=5, color=INK, mfc="white", mew=1.2, label="highest r in each window alone")
    ax.plot(tmin, est[ks], color=BLUE, lw=2, label="tracked estimate")
    ax.set_ylim(40, 180); ax.set_ylabel("Candidate rate (bpm)"); ax.set_title(f"{title}, λ = {lam:g}")
    e = np.abs(est[ks] - dev); p = np.abs(pk[ks] - dev)
    ax.text(0.995, 0.96, f"tracked MAE {np.nanmean(e):.1f} bpm; peak within 3 bpm in {np.nanmean(p <= 3):.0%} of windows", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK2, bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.colorbar(im, ax=ax, label="r", shrink=0.8)
axs[-1].set_xlabel("Time (min)")
axs[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
fig.savefig(os.path.join(N, "figures", "inputs_vs_current_sleepaccel_example.png"), dpi=130)
print(fold, fs)

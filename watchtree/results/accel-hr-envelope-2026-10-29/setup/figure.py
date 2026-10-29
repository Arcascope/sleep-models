"""r of every 30 s window over the explainer's 20 min SleepAccel stretch (accel-hr-explainer-2026-10-29, panel C): current, Hilbert envelope and
squared + low-pass envelope estimators (candidates chosen in the CV fold that holds the night, 'all' rule). Same style as explainer.py."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
import ahr2 as A, envlib as E
from splits import night_folds
N = os.environ["N"]
INK, INK2, MUTED, GRID = "#1f1f1e", "#5f5e58", "#9b9a93", "#e4e3dd"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "normal",
                     "axes.titlelocation": "left", "legend.frameon": False})
rep, e0, L = "781756", 857, 40
from arcasleep.data.prepare import load_dataset
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from loadnight import load_with_rows
c = COHORTS["sleepaccel"]; ids, reps = subject_ids(c); sub = next(s for s in ids if reps[s] == rep)
night, _ = load_with_rows("sleepaccel", sub, rep, load_dataset(DATA_ROOT / c.directory), None)
f = night_folds(); fold = int(f[(f.cohort == "sleepaccel") & (f.subject == sub if sub in set(f.subject) else f.subject == rep)].fold.iloc[0])
sel = pd.read_csv(os.path.join(N, "sleepaccel_selection.csv")).set_index(["estimator", "fold"])
def parse(name):  # e.g. hil_b48_l2p0 or lp_b48_f3_l0p5
    base, lam = name.rsplit("_l", 1); return base, float(lam.replace("p", "."))
ne = night.labels.size; C = E.all_curves(night.acc25, night.valid25, ne)
rows = [("Current: 0.67–8 Hz signal, λ = 1", "cur", 1.0)]
for fam, title in (("hil", "Hilbert envelope"), ("lp", "Squared + low-pass envelope")):
    base, lam = parse(sel.loc[(fam, fold), "pick_all"])
    rows.append((f"{title}: {base.split('_', 1)[1].replace('_f', ', low-pass ').replace('b38', '3–8 Hz').replace('b48', '4–8 Hz')}"
                 .replace("3–8 Hz, low-pass", "3–8 Hz, low-pass ").replace("4–8 Hz, low-pass", "4–8 Hz, low-pass ") + f", λ = {lam:g}", base, lam))
rows = [(t.replace("low-pass  ", "low-pass ").replace("low-pass 2", "low-pass 2 Hz").replace("low-pass 3", "low-pass 3 Hz").replace("low-pass 4", "low-pass 4 Hz"), k, l) for t, k, l in rows]
ht, hb = np.asarray(night.hr_t, float), np.asarray(night.hr_bpm, float)
ks = np.arange(e0, e0 + L); tmin = (ks - e0 + 1) * 0.5
dev = np.array([np.nanmean(hb[(ht >= 30 * k) & (ht < 30 * (k + 1))]) for k in ks])
fig, axs = plt.subplots(3, 1, figsize=(11, 10.5), layout="constrained", sharex=True)
vmax = max(np.nanpercentile(C[k][ks], 99.5) for _, k, _ in rows)
for ax, (title, key, lam) in zip(axs, rows):
    Ck = C[key]; est, _ = A.viterbi_filter(10.0 * Ck, lam); pk, _ = A.pick(Ck, lin=False)
    im = ax.imshow(Ck[ks].T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                   extent=(tmin[0] - 0.25, tmin[-1] + 0.25, A.GRID_BPM[0] - 0.5, A.GRID_BPM[-1] + 0.5))
    ax.plot(tmin, dev, color=ORANGE, lw=2, label="Apple Watch HR (mean of the window's ~6 samples)")
    ax.plot(tmin, pk[ks], "o", ms=5, color=INK, mfc="white", mew=1.2, label="highest r in each window alone")
    ax.plot(tmin, est[ks], color=BLUE, lw=2, label="tracked estimate")
    ax.set_ylim(40, 180); ax.set_ylabel("Candidate rate (bpm)"); ax.set_title(title)
    e = np.abs(est[ks] - dev); p = np.abs(pk[ks] - dev)
    ax.text(0.995, 0.96, f"tracked MAE {np.nanmean(e):.1f} bpm; peak within 3 bpm in {np.nanmean(p <= 3):.0%} of windows", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK2, bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.colorbar(im, ax=ax, label="r", shrink=0.8)
axs[-1].set_xlabel("Time (min)")
axs[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
fig.savefig(os.path.join(N, "figures", "envelope_vs_current_sleepaccel_example.png"), dpi=150)
print(rows, fold)

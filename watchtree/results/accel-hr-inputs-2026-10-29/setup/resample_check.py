"""Resampling check: (1) timestamp steps of the raw streams, (2) per-window NUFFT PSD of the raw samples (own timestamps) vs the 25 Hz stream.
Windows: still PSG-sleep epochs (lowest quarter of the night's movement among sleep epochs), spread over nights. Nothing is averaged over windows.
A 'line' is a frequency bin whose PSD is >= T times the running median of the bins within +-0.15 Hz (T = 5 and 10); a line is 'in the other' PSD when any bin
within +-1 bin of it is a line there. Writes resample_*.csv at the run dir top level (no participant ids) and figures/resampling_psd_examples.png."""
import os, glob
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
N = os.environ["N"]
BANDS = [(0.67, 2), (2, 3), (3, 5), (5, 8), (8, 11), (11, 12.5), (0.67, 8), (0.67, 12.5)]
TS = (5.0, 10.0)
dt_rows, psd_rows, line_rows, win = [], [], [], {}
for coh in ("sleepaccel", "wearanize"):
    sub = {}
    for p in sorted(glob.glob(f"{N}/dt/{coh}/*.npy")):
        z = np.load(p); fs = int(z[0]); g = sub.setdefault(fs, dict(steps=0, gaps=0, off=0, samp=[], nights=0))
        g["steps"] += z[1]; g["gaps"] += z[2]; g["off"] += z[3]; g["samp"].append(z[4:]); g["nights"] += 1
    groups = {"all": list(sub)} | {f"{fs}": [fs] for fs in sub}
    for lab, fss in groups.items():
        steps = sum(sub[f]["steps"] for f in fss); s = np.concatenate([x for f in fss for x in sub[f]["samp"]])
        fsn = np.mean([f for f in fss]) if lab != "all" else np.nan
        dt_rows.append(dict(cohort=coh, nominal_rate_hz=lab, nights=sum(sub[f]["nights"] for f in fss), steps=int(steps), median_dt_s=np.median(s),
                            p1_dt_s=np.percentile(s, 1), p99_dt_s=np.percentile(s, 99),
                            share_off_nominal_gt1pct=sum(sub[f]["off"] for f in fss) / steps, share_gap=sum(sub[f]["gaps"] for f in fss) / steps))
    # PSD windows
    raws, r25s, fr = [], [], None
    for p in sorted(glob.glob(f"{N}/psd/{coh}/*.npz")):
        z = np.load(p)
        if z["f_raw"].size != z["f_25"].size:   # nominal rate below 25 Hz: the raw PSD does not reach 12.5 Hz
            continue
        raws.append(z["raw"]); r25s.append(z["r25"]); fr = z["f_raw"]
    raw = np.concatenate(raws); r25 = np.concatenate(r25s)
    keep = raw.sum(1) > 0   # a window whose raw samples are constant has no raw power
    print(coh, "windows with zero raw power, dropped:", int((~keep).sum())); raw, r25 = raw[keep], r25[keep]; win[coh] = (fr, raw, r25)
    n = raw.shape[0]
    for lo, hi in BANDS:
        sel = (fr >= lo) & (fr < hi)
        ratio = r25[:, sel].sum(1) / raw[:, sel].sum(1)
        psd_rows.append(dict(cohort=coh, windows=n, band_hz=f"{lo}-{hi}", median_ratio=np.median(ratio), p5=np.percentile(ratio, 5), p95=np.percentile(ratio, 95),
                             share_windows_ratio_outside_0p9_1p1=((ratio < 0.9) | (ratio > 1.1)).mean()))
    k = 2 * int(round(0.15 / (fr[1] - fr[0]))) + 1; band = (fr >= 0.67)
    for T in TS:
        def lines(P):
            return (P / np.maximum(median_filter(P, size=(1, k), mode="nearest"), 1e-30) >= T) & band
        Lr, L2 = lines(raw), lines(r25)
        dil = lambda L: L | np.roll(L, 1, axis=1) | np.roll(L, -1, axis=1)
        only25, onlyraw = L2 & ~dil(Lr), Lr & ~dil(L2)
        line_rows.append(dict(cohort=coh, windows=n, T=T, share_windows_with_line_raw=Lr.any(1).mean(), share_windows_with_line_25hz=L2.any(1).mean(),
                              share_windows_25hz_only_line=only25.any(1).mean(), share_windows_raw_only_line=onlyraw.any(1).mean(),
                              mean_lines_raw=Lr.sum(1).mean(), mean_lines_25hz=L2.sum(1).mean(), mean_25hz_only=only25.sum(1).mean(), mean_raw_only=onlyraw.sum(1).mean()))
        if T == TS[0]:
            win[coh] += (only25.sum(1), np.median(r25[:, (fr >= .67) & (fr < 8)].sum(1) / raw[:, (fr >= .67) & (fr < 8)].sum(1)))
pd.DataFrame(dt_rows).to_csv(f"{N}/resample_dt.csv", index=False); pd.DataFrame(psd_rows).to_csv(f"{N}/resample_psd_ratio.csv", index=False)
pd.DataFrame(line_rows).to_csv(f"{N}/resample_lines.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for t in ("dt", "psd_ratio", "lines"):
    print(t); print(pd.read_csv(f"{N}/resample_{t}.csv").round(4).to_string(index=False))
# figure: median-ratio window of each cohort, and the window with the most 25 Hz-only lines (T = 5)
INK, INK2, MUTED = "#1f1f1e", "#5f5e58", "#9b9a93"; BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlelocation": "left", "legend.frameon": False})
picks = []
for coh in ("sleepaccel", "wearanize"):
    fr, raw, r25, only, med = win[coh]; sel = (fr >= .67) & (fr < 8); ratio = r25[:, sel].sum(1) / raw[:, sel].sum(1)
    picks.append((coh, int(np.argmin(np.abs(ratio - med))), f"{'SleepAccel' if coh == 'sleepaccel' else 'Wearanize+'}: window with the median 0.67–8 Hz power ratio"))
c = max(("sleepaccel", "wearanize"), key=lambda k: win[k][3].max()); picks.append((c, int(np.argmax(win[c][3])), f"{'SleepAccel' if c == 'sleepaccel' else 'Wearanize+'}: window with the most lines only in the 25 Hz stream"))
fig, axs = plt.subplots(3, 1, figsize=(10, 9), layout="constrained", sharex=True)
for ax, (coh, i, title) in zip(axs, picks):
    fr, raw, r25 = win[coh][:3]
    ax.semilogy(fr, raw[i], color=ORANGE, lw=1.4, label="raw samples, own timestamps"); ax.semilogy(fr, r25[i], color=BLUE, lw=1.0, label="25 Hz stream")
    ax.set_title(title); ax.set_ylabel("PSD (g²/Hz, sum of 3 axes)"); ax.axvspan(0.67, 8, color="#e4e3dd", alpha=0.35, lw=0)
axs[-1].set_xlabel("Frequency (Hz)"); axs[0].legend(loc="upper right"); axs[0].set_xlim(0, 12.5)
fig.savefig(f"{N}/figures/resampling_psd_examples.png", dpi=150)

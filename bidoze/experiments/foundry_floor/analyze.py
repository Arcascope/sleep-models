#!/usr/bin/env python
"""Land device floor analysis on the Foundry Galaxy Watch and iPhone data.

Non-PSG data.  Output:
  bidoze/results/foundry-floor-land-YYYYMMDD/summary.csv
  .../figures/*.png
"""
import glob, os, re, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/media/eric/Extreme SSD/pisces2-data/Foundry"
OUT = "/home/eric/Engineering/Work/autopisces/sleep-models-foundry-floor/bidoze/results/foundry-floor-land-20261013"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

FS_TARGET = 50.0
EPOCH_SEC = 30.0
BAND = (2.0, 6.0)
DEVICE_SCALE = {"Eric-Galaxy": 1.0, "Eric-iPhone": 1.0}

def subject_of(fn): return fn.split("--")[0] if "--" in fn else None
def windows_for(subject):
    return sorted(glob.glob(os.path.join(DATA, "cleaned_accelerometer", f"{subject}-*_accelerometer.csv")))

def read_window(csv):
    df = pd.read_csv(csv)
    t = df["timestampMs"].to_numpy(float) / 1000.0
    xyz = df[["x", "y", "z"]].to_numpy(float)
    # Units are already in g for this cleaned data set
    zr = (t[0], t[-1], len(df))
    # uniform resample onto fixed grid
    fs = 1.0 / np.median(np.diff(t))
    # use union range of data
    t_uniform = np.arange(t[0], t[-1], 1.0/FS_TARGET)
    xyz_uniform = np.zeros((len(t_uniform), 3))
    for i in range(3):
        xyz_uniform[:, i] = np.interp(t_uniform, t, xyz[:, i])
    return t, xyz, xyz_uniform, fs, t_uniform

def frames_for_night(xyz_uniform, frame_sec=10.0, stride_sec=10.0):
    fs = FS_TARGET
    frame_n = int(frame_sec * fs)
    stride_n = int(stride_sec * fs)
    out = []
    for st in range(0, len(xyz_uniform) - frame_n, stride_n):
        seg = xyz_uniform[st:st+frame_n, :]
        out.append((st / fs, seg))
    return out

# frame PSD implemented in frame_power_db
# Need implement welch
from scipy.signal import welch
def frame_power_db(seg, fs=50.0, band=(2,6)):
    # mean per-bin trace PSD across 10s window; return dB of mean PSD in band for all axes
    pxxs = []
    for ax in range(3):
        f, pxx = welch(seg[:, ax], fs=fs, nperseg=len(seg), noverlap=0, detrend="linear")
        pxxs.append(pxx)
    pxxs = np.array(pxxs)  # 3 x F
    trace = pxxs.mean(axis=0)  # average power per Hz across axes? per-bin trace
    # dB relative to g^2/Hz
    ind = (f >= band[0]) & (f <= band[1])
    mean_db = 10*np.log10(np.mean(trace[ind])) if np.any(trace[ind] > 0) else np.nan
    # RMS in band (mg)
    df = f[1]-f[0]
    rms_g = np.sqrt(np.trapezoid(trace[ind], f[ind]))
    rms_mg = rms_g * 1000.0
    return mean_db, rms_mg, f, trace

def floor_for_window(csv):
    t, xyz, uni, fs, tu = read_window(csv)
    frames = frames_for_night(uni)
    rows = []
    for time_s, seg in frames:
        db, rms, f, trace = frame_power_db(seg)
        # SW magnitude (movement): std of magnitude
        mag = np.sqrt(np.sum(seg**2, axis=1))
        sw = np.std(mag) * 1000.0
        if not np.isfinite(db) or sw > 200.0 or sw <= 1e-9:
            continue
        rows.append({"t": time_s, "db": db, "rms_mg": rms, "move_mg": sw})
    dd = pd.DataFrame(rows)
    # overall q10 db; q10 by 30s epoch not necessary but store
    q10_db = np.percentile(dd["db"].dropna(), 10)
    # corresponding mg from dB: 10^(q10/10) integrated? approximate RMS for 4Hz band
    q10_rms = np.percentile(dd["rms_mg"].dropna(), 10)
    return dd, q10_db, q10_rms, fs

rows = []
for subject in ["Eric-Galaxy", "Eric-iPhone"]:
    for csv in windows_for(subject):
        dd, q10_db, q10_rms, fs = floor_for_window(csv)
        # quietest 10 frames
        quiet = dd.nsmallest(10, "db")
        rows.append({
            "subject": subject,
            "window": os.path.basename(csv),
            "n_frames": len(dd),
            "floor_db_q10": q10_db,
            "rms_mg_q10": q10_rms,
            "quiet_db_median": quiet["db"].median(),
            "quiet_rms_mg_median": quiet["rms_mg"].median(),
            "quiet_move_mg_p50": dd["move_mg"].median(),
            "fs_est": fs
        })
        outcsv = os.path.join(FIG, f"frame_levels_{os.path.basename(csv)}.csv")
        dd.to_csv(outcsv, index=False)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
print(summary.groupby("subject")[["floor_db_q10", "rms_mg_q10", "quiet_rms_mg_median"]].agg(["median", "min", "max"]))

# Plot per-device floor distribution
fig, ax = plt.subplots(figsize=(8, 4))
for subject, color in [("Eric-Galaxy", "tab:blue"), ("Eric-iPhone", "tab:orange")]:
    dd = summary[summary.subject == subject]["floor_db_q10"]
    ax.hist(dd, bins=12, alpha=0.6, label=subject, color=color)
ax.set_xlabel("Floor, 10th percentile of 2-6 Hz in-band trace power (dB re g^2/Hz)")
ax.set_ylabel("Windows")
ax.set_title("Land device floor (daily window level)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "floor_db_histogram.png"), dpi=150)
plt.close(fig)

# Zoomed plots for one Galaxy and one Apple window (first each)
for subject, color in [("Eric-Galaxy", "tab:blue"), ("Eric-iPhone", "tab:orange")]:
    csv = windows_for(subject)[0]
    t, xyz, uni, fs, tu = read_window(csv)
    start = int(3*3600*FS_TARGET); end=start+int(120*FS_TARGET)
    sel = np.arange(start,end)
    tsel = tu[sel]
    mag_g = np.sqrt(np.sum(uni[sel]**2, axis=1))
    # millig values
    ax0 = None
    fig = plt.figure(figsize=(11, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(tsel - tsel[0], uni[sel]*1000.0)
    ax1.set_ylabel("axis (mg)")
    ax1.set_title(f"{subject} land wear - raw axes around a quiet period")
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(tsel - tsel[0], (mag_g-1.0)*1000.0, color="k")
    ax2.axhline(0, color="gray", linewidth=1)
    ax2.set_ylabel("|a|-1 (mg)")
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    # PSD of trace in zoom
    f0, p0 = [], []
    for axi in range(3):
        fr, pxx = welch(uni[sel, axi], fs=FS_TARGET, nperseg=min(512, len(sel)//4))
        if axi == 0:
            f0 = fr
            p0 = pxx.copy()
        else:
            p0 += pxx
    trace = p0/3.0
    ax3.semilogy(f0, trace)
    ax3.axvspan(BAND[0], BAND[1], alpha=0.2)
    ax3.set_xlim(0, 25)
    ax3.set_xlabel("Time(s)")
    ax3.set_ylabel("Trace PSD (g^2/Hz)")
    ax3.set_title("Zoom PSD (2-6 Hz band shaded)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"zoom_{subject}.png"), dpi=150)
    plt.close(fig)

# Combined floor-vs-mg summary figure
fig, ax = plt.subplots(figsize=(7, 4))
for subject, color in [("Eric-Galaxy", "tab:blue"), ("Eric-iPhone", "tab:orange")]:
    d = summary[summary.subject == subject]
    ax.scatter(d["quiet_move_mg_p50"], d["rms_mg_q10"], label=subject, color=color, alpha=0.7)
ax.set_xlabel("Median per-frame movement, std(|a|) in mg")
ax.set_ylabel("Estimated floor RMS in 2-6 Hz, mg")
ax.set_title("Land floor level as milli-g")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "floor_mg_scatter.png"), dpi=150)
plt.close(fig)
print("DONE", OUT)

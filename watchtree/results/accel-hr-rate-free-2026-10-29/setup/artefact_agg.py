"""Aggregate p1/*.pkl into artefact_*.csv (no ids; nights numbered within a cohort/nominal group) and figures/artefact_pulse_psd.png."""
import os, glob, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
N = os.environ["N"]
D = []
for p in sorted(glob.glob(f"{N}/p1/*/*.pkl")):
    D.append(pickle.load(open(p, "rb")))
rows, bands, est, meta = [], [], [], []
for i, d in enumerate(D):
    for key, lst in (("spec", rows), ("bands", bands), ("est", est)):
        for r in d[key]: lst.append(dict(r, _i=i))
    meta.append(dict(_i=i, **d["meta"]))
meta = pd.DataFrame(meta); meta["grp"] = meta.cohort.map({"sleepaccel": "SleepAccel", "wearanize": "Wearanize+"}) + " " + meta.nominal.astype(str) + " Hz"
meta["night"] = meta.groupby("grp").cumcount() + 1
g = meta.set_index("_i")
rows, bands, est = (pd.DataFrame(x) for x in (rows, bands, est))
for x in (rows, bands, est):
    x["group"] = x._i.map(g.grp); x["night"] = x._i.map(g.night)
mrow = {"truth": "truth", "a": "(a) to_25hz", "b1": "(b1) NUFFT PSD, raw timestamps", "b2": "(b2) least-squares reconstruction"}
rows["method"] = rows.method.map(mrow)
cols = ["group", "night", "kind", "f_hz", "method", "windows", "line_dB", "spur_band_total_dB", "spur_band_peak_dB", "spur_band_peak_hz", "spur_band_peak_dB_max", "spur_all_total_dB", "spur_all_peak_dB", "spur_all_peak_hz", "peak_at_image"]
rows[cols].round(2).to_csv(f"{N}/artefact_spectral_per_night.csv", index=False)
agg = rows.groupby(["group", "method", "f_hz"]).agg(nights=("night", "size"), line_dB_median=("line_dB", "median"), line_dB_worst=("line_dB", "min"),
      spur_band_total_dB_median=("spur_band_total_dB", "median"), spur_band_total_dB_worst=("spur_band_total_dB", "max"),
      spur_band_peak_dB_median=("spur_band_peak_dB", "median"), spur_all_peak_dB_median=("spur_all_peak_dB", "median"), spur_all_peak_hz_median=("spur_all_peak_hz", "median"),
      share_peak_at_predicted_image=("peak_at_image", "mean")).reset_index()
agg.round(2).to_csv(f"{N}/artefact_spectral_by_nominal.csv", index=False)
bb = bands.groupby(["group", "method", "band"]).ratio.median().unstack().reset_index(); bb.round(3).to_csv(f"{N}/artefact_noise_band_ratio.csv", index=False)
est["group"] = est._i.map(g.grp); est["arm"] = est.arm
es = est.groupby(["group", "signal", "arm"]).agg(nights=("night", "size"), peak_mae_median=("peak_mae", "median"), peak_within3_median=("peak_within3", "median"), track_mae_mean=("track_mae", "mean"), track_mae_worst=("track_mae", "max"), track_bias_mean=("track_bias", "mean")).reset_index()
es.round(2).to_csv(f"{N}/artefact_pulse_estimator.csv", index=False)
est[["group", "night", "signal", "arm", "windows", "peak_mae", "peak_within3", "track_mae", "track_bias"]].round(3).to_csv(f"{N}/artefact_pulse_estimator_per_night.csv", index=False)
# figure
sc = rows[(rows.method == mrow["a"]) & (rows.f_hz <= 8)].groupby("_i").spur_band_total_dB.median()
order = sc.sort_values().index; picks = {"median night by (a) spurious power in 0.67-8 Hz": order[len(order) // 2], "worst night by (a) spurious power": order[-1]}
low = meta.sort_values("effective_hz").iloc[0]._i; picks["lowest effective sampling rate"] = low
INK2, MUTED = "#5f5e58", "#9b9a93"; col = {"truth": "#1f1f1e", "a": "#eb6834", "b1": "#2a78d6", "b2": "#2a9d6f"}
lab = {"truth": "truth (exact 25 Hz grid)", "a": "(a) to_25hz", "b1": "(b1) NUFFT PSD, raw timestamps", "b2": "(b2) band-limited least-squares reconstruction"}
fig, axs = plt.subplots(3, 1, figsize=(10, 10), layout="constrained", sharex=True)
for ax, (t, i) in zip(axs, picks.items()):
    ex = D[i]["example"]
    if ex is None: continue
    for k in ("truth", "a", "b1", "b2"):
        ax.semilogy(ex["f"], np.maximum(ex[k], 1e-30), color=col[k], lw=1.2, label=lab[k])
    ax.set_title(f"{g.loc[i, 'grp']}: {t}", loc="left"); ax.set_ylabel("PSD (g²/Hz), x axis"); ax.axvspan(0.67, 8, color="#e4e3dd", alpha=0.35, lw=0); ax.set_ylim(1e-14, None)
axs[-1].set_xlabel("Frequency (Hz)"); axs[0].legend(loc="upper right", frameon=False); axs[0].set_xlim(0, 12.5)
fig.suptitle("Synthetic pulse, 60 bpm: 5.5 Hz burst once per beat plus 1 Hz; one 30 s window per night", x=0.01, ha="left")
os.makedirs(f"{N}/figures", exist_ok=True); fig.savefig(f"{N}/figures/artefact_pulse_psd.png", dpi=150)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
print(meta.groupby("grp").agg(nights=("night", "size"), eff=("effective_hz", "median")))

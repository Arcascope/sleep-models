"""Figure 1: section 14 per-window BPM statistics (5-min windows) by cohort with the accelerometer-derived HR series added
as its own curves (one per cohort, window length --window). Figure 2: accelerometer-derived minus device HR error by cohort.
Device curves are the section 14 audit's per-window tables (setup/audit, run bpm-audit-exp4-2026-10-28/local-v2)."""
import argparse, sys, zlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "audit"))
import audit_night as AN
from bpm_common import OUT, OLD_EXP

ap = argparse.ArgumentParser(); ap.add_argument("--window", type=int, required=True)
W = ap.parse_args().window
AUD = OLD_EXP.parent / "bpm-audit-exp4-2026-10-28" / "local-v2"
COH = ["wearanize", "sleepaccel", "weaver_garmin", "weaver_fitbit"]
LAB = {"wearanize": "Wearanize+", "sleepaccel": "SleepAccel", "weaver_garmin": "Weaver Garmin", "weaver_fitbit": "Weaver Fitbit"}
COL = {"wearanize": "#0072B2", "sleepaccel": "#D55E00", "weaver_garmin": "#009E73", "weaver_fitbit": "#CC79A7"}
WIN = 300.0

# accelerometer-derived series: the audit's per-window statistics on the 5 s estimate series
cache = OUT / "accelhr_window_stats.parquet"
if cache.exists():
    A = pd.read_parquet(cache)
else:
    rows = []
    for c in COH:
        for sp in sorted((OUT / "accelhr" / c / "series").glob("*.parquet")):
            s = pd.read_parquet(sp)
            ok = s[f"bpm{W}"].notna().to_numpy()
            tv = 30.0 + 5.0 * s.j.to_numpy()[ok]
            bv = s[f"bpm{W}"].to_numpy(float)[ok]
            span = 30.0 * ((len(s) - 1) // 6 + 1)
            nwin = int(span // WIN)
            if tv.size < 40 or nwin == 0:
                continue
            sp_ = AN.spectra(tv, bv, WIN, nwin, seed=zlib.crc32(sp.stem.encode()))
            wid = (tv // WIN).astype(int)
            st, en = np.searchsorted(wid, np.arange(nwin)), np.searchsorted(wid, np.arange(nwin), side="right")
            for w in range(nwin):
                tw, bw = tv[st[w]:en[w]], bv[st[w]:en[w]]
                n = tw.size
                cov = min(1.0, n * 5.0 / WIN)
                gap = np.diff(np.r_[w * WIN, tw, (w + 1) * WIN]).max() if n else WIN
                if cov < 0.8 or gap > 15.0 or n < 10:
                    continue
                d = np.diff(bw); cons = np.diff(tw) <= 7.5
                tri = cons[1:] & cons[:-1]
                a1, a2 = d[:-1][tri], d[1:][tri]
                acf = np.corrcoef(a1, a2)[0, 1] if a1.size > 10 and a1.std() > 0 and a2.std() > 0 else np.nan
                ratio = np.nan
                if sp_ is not None and np.isfinite(sp_["obs"][w]).all():
                    f = sp_["freqs"]
                    po = AN.band_power(f, sp_["obs"][[w]], 0.04, 0.15)[0]; ps = AN.band_power(f, sp_["shuf"][[w]], 0.04, 0.15)[0]
                    ratio = po / ps if ps > 0 else np.nan
                rows.append({"cohort": c, "acf1_diffs": acf, "ratio_mid": ratio})
    A = pd.DataFrame(rows); A.to_parquet(cache)

wins = {c: pd.read_parquet(AUD / f"windows-{c}.parquet") for c in COH}
nights = {c: pd.read_parquet(AUD / f"nights-{c}.parquet") for c in COH}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for c in COH:
    g = wins[c][(wins[c].win_min == 5) & wins[c].valid_data]
    for a, col in zip(ax, ("acf1_diffs", "ratio_mid", "ratio_hi")):
        if col == "ratio_hi" and nights[c].nominal_dt.median() > 2:
            continue
        x = np.sort(g[col].dropna().to_numpy())
        if x.size:
            a.plot(x, np.arange(1, x.size + 1) / x.size, color=COL[c], label=f"{LAB[c]} device HR (n={x.size})")
    h = A[A.cohort == c]
    for a, col in zip(ax[:2], ("acf1_diffs", "ratio_mid")):
        x = np.sort(h[col].dropna().to_numpy())
        if x.size:
            a.plot(x, np.arange(1, x.size + 1) / x.size, color=COL[c], ls="--", lw=1.4, label=f"{LAB[c]} accelerometer-derived (n={x.size})")
ax[0].set_xlabel("lag-1 autocorrelation of successive differences"); ax[0].axvline(-0.5, color="gray", ls=":")
ax[1].set_xlabel("power 0.04-0.15 Hz / same window shuffled"); ax[1].axvline(1, color="gray", ls=":"); ax[1].set_xscale("log"); ax[1].set_xlim(1e-5, 1e2)
ax[2].set_xlabel("power 0.10-0.40 Hz / same window shuffled (device HR only; 1 s streams)"); ax[2].axvline(1, color="gray", ls=":"); ax[2].set_xscale("log"); ax[2].set_xlim(1e-7, 1e1)
for a in ax:
    a.set_ylabel("share of 5-min windows"); a.grid(alpha=.3)
ax[0].legend(fontsize=6.5, loc="upper left")
fig.suptitle(f"Per-window BPM statistics, 5-min windows with >= 80% coverage; solid: device HR, dashed: accelerometer-derived ({W} s windows, 5 s steps); dotted: white noise", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "figures" / "bpm_window_statistics_with_accelerometer_hr.png", dpi=130)

# figure 2: error by cohort
E = pd.concat([pd.read_parquet(p) for p in [OUT / "accelhr_epochs_all.parquet"]])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for c in COH:
    g = E[E.cohort == c]
    e = (g[f"est{W}"] - g[f"dev{W}"]).dropna()
    x = np.sort(e.abs().to_numpy())
    ax[0].plot(x, np.arange(1, x.size + 1) / x.size, color=COL[c], label=f"{LAB[c]} all (n={x.size:,})")
    hi = g[g[f"q{W}"] >= g[f"q{W}"].quantile(0.9)]
    e2 = (hi[f"est{W}"] - hi[f"dev{W}"]).dropna()
    x2 = np.sort(e2.abs().to_numpy())
    ax[0].plot(x2, np.arange(1, x2.size + 1) / x2.size, color=COL[c], ls="--", label=f"{LAB[c]} top quality decile")
    qd = pd.qcut(g[f"q{W}"].rank(method="first"), 10, labels=False)
    m = (g[f"est{W}"] - g[f"dev{W}"]).abs().groupby(qd).mean()
    ax[1].plot(m.index + 1, m.values, "o-", color=COL[c], label=LAB[c])
ax[0].set_xlim(0, 80); ax[0].set_xlabel("absolute error, accelerometer-derived minus device HR (bpm)"); ax[0].set_ylabel("share of epochs")
ax[1].set_xlabel("peak-quality decile (within cohort)"); ax[1].set_ylabel("mean absolute error (bpm)")
for a in ax:
    a.grid(alpha=.3)
ax[0].legend(fontsize=6.5, loc="lower right"); ax[1].legend(fontsize=7)
fig.suptitle(f"Accelerometer-derived vs device HR, {W} s window ending at the epoch end, all scored epochs with both", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "figures" / "accelhr_error_by_cohort.png", dpi=130)

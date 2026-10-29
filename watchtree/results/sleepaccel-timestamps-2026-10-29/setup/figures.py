"""Figures and timestamps_summary.csv of the raw SleepAccel timestamps (private/<id>.npy from extract.py). Nothing is resampled or interpolated.
Nights are labelled 1..N by median dt; the id mapping stays in private/."""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
N = Path(os.environ["N"]); P = N / "private"; FIG = N / "figures"
INK, INK2, MUTED = "#1f1f1e", "#5f5e58", "#9b9a93"; BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2a9d6f"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlelocation": "left", "legend.frameon": False})
nights = pd.read_csv(P / "nights.csv", dtype={"id": str})
T = {r.id: np.load(P / f"{r.id}.npy") for r in nights.itertuples()}
med = {k: np.median(np.diff(t)) for k, t in T.items()}
order = sorted(T, key=lambda k: med[k])
lab = {k: i + 1 for i, k in enumerate(order)}
pd.DataFrame({"id": order, "label": range(1, len(order) + 1)}).to_csv(P / "label_map.csv", index=False)
infold = dict(zip(nights.id, nights.in_fold))
D = {lab[k]: np.diff(T[k]) for k in T}; TT = {lab[k]: T[k] for k in T}; FOLD = {lab[k]: bool(infold[k]) for k in T}
n = len(D)

def stats(i):
    d = D[i]; m = np.median(d); ok = d[d <= 1.0]; t = TT[i]
    eff = ok.size / ok.sum()   # (N-1)/(t_last-t_first) with gaps > 1 s excluded
    return dict(night=i, in_bidoze_test_fold=FOLD[i], n_samples=t.size, median_dt_ms=m * 1e3, p1_dt_ms=np.percentile(d, 1) * 1e3, p5_dt_ms=np.percentile(d, 5) * 1e3,
                p95_dt_ms=np.percentile(d, 95) * 1e3, p99_dt_ms=np.percentile(d, 99) * 1e3, nominal_rate_hz=round(1 / m), effective_rate_hz=eff,
                share_off_median_gt1pct=(np.abs(d / m - 1) > 0.01).mean(), gaps_gt_1s=int((d > 1).sum()), share_dt_le_0=(d <= 0).mean())
S = pd.DataFrame([stats(i) for i in range(1, n + 1)])
S.round(5).to_csv(N / "timestamps_summary.csv", index=False)
print(S.round(3).to_string(index=False))

def tag(ax, i):  # in-fold marker via title colour handled by callers
    pass

# 1 pooled
allp = np.concatenate([d[d > 0] for d in D.values()]); nonpos = sum((d <= 0).sum() for d in D.values())
bins = np.logspace(-3, 1, 161)
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
ax.hist(allp, bins=bins, color=BLUE, lw=0); ax.set_xscale("log"); ax.set_yscale("log")
for ms in (10, 20, 25, 30, 100):
    ax.axvline(ms / 1e3, color=ORANGE, lw=0.9, ls="--"); ax.text(ms / 1e3, 1.15, f"{ms} ms", transform=ax.get_xaxis_transform(), color=ORANGE, ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Δt between consecutive samples (s)"); ax.set_ylabel("Steps per bin (40 log bins per decade)")
ax.set_title(f"Pooled Δt, {n} SleepAccel nights, {allp.size:,} steps with Δt > 0 ({nonpos:,} steps with Δt ≤ 0 not shown)", pad=22)
ax.set_xlim(1e-3, 10); fig.savefig(FIG / "1_pooled_dt_hist.png", dpi=150); plt.close(fig)

# 2 per night
XMAX = 0.12; b2 = np.arange(0, XMAX + 5e-4, 1e-3)  # linear 1 ms bins on one shared 0–120 ms axis
cols = 6; rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(16, 2.35 * rows), sharex=True, sharey=True, layout="constrained")
for ax in axs.ravel(): ax.set_visible(False)
for i, ax in zip(range(1, n + 1), axs.ravel()):
    ax.set_visible(True); d = D[i]; m = np.median(d); s = S.iloc[i - 1]
    ax.hist(d[(d > 0) & (d <= XMAX)] * 1e3, bins=b2 * 1e3, color=BLUE if FOLD[i] else MUTED, lw=0); ax.set_yscale("log"); ax.set_xlim(0, XMAX * 1e3)
    ax.set_xticks(range(0, 121, 20)); ax.set_xticks(range(0, 121, 10), minor=True); ax.grid(True, axis="x", which="both", color=MUTED, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.axvline(m * 1e3, color=ORANGE, lw=0.8)
    ax.set_title(f"Night {i}{'' if FOLD[i] else ' (not in test folds)'}\nmedian {m*1e3:.1f} ms, {1/m:.1f} Hz, off {s.share_off_median_gt1pct:.1%}", fontsize=8.5, loc="left")
    ax.tick_params(labelsize=8)
for ax in axs.ravel(): ax.tick_params(labelbottom=True)
for ax in axs[-1]: ax.set_xlabel("Δt (ms)", fontsize=9)
fig.suptitle("Δt histogram per night, same axes on every panel (1 ms bins, 0–120 ms; log counts; orange = median; 'off' = share of steps > 1% from the median; grey = not in BiDoze test folds 0–4)", x=0.01, ha="left", fontsize=10, color=INK2)
fig.savefig(FIG / "2_dt_hist_per_night.png", dpi=150); plt.close(fig)

# 3 nominal vs effective
fig, ax = plt.subplots(figsize=(8, 9), layout="constrained")
y = np.arange(1, n + 1)
for i in y:
    a, b = S.nominal_rate_hz[i - 1], S.effective_rate_hz[i - 1]; ax.plot([a, b], [i, i], color=MUTED, lw=1, zorder=1)
ax.scatter(S.nominal_rate_hz, y, s=34, color=BLUE, zorder=2, label="nominal = round(1 / median Δt)")
ax.scatter(S.effective_rate_hz, y, s=34, facecolor="white", edgecolor=ORANGE, lw=1.6, zorder=3, label="effective = (N − 1) / (t_last − t_first), gaps > 1 s excluded")
ax.set_yticks(y); ax.set_yticklabels([f"{i}" + ("" if FOLD[i] else " *") for i in y], fontsize=8.5); ax.invert_yaxis()
ax.set_xlabel("Rate (Hz)"); ax.set_ylabel("Night (ordered by median Δt; * = not in BiDoze test folds)"); ax.legend(loc="lower right")
ax.set_title("Nominal vs effective sampling rate per night", pad=8); ax.grid(axis="x", color="#e4e3dd", lw=0.6)
fig.savefig(FIG / "3_nominal_vs_effective_rate.png", dpi=150); plt.close(fig)

# 4 rate over the night
fig, axs = plt.subplots(rows, cols, figsize=(16, 2.35 * rows), sharex=True, layout="constrained")
for ax in axs.ravel(): ax.set_visible(False)
blocks = {}
for i, ax in zip(range(1, n + 1), axs.ravel()):
    ax.set_visible(True); t = TT[i]; d = np.diff(t); tm = t[1:]; blk = np.floor((tm - t[0]) / 300).astype(int)
    xs, ys = [], []
    for b in np.unique(blk):
        dd = d[blk == b]
        if dd.size >= 10: xs.append((b + 0.5) * 5 / 60); ys.append(1 / np.median(dd))
    blocks[i] = (np.array(xs), np.array(ys))
    ax.plot(xs, ys, color=BLUE if FOLD[i] else MUTED, lw=1, marker=".", ms=2)
    ax.set_title(f"Night {i}: {np.nanmin(ys):.1f}–{np.nanmax(ys):.1f} Hz", fontsize=8.5, loc="left"); ax.tick_params(labelsize=8)
    lo, hi = np.min(ys), np.max(ys); pad = max(0.5, 0.15 * (hi - lo)); ax.set_ylim(lo - pad, hi + pad)
for ax in axs[-1]: ax.set_xlabel("Hours since start", fontsize=9)
for ax in axs[:, 0]: ax.set_ylabel("1 / median Δt (Hz)", fontsize=9)
fig.suptitle("Rate over the night: 1 / median Δt per 5-min block (display only; each panel has its own y range; grey = not in BiDoze test folds)", x=0.01, ha="left", fontsize=10, color=INK2)
fig.savefig(FIG / "4_rate_over_night.png", dpi=150); plt.close(fig)
pd.DataFrame([dict(night=i, blocks=len(blocks[i][0]), min_block_rate_hz=blocks[i][1].min(), max_block_rate_hz=blocks[i][1].max(), n_distinct_rounded_rates=len(set(np.round(blocks[i][1]))))
              for i in blocks]).round(3).to_csv(N / "rate_over_night_ranges.csv", index=False)

# 5 lag coverage
off = S.share_off_median_gt1pct.to_numpy(); o = np.argsort(off, kind="stable")
picks = {"most regular": int(o[0]) + 1, "median": int(o[n // 2]) + 1, "least regular": int(o[-1]) + 1}
SIG = (2, 5, 10, 20, 50); TAUS = (1 / 3, 1.0); W = 30.0; LO, HI = 0.3, 1.6
edges = np.arange(LO, HI + 1e-9, 1e-3); ctr = (edges[:-1] + edges[1:]) / 2
rng = np.random.default_rng(0); NW = 200
res, tab = {}, []
for name, i in picks.items():
    t = TT[i]; m = np.median(D[i]); w0 = np.arange(t[0], t[-1] - W, W)
    cand = []
    for a in w0:
        lo, hi = np.searchsorted(t, [a, a + W])
        if hi - lo >= 0.5 * W / m and np.diff(t[lo:hi]).max() <= 1.0: cand.append((lo, hi))   # full windows: no gap > 1 s
    sel = [cand[j] for j in rng.choice(len(cand), min(NW, len(cand)), replace=False)]
    H = np.zeros((len(sel), ctr.size))
    for k, (lo, hi) in enumerate(sel):
        tw = t[lo:hi]; j1 = np.searchsorted(tw, tw + LO, "left"); j2 = np.searchsorted(tw, tw + HI, "right")
        cnt = j2 - j1; idx = np.repeat(np.arange(tw.size), cnt); pos = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(j1, cnt)
        H[k] = np.histogram(tw[pos] - tw[idx], bins=edges)[0]
    res[name] = (i, H, len(cand))
    for tau in TAUS:
        for s in SIG:
            K = np.exp(-0.5 * ((ctr - tau) / (s / 1e3)) ** 2); tab.append(dict(representative=name, night=i, tau_s=round(tau, 3), sigma_ms=s, windows_used=len(sel), median_effective_pairs=np.median(H @ K)))
tab = pd.DataFrame(tab); tab.round({"median_effective_pairs": 1, "tau_s": 4}).to_csv(N / "lag_effective_pairs.csv", index=False)
print(picks); print(tab.round(1).to_string(index=False))
fig, axs = plt.subplots(3, 3, figsize=(16, 10), layout="constrained", gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
kc = ["#9ecae1", "#6baed6", "#3182bd", "#e6550d", "#a63603"]
for r, (name, (i, H, nc)) in enumerate(res.items()):
    mean = H.mean(0); m = np.median(D[i])
    ax = axs[r, 0]; ax.bar(ctr, mean, width=1e-3, color=BLUE, lw=0)
    for tau in TAUS: ax.axvline(tau, color=ORANGE, lw=0.8, ls="--")
    ax.set_xlim(LO, HI); ax.set_ylabel("Pairs per 1 ms bin\n(mean over windows)")
    ax.set_title(f"{name}: night {i} (median Δt {m*1e3:.1f} ms, off-median {off[i-1]:.1%}); {len(H)} of {nc} full 30 s windows", fontsize=9.5)
    for c, tau in zip((1, 2), TAUS):
        ax = axs[r, c]; msk = np.abs(ctr - tau) <= 0.1
        ax.bar(ctr[msk], mean[msk], width=1e-3, color=BLUE, lw=0); top = mean[msk].max()
        for s, col in zip(SIG, kc):
            ax.plot(ctr[msk], top * np.exp(-0.5 * ((ctr[msk] - tau) / (s / 1e3)) ** 2), color=col, lw=1.4, label=f"σ = {s} ms")
        ax.axvline(tau, color=ORANGE, lw=0.8, ls="--"); ax.set_xlim(tau - 0.1, tau + 0.1); ax.set_ylim(0, top * 1.08)
        ax.set_title(f"τ = {tau:.3f} s ({60/tau:.0f} bpm), kernels scaled to tallest bin", fontsize=9)
for ax in axs[-1]: ax.set_xlabel("Pair lag τ = t_j − t_i (s)")
axs[0, 1].legend(loc="upper right", fontsize=8, ncols=2)
fig.savefig(FIG / "5_lag_coverage.png", dpi=150); plt.close(fig)

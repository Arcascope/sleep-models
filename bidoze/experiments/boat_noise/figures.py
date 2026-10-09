"""Figures for the boat-noise zoology, from ``survey.py``'s spectra and the raw accelerometer.

- ``overview``: one whole night: the magnitude spectrogram (0 Hz to Nyquist, and 0–2 Hz), the
  jerk spectrogram (the dashed line is the model's 6 Hz limit), the range of ENMO samples per 2 s, and the PSG
  hypnogram.
- ``zoom``: one stretch of a night, ``minutes`` long around ``at``:
  - the magnitude spectrogram in the model's 10-s windows (0.1 Hz bins);
  - the magnitude and jerk spectrograms in 30-s windows at a 2-s step (0.033 Hz bins), from the
    raw samples, which resolve narrow and drifting lines; a transient shows about 30 s wide;
  - the raw ENMO and jerk at full rate over ``seconds`` around ``at`` (shaded above);
  - beside each spectrogram, the PSD (dB) of its single window centred on ``at``, on the same
    frequency axis.

No panel combines spectra across time windows.

Colour limits are the 1st and 99th percentiles of the panel's own values; nothing is normalized
per frequency or over the night.

Usage:
  python figures.py overview <out dir> [<cohort>_<id> ...]
  python figures.py zoom <data root> <out dir> <cohort>_<id> <at hours> <minutes> <seconds> <name> [fmax]
  python figures.py zoom-averaged ...   (the same, with the superseded time-averaged method, into zoo-time-averaged/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import senpy
from survey import load_accel

STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "REM"}
STAGE_ROW = {0: 4, 1: 3, 2: 2, 3: 1, 4: 3.5, 5: 3.5}
MODEL_FMAX = 6.0
LOW_FMAX = 2.0
WINDOW_S = 10.0
HIRES_S = 30.0
HIRES_STEP_S = 2.0
# The superseded time-averaged method, kept to reproduce its figures for comparison: 60-s
# windows at a 5-s step, each row the running mean of 30 s of windows, and the PSD panel the
# median over 5 minutes. Combining windows across time blurs this noise; do not use it for analysis.
AVERAGED = {"hires_s": 60.0, "step_s": 5.0, "smooth_s": 30.0, "psd_span_s": 300.0}


def running_mean(sxx: np.ndarray, n: int) -> np.ndarray:
    kernel = np.ones(n) / n
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, np.nan_to_num(sxx))


def load(out: Path, key: str) -> dict:
    return dict(np.load(out / "spectra" / f"{key}.npz"))


def show(ax, t_h: np.ndarray, f: np.ndarray, s: np.ndarray, label: str) -> None:
    s = s.astype(float)
    lo, hi = np.nanpercentile(s, [1, 99])
    ax.pcolormesh(t_h, f, s.T, vmin=lo, vmax=hi, cmap="magma", shading="nearest", rasterized=True)
    if f[-1] > MODEL_FMAX:
        ax.axhline(MODEL_FMAX, color="w", lw=0.6, ls="--")
    ax.set_ylim(f[0], f[-1])
    ax.set_ylabel(f"{label}\nHz")


def side_psd(fig, cell, spec_ax, f: np.ndarray, power: np.ndarray, colour: str, title: str) -> None:
    """A PSD beside its spectrogram, frequency on the shared vertical axis, power in dB.

    The power axis starts at the 5th percentile, so the nulls of a single periodogram do not
    stretch it; peaks are what the panel is for.
    """
    ax = fig.add_subplot(cell, sharey=spec_ax)
    db = 10 * np.log10(power + 1e-20)
    ax.fill_betweenx(f, np.nanpercentile(db, 5), db, color=colour, alpha=0.35, lw=0)
    ax.plot(db, f, color=colour, lw=0.6)
    ax.set_xlim(np.nanpercentile(db, 5), np.nanmax(db) + 2)
    ax.set_xlabel("dB")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelleft=False)
    if f[-1] > MODEL_FMAX:
        ax.axhline(MODEL_FMAX, color="0.5", ls="--", lw=0.6)


def hypnogram(ax, d: dict) -> None:
    keep = d["psg"] >= 0
    t = d["psg_t"][keep] / 3600
    ax.step(t, [STAGE_ROW[int(v)] for v in d["psg"][keep]], where="post", color="k", lw=0.8)
    ax.set_yticks([1, 2, 3, 3.5, 4], ["N3", "N2", "N1", "REM", "W"])
    ax.set_ylim(0.5, 4.5)


def overview(out: Path, keys: list[str]) -> None:
    (out / "overview").mkdir(exist_ok=True)
    for key in keys or sorted(p.stem for p in (out / "spectra").glob("*.npz")):
        d = load(out, key)
        fig, ax = plt.subplots(5, 1, figsize=(16, 11), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2, 3, 1.2, 1]})
        show(ax[0], d["mag_t"] / 3600, d["f"], d["mag"], "|a| log PSD")
        low = d["f"] <= LOW_FMAX
        show(ax[1], d["mag_t"] / 3600, d["f"][low], d["mag"][:, low], f"|a| log PSD\n0–{LOW_FMAX:g}")
        show(ax[2], d["jerk_t"] / 3600, d["fj"], d["jerk"], "jerk log PSD")
        ax[3].fill_between(d["enmo_t"] / 3600, np.maximum(d["enmo_min"], 1e-4), np.maximum(d["enmo_max"], 1e-4),
                           color="k", lw=0, step="post")
        ax[3].set_yscale("log")
        ax[3].set_ylabel("ENMO (g),\nmin–max per 2 s")
        hypnogram(ax[4], d)
        ax[4].set_xlabel("hours from recording start")
        ax[0].set_title(f"{key}  (fs {float(d['fs']):.1f} Hz)")
        fig.tight_layout()
        fig.savefig(out / "overview" / f"{key}.png", dpi=90)
        plt.close(fig)
        print(key, flush=True)


def zoom(root: Path, out: Path, key: str, at_h: float, minutes: float, seconds: float, name: str,
         fmax: float | None = None, averaged: bool = False) -> None:
    hires_s, step_s = (AVERAGED["hires_s"], AVERAGED["step_s"]) if averaged else (HIRES_S, HIRES_STEP_S)
    d = load(out, key)
    cohort, stem = key.split("_", 1)
    t, x, y, z = load_accel(root, cohort, stem).T
    at = at_h * 3600
    lo, hi = at - minutes * 30, at + minutes * 30
    zlo, zhi = at - seconds / 2, at + seconds / 2

    stretch = (t >= lo - hires_s) & (t <= hi + hires_s)
    ts = t[stretch]
    mag_s = senpy.compute_magnitude(x[stretch], y[stretch], z[stretch])
    js = senpy.compute_jerk(ts, x[stretch], y[stretch], z[stretch])
    nyquist = float(d["fs"]) / 2
    top = min(fmax or np.inf, nyquist)

    fig = plt.figure(figsize=(16, 19))
    grid = fig.add_gridspec(5, 2, width_ratios=[4, 1], height_ratios=[2.4, 4, 4, 1.4, 1.4])
    ax = fig.add_subplot(grid[0, 0])
    m = (d["mag_t"] >= lo) & (d["mag_t"] <= hi)
    fm = d["f"] <= top
    show(ax, d["mag_t"][m] / 60, d["f"][fm], d["mag"][m][:, fm], f"|a|, {WINDOW_S:g}-s\nwindows (model)")
    ax.axvspan(zlo / 60, zhi / 60, color="c", alpha=0.25)
    row = int(np.nanargmin(np.abs(d["mag_t"] - at)))
    side_psd(fig, grid[0, 1], ax, d["f"][fm], 10.0 ** d["mag"][row, fm].astype(float), "tab:blue",
             f"|a|, the {WINDOW_S:g}-s window at the zoom")
    for row, (sig, tsig, label) in enumerate(((mag_s, ts, "|a|"), (np.asarray(js.jerk), np.asarray(js.timestamps_s), "jerk")), 1):
        sp = senpy.compute_nufft_spectrogram(tsig, sig, hires_s, hires_s - step_s, kind="psd",
                                             empty_windows="keep", origin_s=0.0)
        tt = sp.times + (sp.origin_s or 0.0)
        keep_t = (tt >= lo) & (tt <= hi)
        keep_f = sp.frequencies <= top
        psd = running_mean(sp.Sxx, int(round(AVERAGED["smooth_s"] / step_s))) if averaged else sp.Sxx
        ax = fig.add_subplot(grid[row, 0])
        show(ax, tt[keep_t] / 60, sp.frequencies[keep_f], np.log10(psd[keep_t][:, keep_f] + 1e-12),
             f"{label}, {hires_s:g}-s\nwindows" + (", 30-s\nrunning mean" if averaged else ""))
        ax.axvspan(zlo / 60, zhi / 60, color="c", alpha=0.25)
        colour = "tab:blue" if label == "|a|" else "tab:orange"
        if averaged:
            near = np.abs(tt - at) <= AVERAGED["psd_span_s"] / 2
            side_psd(fig, grid[row, 1], ax, sp.frequencies[keep_f], np.nanmedian(psd[near], axis=0)[keep_f], colour,
                     f"{label}, median over 5 min\n(time-averaged)")
        else:
            near = int(np.nanargmin(np.abs(tt - at)))
            side_psd(fig, grid[row, 1], ax, sp.frequencies[keep_f], psd[near, keep_f], colour,
                     f"{label}, the {hires_s:g}-s window at the zoom")
    ax.set_xlabel("minutes from recording start")
    keep = (d["psg_t"] >= lo - 30) & (d["psg_t"] <= hi)
    stages = ", ".join(f"{STAGES.get(int(v), '?')}" for v in d["psg"][keep] if v >= 0)
    fig.suptitle(f"{key}: {at_h:.3f} h, {minutes:g} min stretch, {seconds:g} s zoom (cyan)   PSG: {stages[:120]}",
                 fontsize=10)

    m = (t >= zlo) & (t <= zhi)
    mag = senpy.compute_magnitude(x[m], y[m], z[m])
    j = senpy.compute_jerk(t[m], x[m], y[m], z[m])
    ax = fig.add_subplot(grid[3, 0])
    ax.plot(t[m] - zlo, np.maximum(mag - 1, 0), lw=0.6, color="k")
    ax.plot(t[m] - zlo, mag - 1, lw=0.4, color="0.6", zorder=0)
    ax.set_ylabel("ENMO (black)\n||a||−1 (grey), g")
    ax = fig.add_subplot(grid[4, 0])
    ax.plot(np.asarray(j.timestamps_s) - zlo, j.jerk, lw=0.5, color="tab:blue")
    ax.set_ylabel("jerk")
    ax.set_xlabel(f"seconds from {zlo / 3600:.4f} h")

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    folder = out / ("zoo-time-averaged" if averaged else "zoo")
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "overview":
        overview(Path(sys.argv[2]), sys.argv[3:])
    else:
        args = sys.argv[2:]
        zoom(Path(args[0]), Path(args[1]), args[2], float(args[3]), float(args[4]), float(args[5]), args[6],
             float(args[7]) if len(args) > 7 else None, averaged=sys.argv[1] == "zoom-averaged")

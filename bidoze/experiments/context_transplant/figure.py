#!/usr/bin/env python3
"""Figure for research log 2026-10-16 §3: dose response of target sleep log odds, and the
per-person foundry − random contrast at the largest dose.

    python figure.py --run <run dir> --out <png>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"foundry": "#2a78d6", "random": "#eb6834", "release": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DOSES = ["60min", "6h", "all"]
DOSE_LABELS = ["60 min", "6 h", "all distant\n(13.3–20.7 h)"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summ = pd.read_csv(args.run / "analysis/summary.csv")
    avc = pd.read_csv(args.run / "analysis/arm_vs_control.csv")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    s = summ[(summ.seed == "ens") & (summ.cls == "sleep") & (summ.stratum == "all")]
    x = range(len(DOSES))
    for model, color in COLORS.items():
        m = s[s.model == model].set_index("dose").loc[DOSES]
        a.plot(x, m["wake_vs_orig"], "-", color=color, lw=2, marker="o", ms=6, label=model)
        a.plot(x, m["sleep_vs_orig"], "--", color=color, lw=2, marker="o", ms=6)
    a.axhline(0, color=MUTED, lw=1)
    a.text(2.1, 0.62, "wake carrier\n(solid)", va="center", fontsize=9, color=MUTED)
    a.text(2.1, -1.55, "sleep carrier\n(dashed)", va="center", fontsize=9, color=MUTED)
    a.legend(frameon=False, loc="lower left", title="model", fontsize=9, title_fontsize=9)
    a.set_xticks(list(x), DOSE_LABELS)
    a.set_xlim(-0.2, 2.7)
    a.set_ylabel("Change in target sleep log odds\nvs unmodified input")
    a.set_title("PSG sleep epochs of 326 targets; mean over 13 people", fontsize=10, loc="left")
    a.grid(axis="y", color=GRID, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)

    d = avc[(avc["pair"] == "foundry - random") & (avc.seed == "ens") & (avc.dose == "all")
            & (avc.cls == "sleep") & (avc.stratum == "all")].sort_values("contrast")
    b.axvline(0, color=MUTED, lw=1)
    b.axvline(-0.25, color=MUTED, lw=1, ls=":")
    b.annotate("gate:\n≤ −0.25", (-0.25, 0), xytext=(-4, 0), textcoords="offset points",
               ha="right", va="center", fontsize=8.5, color=MUTED)
    b.set_xlim(-0.55, 1.3)
    b.scatter(d.contrast, range(len(d)), s=40, color=COLORS["foundry"], edgecolor="white", lw=1.5, zorder=3)
    b.set_yticks(range(len(d)), d.recording)
    b.set_xlabel("foundry − random: (wake − sleep carrier) log odds")
    b.set_title("All distant context, per person", fontsize=10, loc="left")
    b.grid(axis="x", color=GRID, lw=0.8)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor="white")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

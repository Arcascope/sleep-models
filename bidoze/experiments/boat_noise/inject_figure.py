"""Figure of the injection scan: change in wake specificity against the clean nights.

Left: every component at 10 mg RMS, mean paired change over nights with its 95% t-interval.
Right: the white-noise floor against its RMS amplitude. The dashed line is the gap between
these nights clean and SURF.

Usage: python inject_figure.py <inject dir> <SURF wake specificity> <out png>
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

LABELS = {"line_0.88": "line 0.88 Hz", "line_1.4": "line 1.4 Hz", "wander_3.85": "wandering line 3.85 Hz",
          "wobble_1.8": "wobbling line 1.8 Hz", "comb_wobble": "wobbling comb 1.2 Hz", "knock_1.05": "knock comb 1.05 s",
          "jerk_lines": "lines across gravity 1.08/2.15 Hz", "hull_0.4": "hull band 0.3–0.5 Hz",
          "line_8.6": "line 8.6 Hz", "line_24": "line 24 Hz", "white_up": "white noise, along gravity",
          "white_iso": "white noise, every axis"}
COLOURS = {"white_iso": "#0173b2", "white_up": "#de8f05"}


def paired(d: pd.DataFrame, clean: pd.Series) -> pd.DataFrame:
    rows = []
    for condition, g in d.groupby("condition"):
        z = (g.set_index("night").wake_specificity - clean).dropna()
        half = stats.t.ppf(0.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
        name, gain = condition.split("@")
        rows.append({"component": name, "gain": float(gain.removesuffix("mg")), "diff": z.mean(),
                     "lo": z.mean() - half, "hi": z.mean() + half})
    return pd.DataFrame(rows)


def main(inject: Path, surf: float, out: Path) -> None:
    gap = pd.read_csv(inject / "gap_share.csv", index_col=0).gap_share
    d = pd.read_csv(inject / "nights.csv")
    d = d[d.night.isin(gap[gap < 0.1].index)]
    clean = d[d.condition == "clean"].set_index("night").wake_specificity
    t = paired(d[d.condition != "clean"], clean)
    target = surf - clean.mean()
    sns.set_theme(style="whitegrid", context="notebook")
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    at10 = t[t.gain == 10].sort_values("diff", ascending=False)
    y = np.arange(len(at10))
    colours = [COLOURS.get(c, "0.35") for c in at10.component]
    for k, (_, r) in enumerate(at10.iterrows()):
        a.errorbar(r["diff"], k, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="none", ecolor=colours[k],
                   elinewidth=1.6, capsize=3)
    a.scatter(at10["diff"], y, c=colours, s=48, zorder=3)
    a.set_yticks(y, [LABELS[c] for c in at10.component])
    a.axvline(0, color="0.3", lw=0.8)
    a.axvline(target, color="0.3", lw=1, ls="--")
    a.text(target, len(at10) - 1.6, f" SURF\n ({surf:.1f}%)", fontsize=9, va="bottom")
    a.set_xlabel("Change in wake specificity (points), 10 mg RMS")
    a.set_title("Each component alone, whole night", fontsize=11)
    for c in ("white_iso", "white_up"):
        w = t[t.component == c].sort_values("gain")
        b.errorbar(w.gain, w["diff"], yerr=[w["diff"] - w.lo, w.hi - w["diff"]], color=COLOURS[c], marker="o",
                   lw=2, capsize=3, label=LABELS[c])
    b.axhline(target, color="0.3", lw=1, ls="--")
    b.axhline(0, color="0.3", lw=0.8)
    b.set_xscale("log")
    b.set_xticks([2, 5, 10, 20], ["2", "5", "10", "20"])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.text(2, target, f"SURF ({surf:.1f}%)", fontsize=9, va="bottom")
    b.set_xlabel("RMS amplitude (mg)")
    b.set_ylabel("Change in wake specificity (points)")
    b.set_title("Raised floor", fontsize=11)
    b.legend(frameon=False, loc="lower left")
    sns.despine(fig, left=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]))

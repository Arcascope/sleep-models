"""Pilot: each arm minus independent ×3, per draw, on the test adults of both cohorts.

Reads <run root>/analysis/draws.csv (written by evaluate.py). One panel per cohort and metric; one
row per arm; a dot per draw (hollow when the fit was time-capped), the grey band is the spread
(max − min) of the 3 independent seeds around ×3, and the dashed lines are the drop margins
(−0.02 κ, −3 pts wake specificity). Output: tridoze-pilot-diffs.png.

Usage: python pilot_diffs.py <run root> [--capped arm:draw ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from known_style import AQUA, BLUE, GREY, GRID, INK, MUTED, ORANGE, VIOLET, plt, save  # noqa: E402

ARMS = [("tridoze_j", "TriDoze-J", BLUE), ("tridoze_r", "TriDoze-R", ORANGE),
        ("bidoze_xl", "BiDoze-XL", AQUA), ("bidoze_g", "BiDoze-G (1 network)", VIOLET)]
METRICS = [("kappa_4stage", "κ 4-stage", 0.02), ("kappa_sleep_wake", "κ sleep/wake", 0.02),
           ("wake_spec_at_95_sens", "wake spec at 95% sleep sens (pts)", 3.0)]
COHORTS = ["SleepAccel", "SleepAccel-Clinical"]
REF = "independent ×3"
SEEDS = [f"independent seed {s}" for s in range(3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--capped", nargs="*", default=[])
    ap.add_argument("--arm-dir", nargs="*", default=[], help="arm=label to read an arm from another row name")
    a = ap.parse_args()
    capped = {tuple(c.split(":")) for c in a.capped}
    rename = dict(x.split("=") for x in a.arm_dir)
    d = pd.read_csv(a.run_root / "analysis/draws.csv")
    d["model"] = d["model"].replace(rename)
    p = d.pivot_table(index=["cohort", "metric", "draw"], columns="model", values="mean")

    fig, axes = plt.subplots(len(METRICS), len(COHORTS), figsize=(9, 7.2), sharey=True)
    for i, (metric, label, margin) in enumerate(METRICS):
        for j, cohort in enumerate(COHORTS):
            ax = axes[i, j]
            sub = p.loc[(cohort, metric)]
            spread = (sub[SEEDS].max(axis=1) - sub[SEEDS].min(axis=1)).mean()
            ax.axvspan(-spread / 2, spread / 2, color=GRID, zorder=0)
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.axvline(-margin, color=MUTED, lw=0.8, ls="--")
            for k, (arm, name, colour) in enumerate(ARMS):
                if arm not in sub:
                    continue
                diff = (sub[arm] - sub[REF]).dropna()
                for draw, v in diff.items():
                    hollow = (arm, str(draw)) in capped
                    ax.scatter(v, k, s=34, color="white" if hollow else colour, edgecolor=colour, lw=1.4, zorder=3)
                if len(diff):
                    ax.scatter(diff.mean(), k, marker="|", s=260, color=INK, lw=1.6, zorder=4)
            ax.set_yticks(range(len(ARMS)), [n for _, n, _ in ARMS])
            ax.invert_yaxis()
            if i == 0:
                ax.set_title(cohort, loc="left", fontsize=10)
            ax.set_xlabel(f"{label}, minus independent ×3", fontsize=8.5, color=MUTED)
            ax.tick_params(axis="y", length=0)
    fig.text(0.01, -0.02, "Dot: one draw (hollow: fit stopped by the 15-min burst cap). Black bar: mean over draws. "
             "Grey band: mean spread of the 3 independent seeds.\nDashed line: drop margin. Pilot: 1 seed per draw.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "tridoze-pilot-diffs.png")


if __name__ == "__main__":
    main()

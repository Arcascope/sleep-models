"""Ensemble gain over members and per-draw scores, independent x3 vs joint x3."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from known_style import ANALYSIS, BLUE, GREY, INK, MUTED, ORANGE, GRID, save

div = pd.read_csv(ANALYSIS / "diversity.csv")
dr = pd.read_csv(ANALYSIS / "draws.csv")
cohorts = ["SleepAccel", "SleepAccel-Clinical"]
draws = range(5)

fig = plt.figure(figsize=(9, 8))
gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.55, height_ratios=[1, 1.15])

# (a) gain of the ensemble's 4-stage kappa over the members' mean kappa
for i, coh in enumerate(cohorts):
    ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
    for off, ens, col in [(-0.2, "independent ×3", BLUE), (0.2, "joint ×3", ORANGE)]:
        v = div[(div.ensemble == ens) & (div.cohort == coh)].set_index("draw").kappa_4stage_gain
        ax.bar([d + off for d in draws], [v[d] for d in draws], width=0.36, color=col, edgecolor="white")
    ax.set_xticks(list(draws), [f"draw {d}" for d in draws])
    ax.set_ylim(0, 0.19)
    ax.set_ylabel("ensemble κ − members' mean κ")
    ax.set_title(coh, fontsize=10, loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    if i == 0:
        ax.legend(handles=[Line2D([], [], marker="s", ls="", color=BLUE, label="independent ×3"),
                           Line2D([], [], marker="s", ls="", color=ORANGE, label="joint ×3")],
                  frameon=False, loc="upper left", fontsize=8, ncol=2)
fig.text(0.02, 0.93, "(a) 4-stage κ gain", fontsize=11, fontweight="bold")

# (b) paired dots, independent x3 -> joint x3, per draw
k = 0
for metric, mlabel in [("kappa_4stage", "4-stage κ"), ("kappa_sleep_wake", "sleep/wake κ")]:
    for coh in cohorts:
        ax = fig.add_subplot(gs[1, k]); k += 1
        t = dr[(dr.metric == metric) & (dr.cohort == coh)].pivot(index="draw", columns="model", values="mean")
        for d in draws:
            a, b = t.loc[d, "independent ×3"], t.loc[d, "joint ×3"]
            hl = d == 1
            ax.plot([0, 1], [a, b], color=ORANGE if hl else GREY, lw=2 if hl else 1.2, zorder=2)
            ax.scatter([0, 1], [a, b], s=32, color=ORANGE if hl else BLUE, edgecolor="white", linewidth=1, zorder=3)
            if hl:
                ax.text(1.06, b, "draw 1", fontsize=8, va="center", color=INK)
        ax.set_xticks([0, 1], ["independent\n×3", "joint ×3"])
        ax.set_xlim(-0.3, 1.6)
        ax.set_title(f"{mlabel}\n{coh}", fontsize=8.5, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.text(0.02, 0.50, "(b) κ per draw, independent ×3 vs joint ×3 (lines join the same draw)", fontsize=11,
         fontweight="bold")
fig.subplots_adjust(top=0.9, bottom=0.08, left=0.08, right=0.97)
save(fig, "tridoze-known-gain.png")

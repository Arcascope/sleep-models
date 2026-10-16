"""Late-context shift test of the three released seeds (2026-10-15 section 2), from position.csv."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from known_style import BLUE, GRID, INK, MUTED, ORANGE, POSITION, save

p = pd.read_csv(POSITION)
g = p.groupby(["cohort", "placement", "seed"])[["sleep_called_sleep", "sleep_gap_argmax"]].mean() * 100
groups = [("SleepAccel", "SleepAccel", "as cached", "shifted"),
          ("SleepAccel-Clinical", "SleepAccel-Clinical", "as cached", "shifted"),
          ("FoundryApple", "Foundry", "cut", "full 24 h")]
rows = []
for coh, name, e, l in groups:
    for s in range(3):
        rows.append((coh, name, e, l, s))

fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
for ax, col, title, xmax, tx, tcol in [(axes[0], "sleep_called_sleep", "PSG sleep called sleep", 140, [0, 25, 50, 75, 100], 106),
                       (axes[1], "sleep_gap_argmax", "PSG sleep called gap", 85, [0, 10, 20, 30, 40, 50], 56)]:
    for i, (coh, name, e, l, s) in enumerate(rows):
        y = i + (i // 3) * 0.7
        a, b = g.loc[(coh, "early", s), col], g.loc[(coh, "late", s), col]
        ax.plot([a, b], [y, y], color="#c3c8ce", lw=2.2, zorder=1)
        ax.scatter([a], [y], s=42, color=BLUE, edgecolor="white", linewidth=1, zorder=3)
        ax.scatter([b], [y], s=42, color=ORANGE, edgecolor="white", linewidth=1, zorder=3)
        ax.text(tcol, y, f"{a:.1f}% → {b:.1f}%", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(-2, xmax)
    ax.set_xticks(tx, [f"{v}%" for v in tx])
    ax.set_title(title, fontsize=10, loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ys = [i + (i // 3) * 0.7 for i in range(9)]
axes[0].set_yticks(ys, [f"seed {r[4]}" for r in rows])
axes[0].set_ylim(max(ys) + 0.7, -0.7)
for gi, (coh, name, e, l) in enumerate(groups):
    axes[0].text(-0.3, ys[gi * 3 + 1], f"{name}\n({e} → {l})", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="center", fontsize=8.5, fontweight="bold")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="night at the start of the context (as cached / cut)"),
                    Line2D([], [], marker="o", ls="", color=ORANGE, label="night late in the context (shifted / full 24 h)")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.02), fontsize=8.5)
fig.subplots_adjust(left=0.33, wspace=0.12, top=0.88)
save(fig, "tridoze-known-shift.png")

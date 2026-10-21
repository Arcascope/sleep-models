"""What each network inside an ensemble calls: stacked call shares per member, SleepAccel, 5 draws."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from known_style import ANALYSIS, GREY, INK, MUTED, ORANGE, STAGE_COLORS, save

m = pd.read_csv(ANALYSIS / "members.csv")
m = m[m.cohort == "SleepAccel"]
# call_share: share of the member's calls (over PSG-scored epochs) in each of the four stages;
# the remainder is the share of epochs called gap.
share = m.pivot_table(index=["draw", "ensemble", "member"], columns="stage", values="call_share")
share["gap"] = (1 - share[["wake", "light", "deep", "rem"]].sum(axis=1)).clip(lower=0)
share = share.rename(columns={"rem": "REM"})
stages = ["wake", "light", "deep", "REM", "gap"]

fig, axes = plt.subplots(5, 2, figsize=(9, 7.6), sharex=True)
for d in range(5):
    for c, (ens, word) in enumerate([("independent ×3", "seed"), ("joint ×3", "member")]):
        ax = axes[d, c]
        for k in range(3):
            row = share.loc[(d, ens, k)]
            left = 0.0
            for s in stages:
                ax.barh(k, row[s], left=left, color=STAGE_COLORS[s], edgecolor="white", linewidth=1.2, height=0.72)
                left += row[s]
            if row["REM"] < 0.005:
                ax.text(1.015, k, f"REM {row['REM'] * 100:.2f}%", va="center", ha="left", fontsize=8, color=INK,
                        fontweight="bold")
        ax.set_yticks(range(3), [f"{word} {k}" for k in range(3)])
        ax.set_ylim(2.5, -0.5)
        ax.set_xlim(0, 1.0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        if c == 1:
            ax.yaxis.tick_left()
        if d == 0:
            ax.set_title(f"{ens}", fontsize=10, loc="left", color=INK)
        if c == 0:
            ax.text(-0.27, 0.5, f"draw {d}", transform=ax.transAxes, ha="right", va="center", fontsize=10,
                    fontweight="bold")
for ax in axes[-1]:
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of PSG-scored epochs")
fig.legend(handles=[Patch(color=STAGE_COLORS[s], label=s) for s in stages], loc="upper center", ncol=5,
           frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.02, "", fontsize=1)
fig.subplots_adjust(top=0.93, hspace=0.45, wspace=0.55, left=0.14, right=0.93)
fig.text(0.98, 0.935, "REM < 0.5% marked at the bar end", ha="right", fontsize=8, color=MUTED)
save(fig, "tridoze-known-members.png")

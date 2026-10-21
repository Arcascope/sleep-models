"""The member loss weight β as a dial: ensemble gain over members, and ensemble κ, at β = 0, 0.5 and independent seeds.

β = 0 is the 2026-10-15 joint ×3 (draws 0-4); β = 0.5 is the TriDoze-J pilot (draws 0-2, with the
uncapped draw-0 refit in place of the capped fit); independent seeds are the 2026-10-15 independent ×3
(draws 0-4). β = 0.25 is marked as the full-test arm with no data yet.
Top row: ensemble κ 4-stage minus the mean κ 4-stage of its 3 members. Bottom row: ensemble κ 4-stage
minus independent ×3 on the same draw. Output: tridoze-beta-dial.png.

Usage: python beta_dial.py <run root>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from known_style import ANALYSIS, BLUE, GREY, GRID, INK, MUTED, ORANGE, plt, save  # noqa: E402

COHORTS = ["SleepAccel", "SleepAccel-Clinical"]
X = {"joint": 0.0, "dial": 0.25, "tridoze": 0.5, "independent": 1.0}


def kappa(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d[d.metric == "kappa_4stage"].pivot_table(index=["cohort", "draw"], columns="model", values="mean")


def main() -> None:
    run = Path(sys.argv[1])
    base, pilot = kappa(ANALYSIS / "draws.csv"), kappa(run / "analysis/draws.csv")
    t1 = pilot["tridoze_j"].copy()
    t1_members = pilot[[f"tridoze_j member {m}" for m in range(3)]].copy()
    for c in COHORTS:
        t1.loc[(c, 0)] = pilot.loc[(c, 0), "tridoze_j_split"]
        t1_members.loc[(c, 0)] = pilot.loc[(c, 0), [f"tridoze_j_split member {m}" for m in range(3)]].to_numpy()
    rows = {
        "joint": (base["joint ×3"], base[[f"joint ×3 member {m}" for m in range(3)]].mean(axis=1), base["independent ×3"]),
        "tridoze": (t1.dropna(), t1_members.mean(axis=1), pilot["independent ×3"]),
        "independent": (base["independent ×3"], base[[f"independent seed {s}" for s in range(3)]].mean(axis=1),
                        base["independent ×3"]),
    }
    colours = {"joint": ORANGE, "tridoze": BLUE, "independent": GREY}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharex=True)
    for j, c in enumerate(COHORTS):
        for name, (ens, members, ref) in rows.items():
            gain = (ens - members).loc[c].dropna()
            vs_ref = (ens - ref).loc[c].dropna()
            for i, v in enumerate((gain, vs_ref)):
                ax = axes[i, j]
                ax.scatter([X[name]] * len(v), v, s=26, color=colours[name], edgecolor="white", lw=0.8, zorder=3)
                ax.scatter([X[name]], [v.mean()], marker="_", s=420, color=INK, lw=1.6, zorder=4)
        for i in range(2):
            ax = axes[i, j]
            ax.axvline(X["dial"], color=MUTED, lw=0.8, ls=(0, (3, 3)))
            ax.yaxis.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
        axes[0, j].set_title(c, loc="left", fontsize=10)
        axes[0, j].text(X["dial"] + 0.02, axes[0, j].get_ylim()[1] * 0.92, "β = 0.25:\nfull test", fontsize=7.5,
                        color=MUTED, va="top")
        axes[1, j].axhline(0, color=MUTED, lw=0.8)
        axes[1, j].set_xticks([0, 0.25, 0.5, 1.0], ["β = 0\njoint ×3", "0.25", "β = 0.5\nTriDoze-J", "independent\nseeds"])
    axes[0, 0].set_ylabel("ensemble κ 4-stage −\nmembers' mean κ 4-stage")
    axes[1, 0].set_ylabel("ensemble κ 4-stage −\nindependent ×3")
    fig.text(0.01, -0.03, "Dot: one draw (joint ×3 and independent: draws 0–4, 2026-10-15; TriDoze-J: draws 0–2, "
             "uncapped refit on draw 0). Bar: mean over draws.", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "tridoze-beta-dial.png")


if __name__ == "__main__":
    main()

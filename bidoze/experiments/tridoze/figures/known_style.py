"""Shared style and paths for the figures drawn from the 2026-10-15 analysis tables."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/eric/Engineering/Work/autopisces/sleep-models")
ANALYSIS = ROOT / "bidoze/results/bidoze-joint-ensemble-2026-10-15/analysis"
POSITION = ROOT / "bidoze/results/foundry-apple-stages-2026-10-15/gap_probe/position.csv"
OUT = Path("/home/eric/Engineering/Work/autopisces/sleep-models-research-logs-tridoze-alder/reviews/2026-10-16/figures")

INK, MUTED, GRID = "#1f2328", "#57606a", "#e3e6ea"
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9aa1a9"
STAGE_COLORS = {"wake": BLUE, "light": AQUA, "deep": VIOLET, "REM": ORANGE, "gap": GREY}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(OUT / name)

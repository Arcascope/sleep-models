"""Eval kappa against training step for the TriDoze pilot, one panel per draw.

Draws the 2026-10-15 independent seeds (thin grey) and joint3 (dashed), and each new arm found under
the run root (colour), with each curve's best point marked. Output: tridoze-pilot-curves.png.

Usage: python pilot_curves.py [--run-root <run root>] [--baseline <2026-10-15 run dir>] [--draws 0 1 2]
                              [--out-dir <figures dir>]
Without --run-root only the 2026-10-15 logs are drawn. Default draws: those with a new-arm log, else all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logparse import parse_log  # noqa: E402

BASELINE = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15")
OUT = Path("/home/eric/Engineering/Work/autopisces/sleep-models-research-logs-tridoze-alder/reviews/2026-10-16/figures")
# validated categorical slots 1-4 (blue, orange, aqua, magenta); identity is also carried by the legend
NEW_ARMS = {"tridoze_j": ("TriDoze-J", "#2a78d6"), "tridoze_r": ("TriDoze-R", "#eb6834"),
            "bidoze_xl": ("BiDoze-XL", "#1baf7a"), "bidoze_g": ("BiDoze-G", "#e87ba4")}
INK, MUTED, GRID, GREY = "#1f2328", "#57606a", "#e6e8eb", "#9aa1a9"


def curve(ax, fit, **style):
    ax.plot(fit.steps, fit.kappa, **style)
    if fit.best_step is not None:
        ax.plot([fit.best_step], [fit.kappa[fit.epochs.index(fit.best_epoch)]], marker="o", markersize=6,
                markerfacecolor=style["color"], markeredgecolor="white", markeredgewidth=1.2, linestyle="none",
                zorder=5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--draws", type=int, nargs="*")
    p.add_argument("--out-dir", type=Path, default=OUT)
    a = p.parse_args()

    new = {}  # (arm, draw) -> fits
    if a.run_root:
        for arm in NEW_ARMS:
            for log in sorted(a.run_root.glob(f"{arm}/draw*/training.log")):
                fits = [f for f in parse_log(log) if f.steps]
                if fits:
                    new[arm, int(log.parent.name[4:])] = fits
    draws = a.draws if a.draws else sorted({d for _, d in new}) or list(range(5))

    ncol = min(3, len(draws))
    nrow = -(-(len(draws) + 1) // ncol)  # one extra cell for the legend
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow + 0.4), facecolor="white", squeeze=False)
    flat = list(axes.flat)
    for ax, d in zip(flat, draws):
        for i, fit in enumerate(f for f in parse_log(a.baseline / f"independent/draw{d}/training.log")):
            curve(ax, fit, color=GREY, linewidth=0.8, alpha=0.9, zorder=2)
        for fit in parse_log(a.baseline / f"joint3/draw{d}/training.log"):
            curve(ax, fit, color=INK, linewidth=1.1, linestyle=(0, (5, 2)), zorder=3)
        for arm, (_, colour) in NEW_ARMS.items():
            for fit in new.get((arm, d), []):
                curve(ax, fit, color=colour, linewidth=1.4, zorder=4)
        ax.set_title(f"Draw {d}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Training step", fontsize=9, color=MUTED)
        ax.set_ylabel("Eval κ", fontsize=9, color=MUTED)
        ax.set_ylim(0, 0.62)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GREY)
    for ax in flat[len(draws):]:
        ax.axis("off")
    handles = [Line2D([], [], color=GREY, linewidth=0.8, label="Independent, 3 seeds (2026-10-15)"),
               Line2D([], [], color=INK, linewidth=1.1, linestyle=(0, (5, 2)), label="Joint3, 3 networks (2026-10-15)")]
    handles += [Line2D([], [], color=c, linewidth=1.4, label=name) for arm, (name, c) in NEW_ARMS.items()
                if any(k[0] == arm for k in new)]
    handles.append(Line2D([], [], color=INK, marker="o", linestyle="none", markersize=6, label="Best eval κ of each curve"))
    legend_ax = flat[len(draws)] if len(flat) > len(draws) else fig.add_axes([0, 0, 1, 0.01])
    legend_ax.legend(handles=handles, loc="center left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Eval κ during training, by draw", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "tridoze-pilot-curves.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()

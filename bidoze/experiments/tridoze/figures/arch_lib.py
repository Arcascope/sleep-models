"""Shared drawing helpers for the TriDoze architecture diagrams."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("/home/eric/Engineering/Work/autopisces/sleep-models-research-logs-tridoze-alder/reviews/2026-10-16/figures")

INK = "#1f2328"
MUTED = "#57606a"
# kind -> (fill, edge); input/gap handling blue, trunks aqua, heads/outputs yellow, accent orange
KINDS = {
    "input": ("#dce9f9", "#2a78d6"),
    "trunk": ("#d9f2e8", "#1baf7a"),
    "head": ("#fdeecb", "#c98500"),
    "accent": ("#fbdfd3", "#eb6834"),
    "plain": ("#f3f4f6", "#8c959f"),
}


def canvas():
    fig = plt.figure(figsize=(7, 6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, kind="plain", fs=8, bold=False, dashed=False, fill=True, z=2, color=INK, align="center"):
    f, e = KINDS[kind]
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                       fc=f if fill else "none", ec=e, lw=1.3, ls="--" if dashed else "-", zorder=z)
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color, zorder=z + 1,
                fontweight="bold" if bold else "normal", linespacing=1.25, multialignment=align)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, ls="-", rad=0.0, z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.2, color=color,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fs=7.5, color=MUTED, ha="center", va="center", bold=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, fontweight="bold" if bold else "normal", linespacing=1.25)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(OUT / name)

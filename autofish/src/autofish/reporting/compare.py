"""Render the metric-grid comparison figure from several summary tables.

Where :mod:`autofish.reporting.summary` writes one ``metrics_summary.csv`` per
set of runs, this module reads several of those summaries and draws them
together: one panel per metric, cohorts across the x-axis (hue), and the
summaries -- the runs being compared -- distinguished within each cohort by
texture (solid, dotted, hatched, ...).

Every value comes from a summary's own columns, so the figure cannot disagree
with the tables it was drawn from. A (metric, cohort, run) that a summary does
not carry is simply left out, which lets a run that only scored some cohorts
share the figure with one that scored more.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

matplotlib.rcParams["hatch.linewidth"] = 0.6

# ---------------------------------------------------------------- palette ---
# Validated categorical hues (light surface #fcfcfb), assigned to cohorts in
# order of first appearance.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COHORT_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

#: Texture per compared run: the first is solid, the rest step through hatches.
RUN_HATCHES = ["", "...", "///", "xxx", "\\\\\\", "++"]

DPI = 200
BAR_W = 0.20          # bar thickness, in x data units
PAIR_GAP = 0.085      # air between the bars within one cohort group
ROUND_PX = 4.0        # rounded data-end radius, CSS px
FIXED_HEADROOM = 0.14  # label room above a fixed y-range, as a share of it

#: Required columns of a summary table.
SUMMARY_COLUMNS = ("metric", "cohort", "n", "mean", "ci95_lo", "ci95_hi")

#: How to center a bar and what its whisker spans, per aggregation. ``mean``
#: reads the CI of the mean columns; ``median`` the IQR columns beside them.
AGGREGATIONS = {
    "mean": ("mean", "ci95_lo", "ci95_hi"),
    "median": ("median", "iqr_lo", "iqr_hi"),
}

#: The headline metrics, in reading order: (metric, label, unit, higher_is_better).
PANELS = [
    ("sleep_sensitivity", "Sleep sensitivity", "%", True),
    ("wake_specificity", "Wake specificity", "%", True),
    ("kappa_sleep_wake", "Cohen's κ", "sleep/wake", True),
    ("f1_sleep", "Sleep F1", "%", True),
    ("f1_wake", "Wake F1", "%", True),
    ("f1_macro", "Macro F1", "sleep/wake mean · %", True),
    ("kappa_4stage", "Cohen's κ", "4-stage", True),
    ("tst_error_minutes", "TST error", "min, pred − true", None),
    ("tst_mape", "TST error", "%", False),
    ("waso_mape", "WASO error", "%", False),
    ("light_mape", "Light-sleep error", "%", False),
    ("deep_mape", "Deep-sleep error", "%", False),
    ("rem_mape", "REM-sleep error", "%", False),
]


@dataclass(frozen=True)
class RunSummary:
    """One summary table, labeled with the run it came from."""

    path: Path
    label: str
    table: pd.DataFrame


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", text.lower())).strip("_")


def default_label(path: Path) -> str:
    """The run directory a summary sits under (``<run>/analysis/...``)."""
    return path.parent.parent.name or path.stem


def _agg_columns(agg: str) -> tuple[str, str, str]:
    """The (center, low, high) columns an aggregation reads."""
    try:
        return AGGREGATIONS[agg]
    except KeyError:
        raise ValueError(
            f"unknown aggregation {agg!r}; choose from {sorted(AGGREGATIONS)}"
        ) from None


def required_columns(agg: str = "mean") -> tuple[str, ...]:
    """The columns a summary must carry to be read at this aggregation."""
    return SUMMARY_COLUMNS + tuple(
        c for c in _agg_columns(agg) if c not in SUMMARY_COLUMNS
    )


def load_summaries(
    paths: Iterable[Path | str],
    labels: Sequence[str] | None = None,
    *,
    agg: str = "mean",
) -> list[RunSummary]:
    """Read each summary CSV, pairing it with a label in order.

    ``labels`` is one per path; when omitted each is named for the run
    directory it sits under. ``agg`` decides which columns must be present and
    which the figure reads.
    """
    required = required_columns(agg)
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no summary files given")
    if labels is None:
        labels = [default_label(p) for p in paths]
    if len(labels) != len(paths):
        raise ValueError(
            f"{len(labels)} labels for {len(paths)} summaries; give one "
            "per summary, in the same order"
        )

    runs = []
    for path, label in zip(paths, labels):
        if not path.is_file():
            raise FileNotFoundError(f"no such summary file: {path}")
        table = pd.read_csv(path)
        missing = [c for c in required if c not in table.columns]
        if missing:
            raise ValueError(
                f"{path} is not a metrics summary; missing columns: {missing}"
            )
        runs.append(RunSummary(path=path, label=label, table=table))
    return runs


def cohort_order(runs: Sequence[RunSummary]) -> list[str]:
    """Cohort names in order of first appearance, pooled across runs."""
    order: list[str] = []
    for run in runs:
        for name in run.table["cohort"].drop_duplicates():
            if name not in order:
                order.append(str(name))
    return order


def available_panels(runs: Sequence[RunSummary]) -> list[tuple]:
    """The headline panels carried by at least one run, in reading order."""
    present = set()
    for run in runs:
        present.update(run.table["metric"])
    return [p for p in PANELS if p[0] in present]


def _lookup(run: RunSummary, metric: str, cohort: str):
    rows = run.table[(run.table["metric"] == metric)
                     & (run.table["cohort"] == cohort)]
    return rows.iloc[0] if len(rows) else None


def cohort_sizes(runs: Sequence[RunSummary], cohort: str) -> str:
    """The cohort's subject count as ``n=K``, or a range when runs disagree."""
    sizes = []
    for run in runs:
        rows = run.table[run.table["cohort"] == cohort]
        if len(rows) and not pd.isna(rows.iloc[0]["n"]):
            sizes.append(int(rows.iloc[0]["n"]))
    unique = sorted(set(sizes))
    if not unique:
        return "n=?"
    if len(unique) == 1:
        return f"n={unique[0]}"
    return f"n={unique[0]}–{unique[-1]}"


def _darken(hexcode: str, factor: float = 0.62) -> str:
    """A darker step of a fill's own hue, for the tone-on-tone hatch ink."""
    channels = [int(hexcode[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{max(0, round(c * factor)):02x}" for c in channels)


def _fmt(value: float, metric: str) -> str:
    if metric.startswith("kappa"):
        return f"{value:.2f}"
    if metric == "tst_error_minutes":
        return f"{value:+.1f}"
    return f"{value:.1f}"


def rounded_bar(ax, xc, height, width, r_px, color, base=0.0):
    """Bar with a rounded data-end and a square base at ``base``.

    ``height`` may be negative: the bar then hangs below ``base`` and the
    rounding sits at the lower, data end. A zero-height bar draws nothing,
    which lets a signed metric's exact-zero mean share a panel with bars.
    Matches the per-run figures' mark spec: identical corner geometry, so the
    comparison reads as the same chart with more bars.
    """
    p0 = ax.transData.transform((0, 0))
    px_per_x = ax.transData.transform((1, 0))[0] - p0[0]
    px_per_y = ax.transData.transform((0, 1))[1] - p0[1]
    rx = r_px * (DPI / 96.0) / px_per_x
    ry = r_px * (DPI / 96.0) / px_per_y

    x0, x1 = xc - width / 2, xc + width / 2
    y0, y1 = base, height
    dy = y1 - y0
    if not abs(dy) > 0:
        return
    ry = min(ry, 0.45 * abs(dy))
    rx = min(rx, 0.45 * width)
    junction = y1 - math.copysign(ry, dy)

    verts = [
        (x0, y0),
        (x0, junction),
        (x0, y1), (x0 + rx, y1),
        (x1 - rx, y1),
        (x1, y1), (x1, junction),
        (x1, y0),
        (x0, y0),
    ]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
             MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color,
                           edgecolor="none", zorder=3))


def _textured_bar(ax, xc, height, color, hatch):
    """Solid fill, plus the run's hatch on top, reusing the fill's own path."""
    rounded_bar(ax, xc, height, BAR_W, ROUND_PX, color)
    if not hatch:
        return
    path = ax.patches[-1].get_path()
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=_darken(color),
                           hatch=hatch, linewidth=0, zorder=3.5))


def _layout(cohorts: Sequence[str], n_runs: int):
    """Group centers and per-run offsets, wide enough for any run count."""
    span = n_runs * BAR_W + (n_runs - 1) * PAIR_GAP
    pitch = max(1.0, span + 0.30)
    centers = {cohort: i * pitch for i, cohort in enumerate(cohorts)}
    start = -(span - BAR_W) / 2
    offsets = [start + r * (BAR_W + PAIR_GAP) for r in range(n_runs)]
    x_lo = centers[cohorts[0]] - span / 2 - 0.25
    x_hi = centers[cohorts[-1]] + span / 2 + 0.25
    return centers, offsets, x_lo, x_hi


def draw_comparison(
    runs: Sequence[RunSummary],
    *,
    output_dir: Path | str,
    name: str = "metrics_grid",
    title: str | None = None,
    agg: str = "mean",
    metrics: Sequence[str] | None = None,
    ncols: int | None = None,
    ylims: Mapping[str, tuple[float, float]] | None = None,
) -> list[Path]:
    """Draw every run side by side and write ``<name>.png`` and ``.pdf``.

    ``agg`` picks the bar center and whisker: ``mean`` uses the mean and its
    95% CI, ``median`` the median and the IQR.

    ``metrics`` limits the grid to those headline panels, drawn in the order
    given; ``ncols`` sets the grid width (default: up to five per row).
    ``ylims`` pins a panel's y-axis to a fixed ``(low, high)`` by metric, for
    bounded scores that read best on their full range. Ticks and the axis line
    stop at ``high``; a band above it is left clear for the value labels.
    """
    if not runs:
        raise ValueError("no summaries to plot")
    center_col, lo_col, hi_col = _agg_columns(agg)
    panels = available_panels(runs)
    if metrics is not None:
        by_metric = {panel[0]: panel for panel in panels}
        panels = [by_metric[m] for m in metrics if m in by_metric]
    if not panels:
        raise ValueError("no headline metrics in the given summaries")

    cohorts = cohort_order(runs)
    n_runs = len(runs)
    colors = {c: COHORT_HUES[i % len(COHORT_HUES)] for i, c in enumerate(cohorts)}
    centers, offsets, x_lo, x_hi = _layout(cohorts, n_runs)

    ncols = min(ncols or 5, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.7 * nrows),
                             dpi=DPI, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.045, right=0.988, top=0.800, bottom=0.110,
                        wspace=0.42, hspace=0.62)

    for ax, (metric, label, unit, hib) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        lows, highs = [], []
        for cohort in cohorts:
            for run in runs:
                row = _lookup(run, metric, cohort)
                if row is None:
                    continue
                value = float(row[center_col])
                for col, into in ((lo_col, lows), (hi_col, highs)):
                    if not pd.isna(row[col]):
                        into.append(float(row[col]))
                # A row whose CI is NaN (n=1) still owns a bar; scale to it.
                if not pd.isna(value):
                    lows.append(value)
                    highs.append(value)
        if highs:
            top, bottom = max(highs), min(lows)
            top = max(top, 0.0)
            bottom = min(bottom, 0.0)
            span = top - bottom
        else:
            top, bottom, span = 1.0, 0.0, 0.0
        fixed = ylims.get(metric) if ylims else None
        if fixed is not None:
            lo_fixed, hi_fixed = fixed
            ax.set_ylim(lo_fixed, hi_fixed + FIXED_HEADROOM * (hi_fixed - lo_fixed))
        elif span > 0:
            pad = 0.30 * span
            # Bars sit on the zero baseline; keep it on the axis only when
            # the data is one-signed, and leave label headroom above.
            ax.set_ylim(bottom - pad if bottom < 0 else 0.0, top + pad)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_xlim(x_lo, x_hi)

        for cohort in cohorts:
            for i, run in enumerate(runs):
                row = _lookup(run, metric, cohort)
                if row is None:
                    continue
                value = float(row[center_col])
                lo, hi = float(row[lo_col]), float(row[hi_col])
                xc = centers[cohort] + offsets[i]
                _textured_bar(ax, xc, value, colors[cohort], RUN_HATCHES[i])
                ax.errorbar(xc, value, yerr=[[value - lo], [hi - value]], fmt="none",
                            ecolor=INK_2, elinewidth=1.1, capsize=3.4,
                            capthick=1.1, zorder=4)
                # The label sits past the whisker on the data side, so a
                # bar hanging below zero is labelled under its whisker.
                below = value < 0 or (not pd.isna(lo) and hi <= 0)
                y_label, va = ((lo, "top") if below else (hi, "bottom"))
                offset = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(xc, y_label - offset if below else y_label + offset,
                        _fmt(value, metric),
                        ha="center", va=va,
                        fontsize=6.8 if n_runs > 1 else 8.2, color=INK, zorder=5)

            ax.text(centers[cohort], -0.075, cohort, ha="center", va="top",
                    fontsize=7.4 if len(cohort) > 6 else 7.6, color=MUTED,
                    clip_on=False, transform=ax.get_xaxis_transform())

        ax.set_xticks([])
        ax.set_title(label, fontsize=10.2, color=INK, pad=15)
        sub = f"{unit} · " if unit else ""
        if hib is None:
            sub += "signed"
        else:
            sub += f"{'higher' if hib else 'lower'} is better"
        ax.text(0.5, 1.018, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.4, color=MUTED)

        if fixed is not None:
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
                [lo_fixed + (hi_fixed - lo_fixed) * k / 4 for k in range(1, 5)]))
            ax.spines["left"].set_bounds(lo_fixed, hi_fixed)
        else:
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10],
                                              prune="lower"))
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.9)
        if bottom < 0:
            # The signed panels read against the zero line, not the axis.
            ax.axhline(0.0, color=BASELINE, linewidth=0.9, zorder=2.5)
            ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=3)

    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)

    # Two legend rows: hue carries the cohort, texture carries the run.
    cohort_handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=9,
               markerfacecolor=colors[c], markeredgecolor="none",
               label=f"{c} ({cohort_sizes(runs, c)})") for c in cohorts]
    fig.legend(handles=cohort_handles, loc="upper left",
               bbox_to_anchor=(0.026, 0.925), frameon=False,
               fontsize=9.0, labelcolor=INK_2, ncol=min(len(cohorts), 4),
               handletextpad=0.6, columnspacing=2.0)

    run_handles = [
        PathPatch(MPath([(0, 0)], [MPath.MOVETO]), facecolor=BASELINE,
                  edgecolor=INK_2, hatch=hatch, linewidth=0, label=run.label)
        for run, hatch in zip(runs, RUN_HATCHES)]
    fig.legend(handles=run_handles, loc="upper left",
               bbox_to_anchor=(0.026, 0.882), frameon=False, fontsize=9.0,
               labelcolor=MUTED, ncol=min(len(runs), 4), handletextpad=0.6,
               columnspacing=2.4)

    default_title = "Macro average over subjects, with 95% CI of the mean"
    if agg == "median":
        default_title = "Macro median over subjects, with the interquartile range"
    if len(runs) > 1:
        default_title += ": " + " vs ".join(run.label for run in runs)
    fig.text(0.024, 0.972, title or default_title,
             fontsize=12.4, color=INK, ha="left", va="bottom")
    fig.text(0.024, 0.014,
             "Each panel has its own y-scale.  MAPE panels are heavily right-"
             "skewed; medians and IQRs are the robust read, and every value "
             "is in the summary tables.",
             fontsize=7.8, color=MUTED, ha="left", va="bottom")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def compare(
    paths: Iterable[Path | str],
    *,
    output_dir: Path | str,
    labels: Sequence[str] | None = None,
    name: str = "metrics_grid",
    title: str | None = None,
    agg: str = "mean",
) -> list[Path]:
    """Load the summaries, draw the comparison, and return the files written."""
    runs = load_summaries(paths, labels, agg=agg)
    for run in runs:
        print(f"  {run.label:<16} {run.path}")
    return draw_comparison(runs, output_dir=output_dir, name=name, title=title,
                           agg=agg)

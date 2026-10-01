"""Hypnogram of one prediction: raw BiDoze, the GGIR-filtered result, GGIR tracks.

Drawn from a :class:`bidoze.predict.Prediction`, or from the CSV
``bidoze predict`` wrote (via :func:`bidoze.predict.read_prediction_csv`),
so no model or accelerometer data is needed to redraw it. Rows, top to bottom:

- optional PSG stages, when a caller has them;
- raw BiDoze stages, with the sleep the SIB filter removed shaded;
- BiDoze after the filter;
- a track of GGIR's sustained-inactivity bouts (SIB), estimated sleep-period
  windows (SPT) and epochs with no data.

Without the filter only the raw row is drawn. With PSG, removed sleep is
shaded by whether PSG called it wake or sleep; without PSG it is one neutral
shade, since nothing says whether a removal was right.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

EPOCH_HOURS = 30.0 / 3600.0
#: Stage rows run deep (bottom) to wake (top).
STAGE_LEVEL = {"deep": 0, "light": 1, "rem": 2, "wake": 3}
#: Epoch start times above this are taken to be Unix seconds.
UNIX_THRESHOLD_S = 1e9

# Reference palette, light mode: text inks, grid, and categorical slots.
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAW, FILTERED, SIB, SPT = "#2a78d6", "#4a3aa7", "#2a78d6", "#4a3aa7"
REMOVED = "#eda100"  # gold: sleep the filter removed
REMOVED_WAKE, REMOVED_SLEEP = "#1baf7a", "#eb6834"
NO_DATA = "#a9a8a1"  # hatch ink: epochs with no data


def plot_prediction(
    prediction,
    path: Path | str,
    *,
    psg: Sequence[str | None] | None = None,
    title: str | None = None,
) -> Path:
    """Save one recording's hypnogram as a PNG and return its path.

    ``psg`` is an optional per-epoch PSG stage name (``"wake"``, ``"light"``,
    ``"deep"``, ``"rem"``, or ``None`` where unscored), aligned to the
    prediction's epochs.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "drawing hypnograms needs the optional dependency: "
            "pip install 'arcasleep-bidoze[report]'"
        ) from exc

    path = Path(path)
    start_s = np.asarray(prediction.epoch_start_s, dtype=np.float64)
    n = start_s.shape[0]
    hours = (start_s - start_s[0]) / 3600.0
    raw = np.asarray(prediction.raw_labels, dtype=object)
    filtered = np.asarray(prediction.labels, dtype=object)
    is_gap = np.asarray(prediction.is_gap, dtype=bool)
    is_sib = _optional_flags(getattr(prediction, "is_sib", None), n)
    is_spt = _optional_flags(getattr(prediction, "is_spt", None), n)
    filter_ran = is_sib is not None

    raw_sleep = ~np.isin(raw, ["wake", "gap"])
    filtered_sleep = ~np.isin(filtered, ["wake", "gap"])
    removed = raw_sleep & ~filtered_sleep & ~is_gap
    psg_levels = None
    if psg is not None:
        psg = np.asarray(psg, dtype=object)
        if psg.shape[0] != n:
            raise ValueError(f"psg has {psg.shape[0]} epochs; the prediction has {n}")
        psg_levels = _levels(psg)
        psg_sleep = np.isin(psg, ["light", "deep", "rem"])
        psg_wake = psg == "wake"

    panels = (["psg"] if psg_levels is not None else []) + ["raw"]
    if filter_ran:
        panels += ["filtered", "track"]
    ratios = [1.4 if name == "track" else 3 for name in panels]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(16, 1.2 + 1.8 * sum(ratios) / 3), sharex=True,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.32}, squeeze=False,
    )
    axes = dict(zip(panels, axes[:, 0], strict=True))

    if psg_levels is not None:
        _stages(axes["psg"], hours, psg_levels, TEXT, "PSG")
    legend = []
    if filter_ran:
        if psg_levels is not None:
            _shade(axes["raw"], hours, removed & psg_wake, REMOVED_WAKE, 0.35)
            _shade(axes["raw"], hours, removed & psg_sleep, REMOVED_SLEEP, 0.45)
            legend += [
                Patch(color=REMOVED_WAKE, alpha=0.35, label="Removed by SIB, PSG wake"),
                Patch(color=REMOVED_SLEEP, alpha=0.45, label="Removed by SIB, PSG sleep"),
            ]
        else:
            _shade(axes["raw"], hours, removed, REMOVED, 0.4)
            legend.append(Patch(color=REMOVED, alpha=0.4, label="Sleep removed by SIB"))
    raw_title = "BiDoze (raw)" + (", sleep removed by SIB shaded" if filter_ran else "")
    _stages(axes["raw"], hours, _levels(raw), RAW, raw_title)

    if filter_ran:
        _stages(
            axes["filtered"], hours, _levels(filtered), FILTERED, "BiDoze after the SIB filter"
        )
        track = axes["track"]
        _shade(track, hours, is_sib, SIB, 0.8, 0.55, 0.95)
        spt_found = is_spt is not None and bool(is_spt.any())
        if spt_found:
            _shade(track, hours, is_spt, SPT, 0.8, 0.05, 0.45)
        spt_label = "SPT window" if spt_found else "SPT (not estimated)"
        track.set_yticks([0.25, 0.75], [spt_label, "SIB"])
        track.set_ylim(0, 1)
        legend.append(Patch(color=SIB, alpha=0.8, label="SIB"))
        if spt_found:
            legend.append(Patch(color=SPT, alpha=0.8, label="SPT window"))
    bottom = axes[panels[-1]]
    for ax in axes.values():
        _hatch(ax, hours, is_gap)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    legend.append(
        Patch(facecolor="none", edgecolor=NO_DATA, hatch="////", lw=0, label="No data")
    )
    bottom.set_xlim(hours[0], hours[-1] + EPOCH_HOURS)
    bottom.set_xlabel(_time_label(start_s[0]), color=MUTED)

    fig.suptitle(
        _headline(title or path.stem, raw_sleep, filtered_sleep, removed, prediction, filter_ran,
                  psg_levels is not None and (psg_sleep, psg_wake)),
        x=0.01, ha="left", fontsize=11, color=TEXT,
    )
    fig.legend(handles=legend, loc="upper right", ncol=len(legend), fontsize=8, frameon=False,
               bbox_to_anchor=(0.99, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _optional_flags(flags, n: int) -> np.ndarray | None:
    if flags is None:
        return None
    flags = np.asarray(flags, dtype=bool)
    if flags.shape[0] != n:
        raise ValueError(f"flag array has {flags.shape[0]} epochs; the prediction has {n}")
    return flags


def _levels(stages: np.ndarray) -> np.ndarray:
    return np.array([STAGE_LEVEL.get(stage, np.nan) for stage in stages], dtype=float)


def _stages(ax, hours: np.ndarray, levels: np.ndarray, color: str, title: str) -> None:
    ax.step(hours, levels, where="post", color=color, lw=1.0)
    ax.set_yticks([0, 1, 2, 3], ["Deep", "Light", "REM", "Wake"])
    ax.set_ylim(-0.5, 3.5)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT)


def _spans(hours, mask):
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        yield hours[a], hours[b - 1] + EPOCH_HOURS


def _shade(ax, hours, mask, color, alpha, lo=0.0, hi=1.0) -> None:
    for x0, x1 in _spans(hours, mask):
        ax.axvspan(x0, x1, lo, hi, color=color, alpha=alpha, lw=0)


def _hatch(ax, hours, mask) -> None:
    """No-data epochs: a texture, so they never read as a shaded removal."""
    for x0, x1 in _spans(hours, mask):
        ax.axvspan(x0, x1, facecolor="none", edgecolor=NO_DATA, hatch="////", lw=0)


def _time_label(first_start_s: float) -> str:
    if first_start_s > UNIX_THRESHOLD_S:
        start = datetime.fromtimestamp(first_start_s, tz=UTC)
        return f"Hours from {start:%Y-%m-%d %H:%M} UTC"
    return "Hours from the first epoch"


def _headline(name, raw_sleep, filtered_sleep, removed, prediction, filter_ran, psg) -> str:
    minutes = 0.5
    parts = [name]
    if filter_ran:
        parts.append(
            f"sleep {raw_sleep.sum() * minutes:.0f} → {filtered_sleep.sum() * minutes:.0f} min "
            f"({removed.sum() * minutes:.0f} removed by SIB)"
        )
        is_nap = getattr(prediction, "is_nap", None)
        if is_nap is not None and np.any(is_nap):
            parts.append(f"{np.count_nonzero(is_nap) * minutes:.0f} min outside SPT (naps)")
    else:
        parts.append(f"sleep {raw_sleep.sum() * minutes:.0f} min")
    if psg:
        psg_sleep, psg_wake = psg
        def rates(sleep):
            return sleep[psg_sleep].mean(), (~sleep[psg_wake]).mean()
        s0, p0 = rates(raw_sleep)
        if filter_ran:
            s1, p1 = rates(filtered_sleep)
            parts.append(f"sleep sens. {s0:.3f} → {s1:.3f}, wake spec. {p0:.3f} → {p1:.3f}")
        else:
            parts.append(f"sleep sens. {s0:.3f}, wake spec. {p0:.3f}")
    return "   |   ".join(parts)

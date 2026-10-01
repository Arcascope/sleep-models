"""Per-subject hypnogram rendering for an evaluation cohort.

The figure is, top to bottom: 
- the stage probabilities as a stacked area, 
- the argmax confidence, 
- the predicted stages,
- the PSG truth as one lane per stage,
- one panel per spectral channel.

Nothing here reads a config or loads a model; the caller hands over the pooled,
padded inputs, the class probabilities, and the rendered labels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

from bidoze.data.batching import split_gap_row


#: Class order the model emits: four stages plus the gap class.
_CLASS_NAMES = ["Wake", "Light", "Deep", "REM", "Gap"]
_CLASS_COLORS = {
    "Wake": "#f59e0b",
    "Light": "#4f83f1",
    "Deep": "#343c9a",
    "REM": "#24bfd0",
    "Gap": "#8f96a3",
}
#: Stacking order for the probability area -- gap first, then the stages.
_ORDERED_STAGE_INDICES = [4, 0, 3, 1, 2]
#: Top-to-bottom lane order for the stage panels.
_LANE_ORDER = [0, 3, 1, 2, 4]
_VALID_FEATURE_KINDS = {"spectrogram", "embedding"}

#: Gap class index in the rendered labels, and the epoch sentinel the plotter
#: masks away (an epoch with no stage at all).
_GAP_STAGE_INDEX = 4
_MASKED_LABEL = -2

_WAKE_STAGE_INDEX = 0
_SLEEP_STAGE_INDICES = (1, 2, 3)
_MINUTES_PER_EPOCH = 0.5
_STAGE_SUMMARY_NAMES = ("tst", "wake", "light", "deep", "rem")
_TITLE_LABELS = {
    "tst": "MAPE TST",
    "wake": "Wake",
    "light": "Light",
    "deep": "Deep",
    "rem": "REM",
}


# --------------------------------------------------------------- spectrogram --


def normalized_log_spectrogram(
    spec: np.ndarray, missing: np.ndarray | None = None
) -> np.ndarray:
    """The dB surface the spectrogram panel draws.

    Already-log features (``spectrogram_kind: log_psd``) are negative, so they
    are min-maxed directly; linear-magnitude features are clamped at zero and
    logged. ``missing`` flags epochs with no data (gap row at 1); they are drawn
    at the floor so absent data reads as dark silence.
    """
    spec_plot = np.asarray(spec, dtype=float).T.copy()
    if spec_plot.size and spec_plot.min() < 0.0:
        if missing is not None and missing.any():
            spec_plot[:, missing] = spec_plot.min()
        spec_plot -= spec_plot.min()
        max_val = spec_plot.max()
        if max_val > 0:
            spec_plot /= max_val
        return spec_plot
    spec_plot = np.maximum(spec_plot, 0.0)
    spec_plot -= spec_plot.min()
    max_val = spec_plot.max()
    if max_val > 0:
        spec_plot /= max_val
    spec_plot += 1e-8
    return 10 * np.log10(10 * spec_plot)


def _spectrogram_panels(
    inputs: np.ndarray,
    channel_names: list[str] | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Split an ``(T, F)`` or ``(T, F, C)`` surface into one panel per channel."""
    arr = np.asarray(inputs)
    if arr.ndim == 2:
        name = channel_names[0] if channel_names is not None and len(channel_names) == 1 else "Freq Bin"
        return [(arr, name)]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D spectrogram input, got shape {arr.shape}")

    if channel_names is not None and arr.shape[-1] == len(channel_names):
        return [(arr[:, :, idx], channel_names[idx]) for idx in range(arr.shape[-1])]
    if channel_names is not None and arr.shape[0] == len(channel_names):
        return [(arr[idx], channel_names[idx]) for idx in range(arr.shape[0])]

    # Stacked processing config produces channel-last spectrograms: (T, F, C).
    if arr.shape[-1] <= arr.shape[0] and arr.shape[-1] <= arr.shape[1]:
        return [(arr[:, :, idx], f"Spec [{idx}]") for idx in range(arr.shape[-1])]

    return [(arr[idx], f"Spec [{idx}]") for idx in range(arr.shape[0])]


# ------------------------------------------------------------------- stages --


def _stage_runs(labels: np.ndarray, class_idx: int) -> list[tuple[int, int]]:
    indices = np.where(labels == class_idx)[0]
    if len(indices) == 0:
        return []
    gaps = np.where(np.diff(indices) > 1)[0]
    return [(int(block[0]), int(block[-1]) + 1) for block in np.split(indices, gaps + 1)]


def _draw_stage_panel(ax, labels: np.ndarray, title: str, *, class_count: int) -> None:
    valid_labels = labels[labels >= 0]
    if valid_labels.size:
        class_count = max(class_count, int(valid_labels.max()) + 1)
    stage_indices = [idx for idx in _LANE_ORDER if idx < class_count]
    if len(stage_indices) == 0:
        ax.set_ylabel(title)
        return

    lane_for_stage = {
        class_idx: len(stage_indices) - 1 - lane_idx
        for lane_idx, class_idx in enumerate(stage_indices)
    }
    lane_height = 0.62
    seen: set[str] = set()
    for class_idx in stage_indices:
        class_name = _CLASS_NAMES[class_idx]
        lane_y = lane_for_stage[class_idx] - lane_height / 2.0
        for start, end in _stage_runs(labels, class_idx):
            label = class_name if class_name not in seen else None
            if label is not None:
                seen.add(class_name)
            ax.broken_barh(
                [(start, end - start)],
                (lane_y, lane_height),
                facecolors=_CLASS_COLORS[class_name],
                edgecolors="tab:blue" if class_name == "Gap" else "none",
                linewidth=1.2 if class_name == "Gap" else 0.0,
                alpha=0.95,
                label=label,
            )

    tick_positions = [lane_for_stage[idx] for idx in stage_indices]
    tick_labels = [_CLASS_NAMES[idx] for idx in stage_indices]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-0.65, len(stage_indices) - 0.35)
    ax.set_ylabel(title, rotation=0, ha="right", va="center", labelpad=42, fontweight="bold")
    ax.grid(axis="y", color="#d7dde8", linewidth=0.7, alpha=0.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    if seen:
        ax.legend(loc="upper left", ncol=min(len(seen), 5), frameon=False)


def _add_masked_epoch_overlay(axes: list[plt.Axes], labels: np.ndarray) -> None:
    masked_indices = np.where(labels < 0)[0]
    if len(masked_indices) == 0:
        return
    masked_gaps = np.where(np.diff(masked_indices) > 1)[0]
    masked_blocks = np.split(masked_indices, masked_gaps + 1)
    for ax in axes:
        for block in masked_blocks:
            ax.axvspan(int(block[0]), int(block[-1]) + 1, color="gray", alpha=0.3, zorder=10)


# ------------------------------------------------------------------ summary --


def _scored_epoch_mask(labels: np.ndarray) -> np.ndarray:
    """Epochs with real sleep-stage truth (gap and pad dropped)."""
    y = np.asarray(labels)
    return (y >= 0) & (y != _GAP_STAGE_INDEX)


def _stage_mape(true_count: int, pred_count: int) -> float:
    if true_count == 0:
        return 0.0 if pred_count == 0 else float("nan")
    return 100.0 * abs(pred_count - true_count) / float(true_count)


def _signed_error_minutes(true_count: int, pred_count: int) -> float:
    return (pred_count - true_count) * _MINUTES_PER_EPOCH


def _format_mape(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.1f}%"


def _format_signed_error(value: float) -> str:
    return f"{value:+.1f}m"


def _stage_counts(labels: np.ndarray, pred_stage: np.ndarray) -> dict[str, tuple[int, int]]:
    labels = np.asarray(labels)
    pred_stage = np.asarray(pred_stage)
    if labels.shape != pred_stage.shape:
        raise ValueError(
            "labels and pred_stage must have the same shape; "
            f"got {labels.shape} and {pred_stage.shape}"
        )

    scored = _scored_epoch_mask(labels)
    y = labels[scored].astype(np.int64)
    p = pred_stage[scored].astype(np.int64)

    def pair(true_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[int, int]:
        return int(np.sum(true_mask)), int(np.sum(pred_mask))

    return {
        "tst": pair(np.isin(y, _SLEEP_STAGE_INDICES), np.isin(p, _SLEEP_STAGE_INDICES)),
        "wake": pair(y == _WAKE_STAGE_INDEX, p == _WAKE_STAGE_INDEX),
        "light": pair(y == 1, p == 1),
        "deep": pair(y == 2, p == 2),
        "rem": pair(y == 3, p == 3),
    }


def subject_stage_summary(labels: np.ndarray, pred_stage: np.ndarray) -> dict[str, float]:
    """``{name}_mape`` and ``{name}_err_min`` for every stage plus TST."""
    summary: dict[str, float] = {}
    for name, (true_count, pred_count) in _stage_counts(labels, pred_stage).items():
        summary[f"{name}_mape"] = _stage_mape(true_count, pred_count)
        summary[f"{name}_err_min"] = _signed_error_minutes(true_count, pred_count)
    return summary


def stage_summary_title_fields(
    summary: dict[str, float],
    names: tuple[str, ...] = _STAGE_SUMMARY_NAMES,
) -> list[str]:
    """Rendered ``"Deep: 12.3% (+4.5m)"`` fields for a hypnogram title."""
    fields: list[str] = []
    for name in names:
        mape_key, err_key = f"{name}_mape", f"{name}_err_min"
        if mape_key not in summary or err_key not in summary:
            raise KeyError(f"summary is missing {mape_key}/{err_key}")
        fields.append(
            f"{_TITLE_LABELS.get(name, name)}: "
            f"{_format_mape(summary[mape_key])} "
            f"({_format_signed_error(summary[err_key])})"
        )
    return fields


# ------------------------------------------------------------------- render --


def render_subject_hypnogram(
    output_path: Path,
    inputs: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    title_prefix: str,
    channel_names: list[str] | None = None,
    epoch_seconds: float | None = None,
    feature_kind: str = "spectrogram",
) -> dict[str, float]:
    """Draw one subject's hypnogram PNG and return its stage summary.

    ``inputs`` is the pooled, padded feature surface ``(T, F[, C])``; ``probs``
    is ``(T, num_classes + score_gap)`` softmax; ``labels`` is the rendered
    truth with the gap class at index 4 (pad at ``-2``).
    """
    if feature_kind not in _VALID_FEATURE_KINDS:
        raise ValueError(
            f"feature_kind must be one of {sorted(_VALID_FEATURE_KINDS)}, got {feature_kind!r}"
        )
    time_indices = np.arange(probs.shape[0])
    pred_stage = np.argmax(probs, axis=-1)
    pred_conf = np.max(probs, axis=-1)
    summary = subject_stage_summary(labels, pred_stage)

    # The pooled surface carries the missing-frame fraction as its last
    # frequency row; it is not a spectrum, so it is split off before drawing.
    inputs, missing_fraction = split_gap_row(np.asarray(inputs))
    missing = missing_fraction >= 1.0
    spec_panels = _spectrogram_panels(inputs, channel_names=channel_names)
    n_spec = len(spec_panels)
    n_axes = 4 + n_spec
    fig, axes = plt.subplots(n_axes, 1, figsize=(40, 4 * n_axes), sharex=True)
    axes_list = list(axes)
    ordered_stage_indices = [
        idx for idx in _ORDERED_STAGE_INDICES if idx < probs.shape[1]
    ]

    cumulative = np.zeros(probs.shape[0])
    for class_idx in ordered_stage_indices:
        class_name = _CLASS_NAMES[class_idx]
        next_cumulative = cumulative + probs[:, class_idx]
        axes_list[0].fill_between(
            time_indices,
            cumulative,
            next_cumulative,
            color=_CLASS_COLORS[class_name],
            alpha=0.7,
            label=class_name,
        )
        cumulative = next_cumulative
    axes_list[0].set_ylabel("Class Probabilities")
    axes_list[0].set_ylim(0.0, 1.0)
    axes_list[0].legend(loc="upper left")

    axes_list[1].plot(time_indices, pred_conf, color="black", label="Argmax Confidence")
    axes_list[1].set_ylim(-0.05, 1.05)
    axes_list[1].set_ylabel("Argmax Prob")
    axes_list[1].legend(loc="upper left")

    _draw_stage_panel(
        axes_list[2],
        pred_stage,
        "Pred",
        class_count=probs.shape[1],
    )
    _draw_stage_panel(
        axes_list[3],
        labels,
        "PSG",
        class_count=probs.shape[1],
    )

    for ch_idx, (spec, ylabel) in enumerate(spec_panels):
        axis = axes_list[4 + ch_idx]
        if feature_kind == "embedding":
            # Embedding features are signed and have no dB interpretation, so the
            # log/min-max spectrogram scaling would misrepresent them. Standardise
            # every dimension across time (median / IQR) and draw the deviations
            # on a diverging map.
            spec_plot = np.asarray(spec, dtype=float).T
            median = np.nanmedian(spec_plot, axis=1, keepdims=True)
            q75, q25 = np.nanpercentile(spec_plot, [75, 25], axis=1, keepdims=True)
            scale = (q75 - q25) / 1.349
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
            spec_plot = (spec_plot - median) / scale
            limit = 3.0
            axis.imshow(
                spec_plot,
                aspect="auto",
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                extent=[time_indices[0], time_indices[-1] + 1, 0, float(spec_plot.shape[0])],
            )
            label = ylabel if ylabel != "Freq Bin" else "Embedding dim"
            axis.set_ylabel(f"{label} (z, +/-{limit:.0f})")
        else:
            spec_plot = normalized_log_spectrogram(spec, missing)
            axis.imshow(
                spec_plot,
                aspect="auto",
                origin="lower",
                extent=[
                    time_indices[0],
                    time_indices[-1] + 1,
                    0.0,
                    float(spec_plot.shape[0]),
                ],
            )
            axis.set_ylabel(ylabel)

    _add_masked_epoch_overlay(axes_list, labels)

    fig.suptitle(
        title_prefix
        + " | "
        + " | ".join(stage_summary_title_fields(summary)),
        fontsize=20,
        y=0.995,
    )
    axes_list[-1].set_xlim(0, time_indices[-1] + 1)
    if epoch_seconds is None:
        axes_list[-1].set_xlabel("Epoch")
    else:
        hours_per_epoch = float(epoch_seconds) / 3600.0
        axes_list[-1].xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value * hours_per_epoch:g}")
        )
        axes_list[-1].xaxis.set_major_locator(MultipleLocator(5.0 / hours_per_epoch))
        axes_list[-1].set_xlabel("Elapsed time (hours)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return summary

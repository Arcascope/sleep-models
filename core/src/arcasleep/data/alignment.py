"""Per-night alignment of PSG labels to the accelerometer, from the ETC wake score.

Some cohorts record the accelerometer and the PSG on two clocks, and the two
can disagree by minutes, differently each night. A model trained on such
labels learns to call wake late. This module estimates each night's offset
from the data alone and moves the labels to match.

For each candidate shift ``k`` (in 30 s epochs) the PSG label at ``t + k`` is
compared with the ETC wake score (:mod:`arcasleep.etc`) at ``t``; the shift
whose comparison is best is the night's offset. ``k > 0`` means the PSG runs
late, and aligning sets ``y[t] = y[t + k]``. Two comparisons are offered:

- ``auroc``: the AUROC of the wake score against PSG wake.
- ``kappa``: Cohen's kappa between PSG wake and a wake call made by marking
  the night's highest-scoring epochs as wake, as many as the PSG has.

Both are parameter-free. A night whose best comparison is weak, sits at the
edge of the search, or does not stand clear of the shifts more than
:data:`PEAK_HALF_WIDTH` epochs away is not aligned: it is dropped, with the
reason, rather than kept with labels that may be wrong.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from arcasleep.etc import DEFAULT_CHANNEL, ETCKernel, epoch_energy, wake_score

STYLES = ("auroc", "kappa")
#: Largest shift searched, in epochs, either way.
DEFAULT_MAX_SHIFT = 20
#: Lowest peak comparison a night may have and still be aligned, per style.
#: With the default kernel these keep 95% (kappa) and 93% (auroc) of
#: Weaver, SleepAccel and SleepAccel-Clinical nights.
DEFAULT_MIN_PEAK = {"auroc": 0.6, "kappa": 0.1}
#: How far the peak must stand above the best shift outside its neighbourhood.
DEFAULT_MIN_MARGIN = 0.005
#: Shifts this close to the peak count as its neighbourhood, not a rival.
PEAK_HALF_WIDTH = 2
#: The label written into epochs a shift leaves without a PSG label.
UNSCORED_LABEL = -1
#: The NPZ key recording a cache file's applied shift.
SHIFT_KEY = "label_shift_epochs"


@dataclass(frozen=True)
class AlignmentSettings:
    """How offsets are estimated and when a night is dropped instead."""

    kernel: ETCKernel = field(default_factory=ETCKernel)
    style: str = "kappa"
    max_shift: int = DEFAULT_MAX_SHIFT
    min_peak: float | None = None
    min_margin: float = DEFAULT_MIN_MARGIN
    channel: int = DEFAULT_CHANNEL

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {self.style!r}")
        if self.max_shift < 1:
            raise ValueError(f"max_shift must be at least 1 epoch, got {self.max_shift}")

    @property
    def peak_floor(self) -> float:
        return DEFAULT_MIN_PEAK[self.style] if self.min_peak is None else self.min_peak


@dataclass(frozen=True)
class Offset:
    """One night's estimated offset.

    ``shift`` is ``None`` when the night is dropped; ``dropped`` then says why.
    ``peak`` is the comparison at the best shift, ``at_zero`` at no shift, and
    ``margin`` how far the peak stands above the best shift outside its
    neighbourhood.
    """

    shift: int | None
    best_shift: int | None
    peak: float
    at_zero: float
    margin: float
    dropped: str | None = None


def auroc(positive: np.ndarray, score: np.ndarray) -> float:
    """AUROC of ``score`` for the ``positive`` class, ties at average rank.

    NaN when either class is empty.
    """
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = np.asarray(score, dtype=float)[order]
    _, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    mean_rank = (ends - (counts - 1) / 2.0)[inverse]
    ranks = np.empty(positive.size)
    ranks[order] = mean_rank
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def matched_kappa(positive: np.ndarray, score: np.ndarray) -> float:
    """Kappa between ``positive`` and a call of the top-scoring epochs, as many as are positive."""
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n = positive.size
    if n_pos == 0 or n_pos == n:
        return float("nan")
    called = np.zeros(n, dtype=bool)
    called[np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:n_pos]] = True
    observed = float(np.mean(called == positive))
    rate = n_pos / n
    expected = rate * rate + (1.0 - rate) * (1.0 - rate)
    return (observed - expected) / (1.0 - expected)


_COMPARE = {"auroc": auroc, "kappa": matched_kappa}


def lag_curve(score: np.ndarray, labels: np.ndarray, style: str, max_shift: int) -> np.ndarray:
    """The comparison at each shift ``-max_shift..max_shift``; NaN where it is undefined.

    ``labels`` are raw PSG labels: 0 is wake, a negative value is unscored.
    """
    compare = _COMPARE[style]
    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels)
    n = labels.shape[0]
    curve = np.full(2 * max_shift + 1, np.nan)
    for i, k in enumerate(range(-max_shift, max_shift + 1)):
        t = np.arange(max(0, -k), min(n, n - k))
        shifted = labels[t + k]
        scored = shifted >= 0
        if scored.sum() < 2:
            continue
        curve[i] = compare(shifted[scored] == 0, score[t][scored])
    return curve


def estimate_offset(score: np.ndarray, labels: np.ndarray, settings: AlignmentSettings) -> Offset:
    """The night's offset from its ETC wake ``score`` and raw PSG ``labels``.

    A dropped night has ``shift`` None and says why in ``dropped``.
    """
    m = settings.max_shift
    curve = lag_curve(score, labels, settings.style, m)
    if np.isnan(curve).all():
        return Offset(None, None, float("nan"), float("nan"), float("nan"),
                      dropped="no shift has both wake and sleep to compare")
    best = int(np.nanargmax(curve))
    shift = best - m
    peak = float(curve[best])
    near = np.arange(max(0, best - PEAK_HALF_WIDTH), min(curve.size, best + PEAK_HALF_WIDTH + 1))
    rivals = np.delete(curve, near)
    margin = float(peak - np.nanmax(rivals)) if np.isfinite(rivals).any() else float("inf")
    offset = Offset(shift, shift, peak, float(curve[m]), margin)

    reason = None
    if abs(shift) == m:
        reason = f"best shift {shift:+d} is at the search limit (±{m} epochs)"
    elif peak < settings.peak_floor:
        reason = f"peak {settings.style} {peak:.3f} is below {settings.peak_floor:g}"
    elif margin < settings.min_margin:
        reason = (
            f"peak {settings.style} stands only {margin:.4f} above shifts more than "
            f"{PEAK_HALF_WIDTH} epochs away (needs {settings.min_margin:g})"
        )
    if reason is None:
        return offset
    return Offset(None, shift, offset.peak, offset.at_zero, margin, dropped=reason)


def recording_wake_score(X: np.ndarray, n_epochs: int, settings: AlignmentSettings) -> np.ndarray:
    """The ETC wake score of one cached ``(frames, F, C)`` recording under ``settings``."""
    return wake_score(epoch_energy(X, n_epochs, settings.channel), settings.kernel)


def shift_labels(labels: np.ndarray, shift: int) -> np.ndarray:
    """``labels`` moved by ``shift`` epochs along the last axis; vacated epochs are unscored."""
    labels = np.asarray(labels)
    out = np.full_like(labels, UNSCORED_LABEL)
    if shift > 0:
        out[..., :-shift] = labels[..., shift:]
    elif shift < 0:
        out[..., -shift:] = labels[..., :shift]
    else:
        out[...] = labels
    return out


def _report_row(dataset: str, subject_id: str, offsets: dict[str, Offset], applied: str) -> dict:
    row: dict = {"dataset": dataset, "subject_id": subject_id, "applied_style": applied}
    for style, offset in offsets.items():
        row[f"{style}_shift"] = offset.shift
        row[f"{style}_best_shift"] = offset.best_shift
        row[f"{style}_peak"] = offset.peak
        row[f"{style}_at_zero"] = offset.at_zero
        row[f"{style}_margin"] = offset.margin
        row[f"{style}_dropped"] = offset.dropped or ""
    return row


def align_cache(
    source_dir: Path | str,
    target_dir: Path | str,
    datasets: Sequence[str],
    settings: AlignmentSettings,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[dict]:
    """Write a cache in which each night of ``datasets`` has aligned labels.

    ``source_dir`` and ``target_dir`` are cache directories (a cache root
    joined with its prefix). Every night of ``datasets`` is estimated in both
    :data:`STYLES`, so the returned report rows compare them; ``settings.style``
    is the one applied. An aligned night is written with its labels shifted
    and its shift under :data:`SHIFT_KEY`; a dropped night is not written, and
    is reported on stderr with the reason. Every other cached dataset is
    linked in unchanged, so ``target_dir`` is a complete cache. An
    ``alignment.json`` beside the files records the settings and each night's
    outcome. The source cache is only read.

    ``dry_run`` estimates and reports without writing anything.
    """
    from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR, discover_cached_subjects

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    cached = discover_cached_subjects(datasets, source_dir)

    rows: list[dict] = []
    for dataset in datasets:
        dropped = 0
        for subject in cached[dataset]:
            with np.load(subject.path, allow_pickle=True) as npz:
                arrays = {key: npz[key] for key in npz.files}
            labels = arrays["y"][0]
            score = recording_wake_score(arrays["X"][0], labels.shape[0], settings)
            offsets = {
                style: estimate_offset(score, labels, replace(settings, style=style))
                for style in STYLES
            }
            applied = offsets[settings.style]
            rows.append(_report_row(dataset, subject.subject_id, offsets, settings.style))
            if applied.shift is None:
                dropped += 1
                print(f"DROPPED {dataset}/{subject.subject_id}: {applied.dropped}", file=sys.stderr)
                continue
            if dry_run:
                continue
            destination = target_dir / subject.path.name
            if destination.exists() and not overwrite:
                raise FileExistsError(f"{destination} exists; pass overwrite to replace it")
            arrays["y"] = shift_labels(arrays["y"], applied.shift)
            arrays[SHIFT_KEY] = np.array(applied.shift)
            np.savez_compressed(destination, **arrays)
        total = len(cached[dataset])
        shifts = [r[f"{settings.style}_shift"] for r in rows
                  if r["dataset"] == dataset and r[f"{settings.style}_shift"] is not None]
        median = f", median shift {np.median(shifts):+.0f} epochs" if shifts else ""
        print(f"{dataset}: aligned {total - dropped} of {total} nights, "
              f"dropped {dropped} ({settings.style} style){median}")
        if dropped:
            print(f"WARNING {dataset}: {dropped} of {total} nights dropped; see above",
                  file=sys.stderr)

    if dry_run:
        return rows

    aligned = set(datasets)
    for path in sorted(source_dir.glob(f"*{SUBJECT_SEPARATOR}*{CACHE_SUFFIX}")):
        if path.name.split(SUBJECT_SEPARATOR)[0] in aligned:
            continue
        link = target_dir / path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(path.resolve(), target_dir.resolve()))

    manifest = {
        "source": str(source_dir),
        "datasets": list(datasets),
        "style": settings.style,
        "kernel": {"window": settings.kernel.window, "std": settings.kernel.std,
                   "effective_std": settings.kernel.effective_std},
        "channel": settings.channel,
        "max_shift": settings.max_shift,
        "min_peak": settings.peak_floor,
        "min_margin": settings.min_margin,
        "nights": {
            f"{row['dataset']}/{row['subject_id']}": (
                {"shift": row[f"{settings.style}_shift"]}
                if row[f"{settings.style}_shift"] is not None
                else {"dropped": row[f"{settings.style}_dropped"]}
            )
            for row in rows
        },
    }
    (target_dir / "alignment.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return rows

"""GGIR-style sleep filtering for Autofish predictions.

Autofish -- like any accelerometry sleep model -- reports sleep whenever there
is little movement, so a quiet reading session can look like sleep. GGIR has
solved a version of this: it finds sustained inactivity bouts (SIB) with the
``vanHees2015`` rule (no z-angle change over 5 degrees for 5 minutes) and then
estimates the Sleep Period Time (SPT) window with HDCZA, the main sleep block
of the day. Model sleep that is not a SIB is quiet wakefulness. The SPT window
is informational only: sleep outside it is flagged as a nap, not removed,
because people on split-sleep schedules have several real sleep periods a day.

This module reimplements just those pieces for a raw ``(N, 4)`` accelerometer
array. It is deliberately self-contained: no GGIR, no timezone, no clock. The
only parameters are GGIR's defaults:

- ``anglethreshold = 5`` degrees, ``timethreshold = 5`` minutes (vanHees2015)
- HDCZA: 5-minute rolling median of ``abs(diff(anglez))``, threshold at the
  10th percentile x 15, clipped to ``[0.13, 0.50]``
- 5-second epochs, ``spt_min_block_dur = 30`` min, ``spt_max_gap_dur = 60`` min

Three deviations from GGIR make it work without a clock, on multi-day data,
and as a filter on model sleep:

- The recording is split into sub-recordings wherever there is a lapse of
  ``recording_gap_minutes`` (60) or more with no sample; each is filtered apart.
- GGIR keeps only the single longest no-movement block. We keep the longest
  block in each contiguous 24-hour bin of a sub-recording, so a multi-day
  recording gets one main sleep window per day instead of collapsing to one.
- GGIR's vanHees2015 only marks stillness *between* two posture changes. We
  also mark stillness before the first and after the last change of a
  sub-recording, so a recording that starts or ends in bed keeps its sleep.
- GGIR fills dropouts with the last sample. We leave them empty; for SPT a
  dropout inside a sub-recording keeps its real length and counts as no
  movement only when the epochs on both sides of it do, so it bridges a still
  period without starting or extending one.

SPT is only estimated for a sub-recording spanning at least ``min_span_hours``
(9) hours. A shorter one is assumed to be a sleep bout the researcher chose to
submit, so all of it is marked as SPT (nothing in it is flagged as a nap).

Dropouts are handled differently by the two judgements, on purpose. SIB runs on
the 5 s bins that have data, so a dropout has no duration there: posture
changes either side of a dropout look adjacent, and the dropout does not count
toward the 5-minute stillness requirement. This is acceptable because any
lapse of 60 minutes or more already splits the recording into sub-recordings,
each judged on its own; only shorter dropouts are seen this way. SPT instead
keeps a dropout's real length and bridges it (up to 59 minutes) when SIB-style
stillness is on both sides.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: GGIR's short epoch (``ws3``) and the 5 s grid everything here lives on.
GGIR_EPOCH_SECONDS = 5.0
#: Autofish's epoch: six GGIR bins.
EPOCH_SECONDS = 30.0
#: vanHees2015: no z-angle change over this many degrees ...
SIB_ANGLE_THRESHOLD_DEG = 5.0
#: ... sustained for at least this many minutes.
SIB_TIME_THRESHOLD_MIN = 5.0
#: HDCZA rolling window, in minutes, and percentile x multiplier / clip range.
HDCZA_ROLL_WINDOW_MIN = 5.0
HDCZA_PERCENTILE = 10.0
HDCZA_PERCENTILE_MULTIPLIER = 15.0
HDCZA_THRESHOLD_LO = 0.13
HDCZA_THRESHOLD_HI = 0.50
#: HDCZA morphological steps: drop no-movement blocks this short, fill gaps
#: in no-movement shorter than this.
SPT_MIN_BLOCK_MIN = 30.0
SPT_MAX_GAP_MIN = 60.0
#: A sub-recording must span this long before SPT is worth estimating.
MIN_SPAN_HOURS = 9.0
#: A lapse with no sample at least this long starts a new sub-recording.
RECORDING_GAP_MINUTES = 60.0


@dataclass(frozen=True)
class GGIRResult:
    """Per-epoch GGIR judgement, aligned to Autofish's 30 s epochs.

    ``is_sib`` is the vanHees2015 sustained-inactivity decision; ``is_spt``
    marks the main-sleep window(s). Both are length ``n_epochs`` and ``False``
    for epochs with no data. A sub-recording shorter than ``MIN_SPAN_HOURS`` is
    taken to be a sleep bout the researcher chose to submit, so all of it is
    ``is_spt``; longer ones get the SPT estimate. Epochs in a lapse between
    sub-recordings are not ``is_spt``. ``spt_ran`` records whether any
    sub-recording was long enough for SPT to be estimated.
    """

    is_sib: np.ndarray
    is_spt: np.ndarray
    spt_ran: bool
    spt_windows: list[tuple[float, float]] = field(default_factory=list)


def z_angle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """GGIR's anglez in degrees: ``atan(z / sqrt(x^2 + y^2))``, range -90..90."""
    horizontal = np.sqrt(x * x + y * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.degrees(np.arctan(np.divide(z, horizontal)))
    return np.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median with edge extension, odd ``window``."""
    window = int(window)
    if window < 1 or values.size == 0:
        return values.astype(np.float64)
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strided, axis=-1)


def resample_5s(timestamps_s: np.ndarray, anglez: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool samples onto a 5 s grid anchored at the first sample.

    Returns ``(grid_times_s, anglez_5s)``; grid times are the centre of each
    bin. Bins with no sample are left out, so the grid is not evenly spaced
    across dropouts: always carry ``grid_times_s`` with the values.
    """
    origin = float(timestamps_s[0])
    edges = origin + GGIR_EPOCH_SECONDS * np.arange(
        int(np.floor((timestamps_s[-1] - origin) / GGIR_EPOCH_SECONDS)) + 2
    )
    index = np.searchsorted(edges, timestamps_s, side="right") - 1
    index = np.clip(index, 0, len(edges) - 2)
    n_bins = len(edges) - 1
    sums = np.bincount(index, weights=anglez, minlength=n_bins)[:n_bins]
    counts = np.bincount(index, minlength=n_bins)[:n_bins]
    filled = counts > 0
    centres = edges[:-1][filled] + GGIR_EPOCH_SECONDS / 2.0
    return centres, (sums[filled] / counts[filled]).astype(np.float64)


def sub_recording_bounds(
    timestamps_s: np.ndarray, gap_minutes: float = RECORDING_GAP_MINUTES
) -> list[tuple[int, int]]:
    """Split on lapses of ``gap_minutes`` or more; return ``[start, stop)`` spans.

    Each span is a contiguous run of samples, in sample indices, separated by a
    lapse with no sample of at least ``gap_minutes``.
    """
    n = timestamps_s.size
    if n == 0:
        return []
    lapse = np.diff(timestamps_s) >= gap_minutes * 60.0
    cuts = np.flatnonzero(lapse) + 1
    starts = np.concatenate([[0], cuts])
    stops = np.concatenate([cuts, [n]])
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def van_hees_sib(anglez_5s: np.ndarray) -> np.ndarray:
    """vanHees2015 SIB: 5 s sleep/wake where the angle holds within 5 deg for 5 min.

    As in GGIR's ``HASIB``, a stretch between two posture changes more than 5
    minutes apart is inactivity, both change epochs included; with no such
    stretch, the span is all inactivity under 10 changes and all activity
    otherwise. Unlike GGIR, the span's start and end count as posture changes,
    so stillness before the first and after the last change is inactivity too.
    """
    n = anglez_5s.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    posture_changes = np.flatnonzero(np.abs(np.diff(anglez_5s)) > SIB_ANGLE_THRESHOLD_DEG)
    # Diff index d is the change between bins d and d + 1; -1 and n - 1 stand
    # for the span's edges.
    bounds = np.concatenate([[-1], posture_changes, [n - 1]])
    min_gap = SIB_TIME_THRESHOLD_MIN * 60.0 / GGIR_EPOCH_SECONDS
    quiet = np.flatnonzero(np.diff(bounds) > min_gap)
    if quiet.size == 0:
        return np.full(n, posture_changes.size < 10, dtype=bool)
    sustained = np.zeros(n, dtype=bool)
    for q in quiet:
        sustained[max(bounds[q], 0) : bounds[q + 1] + 1] = True
    return sustained


def hdcza_nomov(anglez_5s: np.ndarray) -> np.ndarray:
    """HDCZA's no-movement mask on the 5 s grid.

    The 5-minute rolling median of ``abs(diff(anglez))`` is compared against
    the 10th percentile x 15, clipped to ``[0.13, 0.50]``.
    """
    n = anglez_5s.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n < 2:
        return np.ones(n, dtype=bool)
    window = int(round(HDCZA_ROLL_WINDOW_MIN * 60.0 / GGIR_EPOCH_SECONDS))
    rolling = rolling_median(np.abs(np.diff(anglez_5s)), window)
    threshold = float(np.quantile(rolling, HDCZA_PERCENTILE / 100.0)) * HDCZA_PERCENTILE_MULTIPLIER
    threshold = min(max(threshold, HDCZA_THRESHOLD_LO), HDCZA_THRESHOLD_HI)
    nomov = np.empty(n, dtype=bool)
    nomov[0] = False
    nomov[1:] = rolling < threshold
    return nomov


def _run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length encode a boolean mask: ``(values, lengths, starts)``."""
    if mask.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty.astype(bool), empty, empty
    change = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [mask.size]])
    return mask[starts], stops - starts, starts


def _consolidate_nomov(nomov: np.ndarray) -> np.ndarray:
    """Drop too-short no-movement blocks, then fill short movement gaps.

    GGIR's ``HASPT`` steps, in order: no-movement blocks of 30 minutes or less
    are removed, and only then are movement gaps shorter than 60 minutes filled
    in what remains. The movement before the first and after the last
    no-movement block is never filled.
    """
    min_block = SPT_MIN_BLOCK_MIN * 60.0 / GGIR_EPOCH_SECONDS
    max_gap = SPT_MAX_GAP_MIN * 60.0 / GGIR_EPOCH_SECONDS
    # GGIR pads with movement at both ends, so the edge runs are always
    # movement and are protected from filling.
    mask = np.concatenate([[False], nomov.astype(bool), [False]])

    values, lengths, _ = _run_lengths(mask)
    values = values & (lengths > min_block)
    mask = np.repeat(values, lengths)

    values, lengths, _ = _run_lengths(mask)
    interior = np.zeros(values.size, dtype=bool)
    interior[1:-1] = True
    values = values | (~values & interior & (lengths < max_gap))
    return np.repeat(values, lengths)[1:-1]


def _longest_block(mask: np.ndarray) -> tuple[int, int] | None:
    """Longest ``True`` run as ``[start, stop)`` sample indices, or ``None``."""
    values, lengths, starts = _run_lengths(mask)
    if not values.any():
        return None
    run = int(np.argmax(np.where(values, lengths, -1)))
    return int(starts[run]), int(starts[run] + lengths[run])


def _day_bins(bin_start_s: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` bin-index spans of consecutive 24 h, from the first bin."""
    if bin_start_s.size == 0:
        return []
    day = np.floor((bin_start_s - bin_start_s[0]) / (24 * 3600.0)).astype(np.int64)
    cuts = np.flatnonzero(np.diff(day)) + 1
    starts = np.concatenate([[0], cuts])
    stops = np.concatenate([cuts, [day.size]])
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _bridge_dropouts(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Missing bins take ``True`` only where the bins on both sides are ``True``.

    ``values`` and ``present`` are on an evenly spaced grid whose first and
    last bins are present, so every missing run has a neighbour on each side.
    """
    out = values & present
    missing, lengths, starts = _run_lengths(~present)
    for is_missing, length, start in zip(missing, lengths, starts, strict=True):
        if is_missing:
            out[start : start + length] = out[start - 1] and out[start + length]
    return out


def _spt_windows(anglez_5s: np.ndarray, bin_start_s: np.ndarray) -> list[tuple[float, float]]:
    """HDCZA main-sleep windows, one per 24 h, as ``[t0, t1)`` seconds.

    No movement is judged on the bins that hold data, then laid on an evenly
    spaced grid so a dropout counts for its real duration; see
    :func:`_bridge_dropouts`.
    """
    position = np.rint((bin_start_s - bin_start_s[0]) / GGIR_EPOCH_SECONDS).astype(np.int64)
    size = int(position[-1]) + 1
    present = np.zeros(size, dtype=bool)
    present[position] = True
    nomov = np.zeros(size, dtype=bool)
    nomov[position] = hdcza_nomov(anglez_5s)
    consolidated = _consolidate_nomov(_bridge_dropouts(nomov, present))

    grid_start_s = bin_start_s[0] + GGIR_EPOCH_SECONDS * np.arange(size)
    windows = []
    for day_start, day_stop in _day_bins(grid_start_s):
        block = _longest_block(consolidated[day_start:day_stop])
        if block is None:
            continue
        lo, hi = day_start + block[0], day_start + block[1]
        windows.append(
            (float(grid_start_s[lo]), float(grid_start_s[hi - 1]) + GGIR_EPOCH_SECONDS)
        )
    return windows


def anglez_bins(accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 5 s mean z-angle series GGIR's judgements are made from.

    ``accel`` is ``(N, 4)``: timestamp seconds then x, y, z in g; non-finite
    rows are dropped, and rows are sorted with one kept per timestamp. Returns
    ``(bin_start_s, anglez_5s)`` for the bins that hold data, anchored at the
    first sample. This is the only part of the filter that needs the raw signal,
    which is why it is split out: the feature cache stores it
    (:func:`epoch_bins`) so the judgements can be made without the raw data.
    """
    accel = np.asarray(accel, dtype=np.float64)
    accel = accel[np.all(np.isfinite(accel), axis=1)]
    if accel.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    accel = accel[np.argsort(accel[:, 0], kind="stable")]
    accel = accel[np.concatenate([[True], np.diff(accel[:, 0]) > 0])]
    centres, anglez_5s = resample_5s(accel[:, 0], z_angle(accel[:, 1], accel[:, 2], accel[:, 3]))
    return centres - GGIR_EPOCH_SECONDS / 2.0, anglez_5s


def epoch_bins(bin_start_s: np.ndarray, anglez_5s: np.ndarray, n_epochs: int) -> np.ndarray:
    """Lay the 5 s series on the 30 s epoch grid: ``(n_epochs, 6)``, ``NaN`` where empty.

    Epoch ``k`` holds bins ``6k..6k+5`` of the grid anchored at the first bin,
    the same anchor the feature cache's epochs use, so no resampling is
    involved. Bins past the last epoch are dropped.
    """
    per_epoch = int(EPOCH_SECONDS / GGIR_EPOCH_SECONDS)
    grid = np.full(n_epochs * per_epoch, np.nan)
    if bin_start_s.size:
        index = np.rint((bin_start_s - bin_start_s[0]) / GGIR_EPOCH_SECONDS).astype(np.int64)
        keep = index < grid.size
        grid[index[keep]] = anglez_5s[keep]
    return grid.reshape(n_epochs, per_epoch)


def bins_from_epochs(epoch_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`epoch_bins`: ``(bin_start_s, anglez_5s)`` with time from 0."""
    flat = np.asarray(epoch_grid, dtype=np.float64).reshape(-1)
    present = np.isfinite(flat)
    return GGIR_EPOCH_SECONDS * np.flatnonzero(present), flat[present]


def evaluate(
    accel: np.ndarray,
    epoch_start_s: np.ndarray,
    n_epochs: int,
) -> GGIRResult:
    """GGIR SIB and SPT judgements aligned to ``n_epochs`` 30 s epochs.

    ``accel`` is ``(N, 4)`` as for :func:`anglez_bins`; see :func:`evaluate_bins`.
    """
    bin_start_s, anglez_5s = anglez_bins(accel)
    return evaluate_bins(bin_start_s, anglez_5s, epoch_start_s, n_epochs)


def evaluate_bins(
    bin_start_s: np.ndarray,
    anglez_5s: np.ndarray,
    epoch_start_s: np.ndarray,
    n_epochs: int,
) -> GGIRResult:
    """GGIR SIB and SPT judgements from the 5 s series, on ``n_epochs`` 30 s epochs.

    Each sub-recording is judged on its own stretch of the series, located by
    time, so a lapse never joins two sub-recordings.
    """
    if bin_start_s.size == 0:
        return GGIRResult(
            is_sib=np.zeros(n_epochs, dtype=bool),
            is_spt=np.zeros(n_epochs, dtype=bool),
            spt_ran=False,
        )

    sib_5s = np.zeros(anglez_5s.size, dtype=bool)
    is_spt = np.zeros(n_epochs, dtype=bool)
    windows: list[tuple[float, float]] = []
    spt_ran = False
    for lo, hi in sub_recording_bounds(bin_start_s):
        first_s = bin_start_s[lo]
        last_s = bin_start_s[hi - 1] + GGIR_EPOCH_SECONDS
        sib_5s[lo:hi] = van_hees_sib(anglez_5s[lo:hi])
        if (last_s - first_s) / 3600.0 < MIN_SPAN_HOURS:
            # Too short to estimate SPT: treat the whole stretch as main sleep.
            is_spt |= (epoch_start_s + EPOCH_SECONDS > first_s) & (epoch_start_s < last_s)
            continue
        spt_ran = True
        for t0, t1 in _spt_windows(anglez_5s[lo:hi], bin_start_s[lo:hi]):
            windows.append((t0, t1))
            is_spt |= (epoch_start_s >= t0) & (epoch_start_s < t1)

    is_sib = _expand_to_epochs(sib_5s, bin_start_s, epoch_start_s, n_epochs)
    return GGIRResult(is_sib=is_sib, is_spt=is_spt, spt_ran=spt_ran, spt_windows=windows)


def _expand_to_epochs(
    values_5s: np.ndarray,
    bin_start_s: np.ndarray,
    epoch_start_s: np.ndarray,
    n_epochs: int,
) -> np.ndarray:
    """Carry a 5 s-grid boolean to 30 s epochs: sleep when any sub-epoch is sleep.

    Each 5 s bin goes to the epoch holding its centre, by time, so bins after a
    dropout land in the right epoch. Epochs with no bin stay ``False``.
    """
    out = np.zeros(n_epochs, dtype=bool)
    if values_5s.size == 0 or n_epochs == 0:
        return out
    epoch_seconds = EPOCH_SECONDS
    centres = bin_start_s + GGIR_EPOCH_SECONDS / 2.0
    epoch = np.floor((centres - epoch_start_s[0]) / epoch_seconds).astype(np.int64)
    inside = (epoch >= 0) & (epoch < n_epochs)
    np.logical_or.at(out, epoch[inside], values_5s[inside])
    return out

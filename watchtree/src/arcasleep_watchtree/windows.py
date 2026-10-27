"""Mean, standard deviation and maximum over backward and forward windows of epochs.

A window is a count of 30-s epochs. A backward window of ``n`` ends at, and includes, the epoch
it is computed for; a forward window of ``n`` starts at the next epoch. NaN epochs (no data) are
left out of the window and a window with nothing in it is NaN.

On the watch a forward window of ``n`` is the backward window of ``n`` at the epoch ``n`` later:
the stager scores epoch ``k`` once epoch ``k + n`` has arrived. :class:`RingWindow` is that
running-sum form; the vectorised functions give the same values for a whole night.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

STATS = ("mean", "std", "max")


def _bounds(n_epochs: int, n: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n_epochs)
    if direction == "b":
        return np.maximum(k - n + 1, 0), k + 1
    return np.minimum(k + 1, n_epochs), np.minimum(k + n + 1, n_epochs)


def window_stat(values: np.ndarray, stat: str, direction: str, n: int) -> np.ndarray:
    """``stat`` of ``values`` over a backward (``"b"``) or forward (``"f"``) window of epochs."""
    values = np.asarray(values, dtype=np.float64)
    n_epochs = values.size
    lo, hi = _bounds(n_epochs, n, direction)
    if stat == "max":
        pad = np.full(n, np.nan)
        if direction == "b":
            windows = sliding_window_view(np.concatenate([pad[:-1], values]), n)
        else:
            windows = sliding_window_view(np.concatenate([values, pad]), n)[1 : n_epochs + 1]
        return np.fmax.reduce(windows, axis=1)
    present = np.isfinite(values)
    filled = np.where(present, values, 0.0)

    def total(x: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(x)])
        return c[hi] - c[lo]

    count = total(present.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total(filled) / count
        if stat == "mean":
            return np.where(count > 0, mean, np.nan)
        var = total(filled * filled) / count - mean * mean
    return np.where(count > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)


def window_count(present: np.ndarray, direction: str, n: int) -> np.ndarray:
    """Fraction of the epochs inside the recording in the window that have data."""
    n_epochs = present.size
    lo, hi = _bounds(n_epochs, n, direction)
    c = np.concatenate([[0.0], np.cumsum(np.asarray(present, dtype=np.float64))])
    length = hi - lo
    return np.where(length > 0, (c[hi] - c[lo]) / np.maximum(length, 1), np.nan)


class RingWindow:
    """A backward window kept as a ring buffer with running sums, as the watch holds it."""

    def __init__(self, n: int):
        self.n = n
        self.buffer: deque[float] = deque()
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def push(self, value: float) -> None:
        self.buffer.append(value)
        if np.isfinite(value):
            self.count += 1
            self.total += value
            self.total_sq += value * value
        if len(self.buffer) > self.n:
            old = self.buffer.popleft()
            if np.isfinite(old):
                self.count -= 1
                self.total -= old
                self.total_sq -= old * old

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def std(self) -> float:
        if self.count < 2:
            return float("nan")
        mean = self.total / self.count
        return float(np.sqrt(max(self.total_sq / self.count - mean * mean, 0.0)))

    def max(self) -> float:
        finite = [v for v in self.buffer if np.isfinite(v)]
        return max(finite) if finite else float("nan")

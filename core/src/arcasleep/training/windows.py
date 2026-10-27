"""Training windows drawn at random offsets from long uncropped recordings.

A fixed-length window (8 h is 960 epochs) is cut from a recording at an offset
the sampler draws, so the model sees whole nights, sleep onsets with the wake
before them, wake-ups with the wake after them, and stretches with no PSG at
all. Four kinds of window are drawn, in the proportions of a :class:`WindowMix`:

- ``night``: the window covers the PSG span when the span fits in it, and lies
  inside the span otherwise, at a random offset;
- ``edge``: the window contains the first or the last epoch of the PSG span,
  well clear of the window's own ends;
- ``day``: the window has no PSG epoch;
- ``uniform``: any offset.

A window with no supervised epoch (a label of at least 0 at a weight above 0)
is redrawn; a kind that cannot produce one falls back to ``night``. A window
that runs past the recording's end, or a recording shorter than the window, is
padded with absent epochs and the gap label, as batches are elsewhere.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from arcasleep.data.batching import pad_recording_input
from arcasleep.data.constants import GAP_LABEL
from arcasleep.data.weak_labels import LabelledRecording

WINDOW_KINDS = ("night", "edge", "day", "uniform")
#: Redraws of a window's offset before its kind falls back to ``night``.
WINDOW_MAX_ATTEMPTS = 100
#: An ``edge`` window keeps the PSG edge at least this share of the window from either end.
EDGE_MARGIN_FRACTION = 0.125


@dataclass(frozen=True)
class WindowMix:
    """Relative frequency of each kind of window; the four are normalised to sum to 1."""

    night: float = 0.4
    edge: float = 0.3
    day: float = 0.2
    uniform: float = 0.1

    def __post_init__(self) -> None:
        if min(self.night, self.edge, self.day, self.uniform) < 0:
            raise ValueError("window mix weights must be nonnegative")
        if self.night + self.edge + self.day + self.uniform <= 0:
            raise ValueError("window mix weights must not all be 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "WindowMix":
        unknown = set(data) - set(WINDOW_KINDS)
        if unknown:
            raise ValueError(f"window_mix: unknown keys {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in data.items()})

    def probabilities(self) -> np.ndarray:
        weights = np.array([getattr(self, kind) for kind in WINDOW_KINDS], dtype=np.float64)
        return weights / weights.sum()


@dataclass(frozen=True)
class WindowRecording:
    """One recording's model input and targets over its whole length.

    ``x`` is the :func:`~arcasleep.data.batching.recording_input` of the
    recording: ``(epochs, F + 1, C)`` pooled, or ``(epochs * frames_per_epoch,
    F, C)`` frames. ``targets`` has one entry per epoch.
    """

    x: np.ndarray
    targets: LabelledRecording

    def __post_init__(self) -> None:
        if self.targets.labels.shape[0] == 0:
            raise ValueError("an empty recording has no window")
        if not self.supervised().any():
            raise ValueError("the recording has no supervised epoch")

    @property
    def epochs(self) -> int:
        return int(self.targets.labels.shape[0])

    def supervised(self) -> np.ndarray:
        return (self.targets.labels >= 0) & (self.targets.weights > 0)

    def psg_span(self) -> tuple[int, int]:
        """First epoch of the PSG span and one past its last."""
        inside = np.flatnonzero(self.targets.in_psg)
        return int(inside[0]), int(inside[-1]) + 1


@dataclass(frozen=True)
class WindowBatch:
    """Windows stacked on a leading axis; ``kinds`` and ``starts`` say how each was drawn."""

    x: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    sleep_wake: np.ndarray
    kinds: tuple[str, ...]
    starts: np.ndarray
    recordings: np.ndarray


class WindowSampler:
    """Draws fixed-length windows from a list of :class:`WindowRecording`."""

    def __init__(
        self,
        recordings: Sequence[WindowRecording],
        window_epochs: int,
        mix: WindowMix = WindowMix(),
        *,
        frames_per_epoch: int = 1,
        windows_per_recording: int = 1,
        max_attempts: int = WINDOW_MAX_ATTEMPTS,
    ) -> None:
        if not recordings:
            raise ValueError("no recordings to sample windows from")
        if window_epochs < 1 or windows_per_recording < 1 or max_attempts < 1:
            raise ValueError("window_epochs, windows_per_recording and max_attempts must be at least 1")
        self.recordings = list(recordings)
        self.window_epochs = int(window_epochs)
        self.mix = mix
        self.frames_per_epoch = int(frames_per_epoch)
        self.windows_per_recording = int(windows_per_recording)
        self.max_attempts = int(max_attempts)
        self._supervised_cumulative = [
            np.concatenate([[0], np.cumsum(r.supervised())]) for r in self.recordings
        ]

    @property
    def windows_per_epoch(self) -> int:
        """Windows one training epoch draws: ``windows_per_recording`` per recording."""
        return len(self.recordings) * self.windows_per_recording

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Shape of one window's ``x``."""
        return (self.window_epochs * self.frames_per_epoch, *self.recordings[0].x.shape[1:])

    def sample(self, rng: np.random.Generator, n_windows: int | None = None) -> WindowBatch:
        """``n_windows`` windows (default one training epoch's), recordings in shuffled rounds.

        Each round visits every recording once in a random order, so no
        recording is drawn twice before every other has been drawn once.
        """
        n_windows = self.windows_per_epoch if n_windows is None else int(n_windows)
        count = len(self.recordings)
        rounds = -(-n_windows // count)
        order = np.concatenate([rng.permutation(count) for _ in range(rounds)])[:n_windows]
        kinds = rng.choice(len(WINDOW_KINDS), size=n_windows, p=self.mix.probabilities())
        drawn = [self._draw(int(order[i]), WINDOW_KINDS[kinds[i]], rng) for i in range(n_windows)]
        return self._stack(order, drawn)

    def centered(self) -> WindowBatch:
        """One ``night`` window per recording, centred on its PSG span; no randomness."""
        starts = []
        for r, rec in enumerate(self.recordings):
            first, last = rec.psg_span()
            starts.append(self._clip(r, (first + last - self.window_epochs) // 2))
        order = np.arange(len(self.recordings))
        return self._stack(order, [(s, "night") for s in starts])

    def _clip(self, r: int, start: int) -> int:
        return int(min(max(start, 0), max(self.recordings[r].epochs - self.window_epochs, 0)))

    def _has_supervision(self, r: int, start: int) -> bool:
        cumulative = self._supervised_cumulative[r]
        end = min(start + self.window_epochs, self.recordings[r].epochs)
        return bool(cumulative[end] - cumulative[start] > 0)

    def _propose(self, r: int, kind: str, rng: np.random.Generator) -> int | None:
        """One offset of ``kind`` in recording ``r``, ``None`` when the kind has none."""
        rec, window = self.recordings[r], self.window_epochs
        latest = max(rec.epochs - window, 0)
        first, last = rec.psg_span()
        if kind == "uniform":
            return int(rng.integers(latest + 1))
        if kind == "night":
            low, high = (last - window, first) if last - first <= window else (first, last - window)
            return self._clip(r, int(rng.integers(low, high + 1)))
        if kind == "edge":
            edges = [e for e in (first, last) if 0 < e < rec.epochs]
            if not edges:
                return None
            edge = edges[int(rng.integers(len(edges)))]
            margin = int(window * EDGE_MARGIN_FRACTION)
            return self._clip(r, edge - int(rng.integers(margin, window - margin + 1)))
        before = max(min(first - window, latest) + 1, 0)
        after = max(latest - last + 1, 0)
        if before + after == 0:
            return None
        pick = int(rng.integers(before + after))
        return pick if pick < before else last + (pick - before)

    def _draw(self, r: int, kind: str, rng: np.random.Generator) -> tuple[int, str]:
        for _ in range(self.max_attempts):
            start = self._propose(r, kind, rng)
            if start is None:
                break
            if self._has_supervision(r, start):
                return start, kind
        for _ in range(self.max_attempts):
            start = self._propose(r, "night", rng)
            if self._has_supervision(r, start):
                return start, "night"
        raise ValueError("no window with a supervised epoch")

    def _stack(self, order: np.ndarray, drawn: Sequence[tuple[int, str]]) -> WindowBatch:
        window, fpe = self.window_epochs, self.frames_per_epoch
        xs, labels, weights, sleep_wake = [], [], [], []
        for r, (start, _) in zip(order, drawn):
            rec = self.recordings[int(r)]
            stop = min(start + window, rec.epochs)
            pad = window - (stop - start)
            xs.append(pad_recording_input(rec.x[start * fpe : stop * fpe], window, fpe))
            t = rec.targets
            labels.append(np.concatenate([t.labels[start:stop], np.full(pad, GAP_LABEL)]))
            weights.append(np.concatenate([t.weights[start:stop], np.ones(pad)]))
            sleep_wake.append(np.concatenate([t.sleep_wake[start:stop], np.zeros(pad, dtype=bool)]))
        return WindowBatch(
            x=np.stack(xs).astype(np.float32),
            labels=np.stack(labels).astype(np.int32),
            weights=np.stack(weights).astype(np.float32),
            sleep_wake=np.stack(sleep_wake),
            kinds=tuple(kind for _, kind in drawn),
            starts=np.array([s for s, _ in drawn], dtype=np.int64),
            recordings=np.asarray(order, dtype=np.int64),
        )

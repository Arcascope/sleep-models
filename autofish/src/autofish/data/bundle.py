"""The feature-cache bundle: pisces-lite's, plus the inputs the GGIR filter needs.

The spectrograms are detrended magnitude and jerk, so a cached night has lost
its orientation and the GGIR sustained-inactivity judgement cannot be made from
it. ``anglez_5s`` carries what it needs: GGIR's 5 s mean z-angle, laid out one
row per 30 s epoch (:func:`autofish.ggir.epoch_bins`).

It lives beside ``X`` rather than in it. Epochs are the one unit the 2 s
spectrogram hop and the 5 s GGIR bin share (15 frames, 6 bins), so a per-epoch
array needs no resampling, whereas rows on the frame axis would change the
model's input shape. pisces-lite's own reader ignores the extra key, so
training and older tools read these files unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pisces_lite.model_io import ModelIOBundle

ANGLEZ_KEY = "anglez_5s"


@dataclass
class AutofishBundle(ModelIOBundle):
    """A :class:`ModelIOBundle` with an optional ``(B, epochs, 6)`` ``anglez_5s``.

    ``NaN`` marks a 5 s bin with no accelerometer samples. ``None`` means the
    cache predates the array (see ``autofish process --sib-only``).
    """

    anglez_5s: Optional[np.ndarray] = None

    @classmethod
    def from_npz(cls, npz_path: "Path | str") -> "AutofishBundle":
        base = ModelIOBundle.from_npz(npz_path)
        with np.load(npz_path, allow_pickle=True) as data:
            anglez = data[ANGLEZ_KEY] if ANGLEZ_KEY in data.files else None
        return cls(
            X=base.X,
            y=base.y,
            datasets=base.datasets,
            subject_ids=base.subject_ids,
            time=base.time,
            freq=base.freq,
            anglez_5s=anglez,
        )

    def save_npz(self, npz_path: "Path | str") -> None:
        extra = {} if self.anglez_5s is None else {ANGLEZ_KEY: self.anglez_5s}
        np.savez_compressed(
            npz_path,
            X=self.X,
            y=self.y,
            datasets=self.datasets,
            subject_ids=self.subject_ids,
            time=self.time if self.time is not None else np.array([]),
            freq=self.freq if self.freq is not None else np.array([]),
            **extra,
        )

"""ETC ("easy to classify"): a parameter-free wake score from spectral energy.

The ETC score of an epoch is the recording's spectrogram energy -- the sum
over frequency of one channel, averaged over the epoch's frames -- smoothed
along time by a Gaussian kernel. Wake is high energy. Nothing is fitted, so the
score does not depend on any training set; the kernel's two settings are the
only choices, and :class:`ETCKernel` exposes both.

This follows pisces-lite's ``ETCModel`` with the kernel's standard deviation
set directly: ``ETCModel(n_timestamps=w, kernel_width=k)`` is
``ETCKernel(window=w, std=floor(k * w / 2))``, with the score measured in
epochs rather than spectrogram rows.

The kernel is sampled over ``window`` epochs and truncated there, so once
``window`` covers about six standard deviations, larger windows change
nothing; a short window makes the kernel effectively narrower than ``std``
(:meth:`ETCKernel.effective_std`). ``window=1`` is no smoothing at all.

``window`` must be odd. An even kernel has no centre epoch, which moves the
score by half an epoch; for alignment that is a bias of the size being
measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arcasleep.data.constants import SPECTROGRAM_PADDING_VALUE

#: Spectrogram channel the energy is read from; 0 is the magnitude channel.
DEFAULT_CHANNEL = 0


@dataclass(frozen=True)
class ETCKernel:
    """A Gaussian over ``window`` epochs with standard deviation ``std`` epochs.

    The defaults (3 epochs, SD 0.5, effective SD 0.46) aligned the most nights
    in a sweep over Weaver, SleepAccel and SleepAccel-Clinical; at an effective
    SD above 1 epoch the score's peaks are wider than the offsets being
    measured.
    """

    window: int = 3
    std: float = 0.5

    def __post_init__(self) -> None:
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError(f"window must be an odd number of epochs, got {self.window}")
        if self.std <= 0:
            raise ValueError(f"std must be positive, got {self.std}")

    @property
    def weights(self) -> np.ndarray:
        """The kernel, centred at ``window // 2`` and summing to 1."""
        idx = np.arange(self.window, dtype=float) - self.window // 2
        weights = np.exp(-0.5 * (idx / self.std) ** 2)
        return weights / weights.sum()

    @property
    def effective_std(self) -> float:
        """Standard deviation of the truncated kernel, in epochs."""
        idx = np.arange(self.window, dtype=float) - self.window // 2
        weights = self.weights
        mean = float(np.sum(weights * idx))
        return float(np.sqrt(np.sum(weights * (idx - mean) ** 2)))


def epoch_energy(X: np.ndarray, n_epochs: int, channel: int = DEFAULT_CHANNEL) -> np.ndarray:
    """``(n_epochs,)`` spectral energy per epoch from a cached ``(frames, F, C)`` spectrogram.

    Frames with no data are left out of their epoch's mean; an epoch with no
    data at all takes the recording's median energy, so it neither looks like
    wake nor like sleep.
    """
    X = np.asarray(X, dtype=np.float64)
    frames_per_epoch = X.shape[0] // n_epochs
    frames = X[: n_epochs * frames_per_epoch]
    missing = np.all(frames == SPECTROGRAM_PADDING_VALUE, axis=(1, 2))
    energy = frames[:, :, channel].sum(axis=1)
    energy[missing] = np.nan
    per_epoch = energy.reshape(n_epochs, frames_per_epoch)
    present = np.isfinite(per_epoch)
    counts = present.sum(axis=1)
    sums = np.where(present, per_epoch, 0.0).sum(axis=1)
    out = np.full(n_epochs, np.nan)
    out[counts > 0] = sums[counts > 0] / counts[counts > 0]
    if np.isnan(out).all():
        return np.zeros(n_epochs)
    out[np.isnan(out)] = np.nanmedian(out)
    return out


DEFAULT_KERNEL = ETCKernel()


def wake_score(energy: np.ndarray, kernel: ETCKernel = DEFAULT_KERNEL) -> np.ndarray:
    """The ETC wake score: ``energy`` smoothed by ``kernel``, same length as ``energy``."""
    energy = np.asarray(energy, dtype=np.float64)
    weights = kernel.weights
    # Reading the full convolution from the kernel's centre keeps the score in
    # register with the energy for even windows too, and returns len(energy)
    # samples also when the recording is shorter than the kernel.
    centre = kernel.window // 2
    return np.convolve(energy, weights, mode="full")[centre : centre + energy.shape[0]]

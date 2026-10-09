"""The release's NUFFT input as one JAX function of the raw samples, for noise added inside a step.

The release computes its frames on the host: ``|a|`` and senpy's jerk from the cleaned samples,
then ``senpy.jax_backend.compute_nustft_many`` (host packing, a jitted transform, a copy back),
then ``log(|X|² + ε)`` and the 0.1–6 Hz crop. Only the transform runs on the device, and it
cannot sit inside a jitted training step.

Here the work is split by what it depends on:

- ``plan(t, origin)`` (host, NumPy, once per recording): which samples fall in each window, their
  local NUFFT coordinates and the density scale. These depend only on the timestamps, which noise
  does not change. The window grid is senpy's own (``_packing.discover``), with every window
  padded to one width so a recording is one static shape.
- ``frames(plan, xyz)`` (JAX, jittable, differentiable): ``|a|``, the jerk, per-window detrend,
  Hann taper, type-1 NUFFT (senpy's ``nufft1``), density scaling, log-PSD and the bin crop.
  ``frames_from_signal`` takes ``channels(xyz)`` instead, so a caller can compute it once.

With ``taps`` from ``lowpass_taps``, each axis is low-passed at ``LOWPASS_HZ`` before the jerk
(``|a|`` is unchanged). The jerk's norm rectifies, so without it vibration above 6 Hz reaches the
0.1–6 Hz band as sum and difference frequencies folded at the sample rate. This is not the
release's input; a model trained with it must be scored with it.

senpy's jerk with ``use_diff=True`` is ``‖a[i] − a[i−1]‖`` (not divided by the time step), with
0 for the first sample, on the same timestamps as the samples. This was checked against
``senpy.compute_jerk`` on irregular timestamps.

The NUFFT is evaluated for ``N_MODES`` modes rather than the recording's ``nfft_padded``: a
type-1 NUFFT's value at mode k does not depend on how many modes are requested, so the kept bins
(k/window_s for k ≤ fmax·window_s) are the same, and recordings at every sampling rate share one
transform size.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import firwin
from senpy import _packing
from senpy._grid import DEFAULT_MIN_SAMPLES
from senpy.jax_backend import nufft1

WINDOW_S = 10.0
STEP_S = 2.0
FMIN, FMAX = 0.1, 6.0
LOG_EPS = float(np.finfo(np.float64).eps)  # pisces-lite's log-PSD floor
NUFFT_EPS = 1e-6
N_MODES = 128
PADDING_VALUE = -1.0  # pisces-lite's SPECTROGRAM_PADDING_VALUE: a frame with no window behind it
LOWPASS_HZ = 6.0
LOWPASS_S = 2.0  # FIR length; a Hamming window this long has a transition about 1.6 Hz wide
MAX_TAPS = 201  # LOWPASS_S at up to 100 Hz; faster recordings get a shorter filter


@dataclass(frozen=True)
class Plan:
    index: np.ndarray  # [W, L] sample index per lane; padded lanes point at a zero sample (N)
    points: np.ndarray  # [W, L] local NUFFT coordinates in [-pi, pi)
    valid: np.ndarray  # [W, L] lane holds a real sample
    row_valid: np.ndarray  # [W] window has enough samples (else its frame holds PADDING_VALUE)
    median_fs: float
    origin_s: float


def plan(t: np.ndarray, origin_s: float, width: int | None = None) -> Plan:
    """Window layout of one recording on senpy's grid (``empty_windows="keep"``)."""
    rec = _packing.discover(t, np.zeros((t.size, 1)), window_s=WINDOW_S, overlap_s=WINDOW_S - STEP_S,
                            ts_unit="s", origin_s=origin_s, min_samples=DEFAULT_MIN_SAMPLES)
    grid = rec.grid
    windows = np.arange(grid.n_windows)
    counts = grid.sample_count[windows]
    width = int(width or _packing.next_power_of_two(np.array([max(int(counts.max()), 2)]))[0])
    lane = np.arange(width)
    mask = (lane[None, :] < counts[:, None]) & grid.valid[:, None]
    index = np.where(mask, grid.first[windows][:, None] + lane[None, :], rec.n_samples)
    local = 2.0 * np.pi * ((rec.t[index] - windows[:, None] * STEP_S) / WINDOW_S) - np.pi
    local[~mask] = 0.0
    return Plan(index=index.astype(np.int32), points=local.astype(np.float32), valid=mask,
                row_valid=grid.valid.copy(), median_fs=float(rec.median_fs), origin_s=float(grid.origin_s))


def safe_norm(v: jax.Array) -> jax.Array:
    """Row norms whose gradient is 0, not NaN, where a row is all zeros (repeated samples)."""
    s = jnp.sum(v**2, axis=1)
    return jnp.where(s > 0, jnp.sqrt(jnp.where(s > 0, s, 1.0)), 0.0)


def lowpass_taps(fs: float) -> np.ndarray:
    """``[MAX_TAPS]`` zero-phase FIR taps, centred; a unit impulse when ``fs`` is too low to filter at ``LOWPASS_HZ``."""
    out = np.zeros(MAX_TAPS, np.float32)
    centre = MAX_TAPS // 2
    if fs / 2 <= 1.1 * LOWPASS_HZ:
        out[centre] = 1.0
        return out
    n = min(2 * int(round(fs * LOWPASS_S / 2)) + 1, MAX_TAPS)
    out[centre - n // 2:centre + n // 2 + 1] = firwin(n, LOWPASS_HZ, fs=fs)
    return out


def channels(xyz: jax.Array, taps: jax.Array | None = None) -> jax.Array:
    """``[N, 3]`` samples to ``[N, 2]``: ``|a|`` and senpy's jerk, of the low-passed axes if ``taps``."""
    smooth = xyz
    if taps is not None:
        smooth = jax.vmap(lambda c: jnp.convolve(c, taps, mode="same", precision=jax.lax.Precision.HIGHEST),
                          in_axes=1, out_axes=1)(xyz)
    jerk = jnp.concatenate([jnp.zeros(1, xyz.dtype), safe_norm(smooth[1:] - smooth[:-1])])
    return jnp.stack([safe_norm(xyz), jerk], axis=1)


def _bins() -> np.ndarray:
    k = np.arange(N_MODES // 2)
    f = k / WINDOW_S
    return k[(f >= FMIN - 1e-9) & (f <= FMAX + 1e-9)]


BINS = _bins()


@jax.jit
def frames(index: jax.Array, points: jax.Array, valid: jax.Array, row_valid: jax.Array, median_fs: jax.Array,
           xyz: jax.Array, taps: jax.Array | None = None) -> jax.Array:
    """``[W, bins, 2]`` log-PSD frames of ``|a|`` and jerk; windows without enough samples hold ``PADDING_VALUE``."""
    return frames_from_signal(index, points, valid, row_valid, median_fs, channels(xyz, taps))


def frames_from_signal(index: jax.Array, points: jax.Array, valid: jax.Array, row_valid: jax.Array,
                       median_fs: jax.Array, signal: jax.Array) -> jax.Array:
    """``frames`` from ``channels(xyz)``."""
    signal = jnp.concatenate([signal, jnp.zeros((1, 2), signal.dtype)])  # trailing zero for padding
    lanes = jnp.moveaxis(signal[index], -1, 1)  # [W, 2, L]
    v = valid.astype(lanes.dtype)
    count = jnp.maximum(v.sum(axis=1), 1.0)
    mean = (lanes * v[:, None, :]).sum(axis=2) / count[:, None]
    tau = (points + jnp.pi) / (2.0 * jnp.pi)
    hann = v * 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau))
    strengths = ((lanes - mean[:, :, None]) * hann[:, None, :]).astype(jnp.complex64)
    modes = nufft1(N_MODES, strengths, points, eps=NUFFT_EPS, iflag=1)  # [W, 2, N_MODES], k = -64..63
    coefficients = modes[..., N_MODES // 2 + BINS]
    ss = jnp.where(row_valid, (hann**2).sum(axis=1), 1.0)
    power = jnp.abs(coefficients) ** 2 / (median_fs * ss)[:, None, None]
    out = jnp.log(power + LOG_EPS)
    out = jnp.where(row_valid[:, None, None], out, PADDING_VALUE)
    return jnp.moveaxis(out, 1, 2)


def device_plan(p: Plan) -> tuple:
    return (jnp.asarray(p.index), jnp.asarray(p.points), jnp.asarray(p.valid), jnp.asarray(p.row_valid),
            jnp.asarray(p.median_fs, jnp.float32))

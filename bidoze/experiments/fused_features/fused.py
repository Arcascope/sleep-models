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

senpy's jerk with ``use_diff=True`` is ``‖a[i] − a[i−1]‖`` (not divided by the time step), with
0 for the first sample, on the same timestamps as the samples. This was checked against
``senpy.compute_jerk`` on irregular timestamps. ``plan(..., jerk=...)`` sets a per-sample factor
the jerk is multiplied by:

- ``release``: 1, the release's jerk.
- ``dt``: ``1 / (t[i] − t[i−1])``, the backward-difference jerk per second (senpy's
  ``divide_by_dt=True``).
- ``median_dt``: ``1 / median(Δt)``, one constant per recording.

Time steps are taken as senpy takes them, from timestamps truncated to whole microseconds.

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


@dataclass(frozen=True)
class Plan:
    index: np.ndarray  # [W, L] sample index per lane; padded lanes point at a zero sample (N)
    points: np.ndarray  # [W, L] local NUFFT coordinates in [-pi, pi)
    valid: np.ndarray  # [W, L] lane holds a real sample
    row_valid: np.ndarray  # [W] window has enough samples (else its frame holds PADDING_VALUE)
    median_fs: float
    origin_s: float
    jerk_scale: np.ndarray  # [N] factor the jerk of each sample is multiplied by


JERK_MODES = ("release", "dt", "median_dt")


def jerk_scale(t: np.ndarray, mode: str) -> np.ndarray:
    """Per-sample factor for ``mode``; the first sample's jerk is 0, so its factor is 1."""
    if mode not in JERK_MODES:
        raise ValueError(f"jerk must be one of {JERK_MODES}, got {mode!r}")
    step = np.diff((np.asarray(t, np.float64) * 1e6).astype(np.int64)) * 1e-6
    if mode == "release":
        scale = np.ones_like(step)
    elif mode == "median_dt":
        scale = np.full_like(step, 1.0 / np.median(step))
    else:
        scale = np.where(step > 0, 1.0 / np.where(step > 0, step, 1.0), 0.0)
    return np.concatenate([[1.0], scale])


def plan(t: np.ndarray, origin_s: float, width: int | None = None, jerk: str = "release") -> Plan:
    """Window layout of one recording on senpy's grid (``empty_windows="keep"``), and its jerk scale."""
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
                row_valid=grid.valid.copy(), median_fs=float(rec.median_fs), origin_s=float(grid.origin_s),
                jerk_scale=jerk_scale(t, jerk).astype(np.float32))


def safe_norm(v: jax.Array) -> jax.Array:
    """Row norms whose gradient is 0, not NaN, where a row is all zeros (repeated samples)."""
    s = jnp.sum(v**2, axis=1)
    return jnp.where(s > 0, jnp.sqrt(jnp.where(s > 0, s, 1.0)), 0.0)


def channels(xyz: jax.Array, scale: jax.Array) -> jax.Array:
    """``[N, 3]`` samples to ``[N, 2]``: ``|a|`` and senpy's jerk times ``scale``."""
    jerk = jnp.concatenate([jnp.zeros(1, xyz.dtype), safe_norm(xyz[1:] - xyz[:-1])])
    return jnp.stack([safe_norm(xyz), jerk * scale], axis=1)


def _bins() -> np.ndarray:
    k = np.arange(N_MODES // 2)
    f = k / WINDOW_S
    return k[(f >= FMIN - 1e-9) & (f <= FMAX + 1e-9)]


BINS = _bins()


@jax.jit
def frames(index: jax.Array, points: jax.Array, valid: jax.Array, row_valid: jax.Array, median_fs: jax.Array,
           jerk_scale: jax.Array, xyz: jax.Array) -> jax.Array:
    """``[W, bins, 2]`` log-PSD frames of ``|a|`` and jerk; windows without enough samples hold ``PADDING_VALUE``."""
    signal = jnp.concatenate([channels(xyz, jerk_scale.astype(xyz.dtype)), jnp.zeros((1, 2), xyz.dtype)])  # trailing zero for padding
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
            jnp.asarray(p.median_fs, jnp.float32), jnp.asarray(p.jerk_scale))

"""Raw night → random boat-like noise → the release's model input → BiDoze, in one jitted step.

The balanced recipe's training step (``BiDozeNetModel.fit_balanced``) reads cached, pooled
spectrograms. Here every step builds them from the raw samples on the device, so each step can
add a fresh noise draw (``jax_generator``) and no noisy cache is ever written.

Host, once per night (``load_night``, from the timestamps only): senpy's window grid as each
window's first sample and sample count, and each sample's time as a 2-s grid cell plus a
fraction, which keeps the NUFFT coordinates exact in float32. Device, every step:

1. noise on the raw ``x, y, z``: ``"boat"``, a ``jax_generator`` draw; ``"floor"``, white Gaussian
   noise per axis and sample at a per-night level drawn from ``FLOOR_MG`` (the Galaxy Watch floor,
   §6 and §9 of the Oct 9 log, is about 3.7 mg); ``"none"``;
2. frames: ``fused.channels`` (with the jerk low-pass if ``lowpass``), the window lanes from the
   grid, then ``fused.frames_from_signal`` (NUFFT, log-PSD);
3. pooling: ``arcasleep.data.batching``'s pooling to one row per epoch with the gap row, empty
   epochs filled with the night's median spectrum, padded to ``EPOCHS`` epochs;
4. the class-balanced cross-entropy and AdamW update of ``fit_balanced``.

Nights are stacked on the device at one length (``n_total`` samples, ``data_epochs`` epochs of
frames, ``width`` lanes per window), so one compiled step serves every batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pisces_lite.datasets.constants import PSG_5C_MAPPING_TO_WLDR
from senpy import _packing
from senpy._grid import DEFAULT_MIN_SAMPLES

import fused
import jax_generator
from arcasleep.data.constants import GAP_LABEL
from arcasleep.data.prepare import prepare_subject
from arcasleep.training.balanced import balanced_cross_entropy

FRAMES_PER_EPOCH = int(30.0 / fused.STEP_S)
# Frames are computed this many epochs at a time, and nights one at a time, to bound the
# NUFFT's spreading memory; a stack's data epochs are a multiple of it.
CHUNK_EPOCHS = 128
# The model input's length: whole chunks, so a night of up to EPOCHS epochs never rounds past it.
EPOCHS = 24 * CHUNK_EPOCHS
FLOOR_MG = (2.5, 5.0)


@dataclass
class Night:
    """One night's host arrays, before padding to the stack's shapes."""

    name: str
    xyz: np.ndarray  # [n, 3] float32, g
    t: np.ndarray  # [n] float32, seconds from the grid origin (for the noise gates)
    cell: np.ndarray  # [n] int32, floor(t / STEP_S)
    frac: np.ndarray  # [n] float32, t − STEP_S · cell
    first: np.ndarray  # [frames] int32
    count: np.ndarray  # [frames] int32, 0 for a window without enough samples or labels
    fs: float
    y: np.ndarray  # [epochs] int32, 4-class labels, GAP_LABEL where there is no data
    taps: np.ndarray  # [fused.MAX_TAPS] float32, the jerk low-pass at this night's rate


def load_night(data_set, subject_id: str) -> Night:
    """A night as the training cache builds it: cleaned, aligned, labels on the frames' epochs."""
    prepared = prepare_subject(data_set, subject_id)
    accel = prepared.accel_values
    origin = float(accel[0, 0])
    rec = _packing.discover(accel[:, 0], np.zeros((accel.shape[0], 1)), window_s=fused.WINDOW_S,
                            overlap_s=fused.WINDOW_S - fused.STEP_S, ts_unit="s", origin_s=origin,
                            min_samples=DEFAULT_MIN_SAMPLES)
    grid = rec.grid
    if rec.n_samples != accel.shape[0]:
        raise ValueError(f"{subject_id}: senpy dropped samples ({rec.n_samples} of {accel.shape[0]})")
    n_epochs = min(grid.n_windows // FRAMES_PER_EPOCH, prepared.labels.shape[0], EPOCHS)
    n_frames = n_epochs * FRAMES_PER_EPOCH
    count = np.where(grid.valid, grid.sample_count, 0)[:n_frames]
    cell = np.floor(rec.t / fused.STEP_S)
    labels = prepared.labels[:n_epochs]
    y = np.vectorize(lambda v: PSG_5C_MAPPING_TO_WLDR.get(v, v), otypes=[np.int32])(labels)
    return Night(name=f"{data_set.name}/{subject_id}", xyz=accel[:, 1:].astype(np.float32),
                 t=rec.t.astype(np.float32), cell=cell.astype(np.int32),
                 frac=(rec.t - fused.STEP_S * cell).astype(np.float32),
                 first=grid.first[:n_frames].astype(np.int32), count=count.astype(np.int32),
                 fs=float(rec.median_fs), y=y, taps=fused.lowpass_taps(float(rec.median_fs)))


def stack(nights: list[Night], n_total: int | None = None, data_epochs: int | None = None,
          width: int | None = None) -> dict:
    """Pad every night to one shape and put the stack on the device."""
    n_total = n_total or int(max(len(n.t) for n in nights))
    data_epochs = data_epochs or int(max(len(n.y) for n in nights))
    data_epochs = -(-data_epochs // CHUNK_EPOCHS) * CHUNK_EPOCHS
    width = width or int(_packing.next_power_of_two(np.array([max(int(n.count.max()) for n in nights)]))[0])
    frames = data_epochs * FRAMES_PER_EPOCH

    def pad(a, length, value=0):
        return np.concatenate([a, np.full((length - a.shape[0], *a.shape[1:]), value, a.dtype)])

    arrays = {
        "xyz": np.stack([pad(n.xyz, n_total) for n in nights]),
        "t": np.stack([pad(n.t, n_total) for n in nights]),
        "cell": np.stack([pad(n.cell, n_total) for n in nights]),
        "frac": np.stack([pad(n.frac, n_total) for n in nights]),
        "first": np.stack([pad(n.first, frames) for n in nights]),
        "count": np.stack([pad(n.count, frames) for n in nights]),
        "n": np.array([len(n.t) for n in nights], np.int32),
        "fs": np.array([n.fs for n in nights], np.float32),
        "y": np.stack([pad(n.y, EPOCHS, GAP_LABEL) for n in nights]),
        "taps": np.stack([n.taps for n in nights]),
    }
    return {"arrays": jax.device_put(arrays), "width": width, "names": [n.name for n in nights]}


def window_lanes(first, count, cell, frac, width: int, w0=0):
    """``fused.frames``'s index, points and lane mask for windows ``w0, w0 + 1, …``; padded lanes point past the night."""
    n_total = cell.shape[0]
    w = w0 + jnp.arange(first.shape[0], dtype=jnp.int32)
    lane = jnp.arange(width, dtype=jnp.int32)
    valid = lane[None, :] < count[:, None]
    index = jnp.where(valid, first[:, None] + lane[None, :], n_total)
    safe = jnp.minimum(index, n_total - 1)
    local = (cell[safe] - w[:, None]).astype(jnp.float32) * fused.STEP_S + frac[safe]
    points = jnp.where(valid, 2.0 * jnp.pi * local / fused.WINDOW_S - jnp.pi, 0.0)
    return index, points, valid


def pool(frames: jax.Array) -> jax.Array:
    """``[frames, bins, 2]`` → ``[EPOCHS, bins + 1, 2]`` as ``batching.pool_frames_to_epochs`` and ``pad_pooled``."""
    n_epochs = frames.shape[0] // FRAMES_PER_EPOCH
    f = frames.reshape(n_epochs, FRAMES_PER_EPOCH, *frames.shape[1:])
    real = ~jnp.all(f == fused.PADDING_VALUE, axis=(2, 3))
    count = real.sum(axis=1)
    pooled = jnp.where(real[..., None, None], f, 0.0).sum(axis=1) / jnp.maximum(count, 1)[:, None, None]
    empty = count == 0
    fill = jnp.nanmedian(jnp.where(empty[:, None, None], jnp.nan, pooled), axis=0)
    fill = jnp.where(jnp.isnan(fill), 0.0, fill)
    pooled = jnp.where(empty[:, None, None], fill, pooled)
    gap = 1.0 - count / FRAMES_PER_EPOCH
    rows = jnp.concatenate([pooled, jnp.broadcast_to(gap[:, None, None], (n_epochs, 1, 2))], axis=1)
    padding = jnp.concatenate([fill, jnp.ones((1, 2))])[None]
    return jnp.concatenate([rows, jnp.broadcast_to(padding, (EPOCHS - n_epochs, *rows.shape[1:]))])


def model_input(night: dict, key: jax.Array, width: int, noise: bool | str, lowpass: bool = False) -> jax.Array:
    """One night's ``[EPOCHS, bins + 1, 2]`` model input with a fresh draw of ``noise`` (``True`` is ``"boat"``)."""
    kind = {True: "boat", False: "none"}.get(noise, noise)
    xyz = night["xyz"]
    k1, k2 = jax.random.split(key)
    if kind == "boat":
        spec = jax_generator.sample_spec(k1, night["fs"] / 2)
        xyz = jax_generator.render(k2, spec, night["t"], xyz, night["n"], night["fs"])
    elif kind == "floor":
        sigma = jax.random.uniform(k1, minval=FLOOR_MG[0], maxval=FLOOR_MG[1]) / 1000.0
        real = (jnp.arange(xyz.shape[0]) < night["n"])[:, None]
        xyz = xyz + jnp.where(real, sigma * jax.random.normal(k2, xyz.shape, xyz.dtype), 0.0)
    elif kind != "none":
        raise ValueError(f"unknown noise {noise!r}")
    signal = fused.channels(xyz, night["taps"] if lowpass else None)
    chunk = CHUNK_EPOCHS * FRAMES_PER_EPOCH
    w0 = jnp.arange(night["first"].shape[0] // chunk, dtype=jnp.int32) * chunk

    def chunk_frames(start):
        first = jax.lax.dynamic_slice(night["first"], (start,), (chunk,))
        count = jax.lax.dynamic_slice(night["count"], (start,), (chunk,))
        index, points, valid = window_lanes(first, count, night["cell"], night["frac"], width, start)
        return fused.frames_from_signal(index, points, valid, count > 0, night["fs"], signal)

    frames = jax.lax.map(chunk_frames, w0)
    return pool(frames.reshape(-1, *frames.shape[2:]))


def make_train_step(module, tx, *, width: int, output_classes: int, num_classes: int,
                    balance_beta: float, noise: bool):
    """``(params, opt_state, stack arrays, night indices, key) → (params, opt_state, loss)``."""
    gap_class = num_classes  # the release scores the gap head

    @jax.jit
    def step(params, opt_state, arrays, rows, key):
        batch = jax.tree.map(lambda a: a[rows], arrays)
        keys = jax.random.split(key, rows.shape[0] + 1)
        x = jax.lax.map(lambda a: model_input(a[0], a[1], width, noise), (batch, keys[1:]))
        y = batch["y"]
        mask = ((y >= 0) | (y == GAP_LABEL)).astype(jnp.float32)
        y = jnp.where(y <= GAP_LABEL, gap_class, y)

        def loss_fn(p):
            logits = module.apply({"params": p}, x, mask, train=True, rngs={"dropout": keys[0], "augment": keys[0]})
            return balanced_cross_entropy(logits, y, output_classes, balance_beta)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step

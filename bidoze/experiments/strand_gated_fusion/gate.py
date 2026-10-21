"""A thin gated fusion of frozen strands.

For each epoch the gate returns a softmax over the strands; the fused output is the gate-weighted
sum of the strands' 5-way probabilities. The gate reads, per strand, its penultimate features
(standardized, projected to ``PROJ`` units by a strand-specific linear map and a GELU) and its
5-way probabilities, then one 1-D convolution over time (``KERNEL`` epochs, both directions) maps
the concatenation to one logit per strand. The convolution starts at zero, so an untrained gate
is exactly the mean of the strands' probabilities.

Training minimises the class-balanced cross-entropy (the strands' own loss) of the fused output
on a set of recordings with full-batch AdamW. The number of steps is chosen by 4-fold
cross-validation over participants within that set, jointly with the learning rate, from
``STEP_GRID`` x ``LRS``; 0 steps (the mean) is in the grid.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from arcasleep.training.balanced import balanced_cross_entropy

PROJ = 4
KERNEL = 5
STEP_GRID = (0, 25, 50, 100, 200, 300, 400, 600)
LRS = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAY = 1e-2
BETA = 0.999
FOLDS = 4


class Gate(nn.Module):
    strands: int
    features: int

    @nn.compact
    def __call__(self, feats, probs, mask):
        """``feats`` (B,T,S,F) standardized, ``probs`` (B,T,S,5), ``mask`` (B,T); returns (B,T,S)."""
        w = self.param("project", nn.initializers.glorot_uniform(in_axis=1, out_axis=2),
                       (self.strands, self.features, PROJ))
        b = self.param("project_bias", nn.initializers.zeros, (self.strands, PROJ))
        h = nn.gelu(jnp.einsum("btsf,sfd->btsd", feats, w) + b)
        z = jnp.concatenate([h.reshape(*h.shape[:2], -1), probs.reshape(*probs.shape[:2], -1)], axis=-1)
        z = z * mask[..., None]
        logits = nn.Conv(self.strands, (KERNEL,), padding="SAME",
                         kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
                         name="context")(z)
        return jax.nn.softmax(logits, axis=-1)


def fuse(weights, probs):
    """``weights`` (B,T,S), ``probs`` (B,T,S,5) -> (B,T,5)."""
    return jnp.einsum("bts,btsc->btc", weights, probs)


def loss_fn(module, params, feats, probs, mask, labels):
    mix = fuse(module.apply({"params": params}, feats, probs, mask), probs)
    return balanced_cross_entropy(jnp.log(mix + 1e-8), labels, 4, BETA)


def parameter_count(params) -> int:
    return int(sum(x.size for x in jax.tree.leaves(params)))


class Data:
    """Stacked, padded recordings: feats (B,T,S,F), probs (B,T,S,5), mask (B,T), labels (B,T)."""

    def __init__(self, recordings, T, mean=None, std=None):
        S, F = recordings[0][2].shape[1:]
        B = len(recordings)
        feats = np.zeros((B, T, S, F), np.float32)
        probs = np.zeros((B, T, S, 5), np.float32)
        probs[..., 4] = 1.0
        labels = -np.ones((B, T), np.int32)
        for i, (y, p, f) in enumerate(recordings):
            n = len(y)
            feats[i, :n] = f
            probs[i, :n] = np.moveaxis(p, 0, 1)
            labels[i, :n] = np.where((y >= 0) & (y < 4), y, -1)
        mask = (probs.mean(axis=2)[..., 4] < 0.5).astype(np.float32)
        if mean is None:
            m = mask.astype(bool)
            mean, std = feats[m].mean(axis=0), feats[m].std(axis=0) + 1e-3
        feats = (feats - mean) / std * mask[..., None, None]
        self.mean, self.std = mean, std
        self.feats, self.probs, self.mask, self.labels = (jnp.asarray(a) for a in (feats, probs, mask, labels))

    def take(self, idx):
        idx = np.asarray(idx)
        out = object.__new__(Data)
        out.feats, out.probs, out.mask, out.labels = (a[idx] for a in (self.feats, self.probs, self.mask, self.labels))
        return out


def train(data, steps, lr, seed=0, record=()):
    """Train a gate for ``steps`` steps; returns ``(params, {step: params})`` for steps in ``record``."""
    S, F = data.feats.shape[2:]
    module = Gate(S, F)
    params = module.init(jax.random.PRNGKey(seed), data.feats[:1, :8], data.probs[:1, :8], data.mask[:1, :8])["params"]
    tx = optax.adamw(lr, weight_decay=WEIGHT_DECAY)
    opt = tx.init(params)
    grad = jax.jit(jax.value_and_grad(lambda p: loss_fn(module, p, data.feats, data.probs, data.mask, data.labels)))
    snaps = {0: params} if 0 in record else {}
    for step in range(1, steps + 1):
        _, g = grad(params)
        updates, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, updates)
        if step in record:
            snaps[step] = params
    return params, snaps


def select_steps(data, groups, seed=0):
    """Mean held-out loss over participant folds at each (learning rate, step) of the grid.

    Returns ``(lr, steps, losses[lr index, step index])`` for the lowest mean held-out loss."""
    uniq = sorted(set(groups))
    fold_of = {g: i % FOLDS for i, g in enumerate(uniq)}
    fold = np.array([fold_of[g] for g in groups])
    S, F = data.feats.shape[2:]
    module = Gate(S, F)
    held = jax.jit(lambda p, f, pr, m, l: loss_fn(module, p, f, pr, m, l))
    losses = np.full((len(LRS), FOLDS, len(STEP_GRID)), np.nan)
    for k in range(FOLDS):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if len(te) == 0:
            continue
        held_out = data.take(te)
        for i, lr in enumerate(LRS):
            _, snaps = train(data.take(tr), max(STEP_GRID), lr, seed, record=STEP_GRID)
            losses[i, k] = [float(held(snaps[s], held_out.feats, held_out.probs, held_out.mask, held_out.labels))
                            for s in STEP_GRID]
    mean = np.nanmean(losses, axis=1)
    i, j = np.unravel_index(int(np.argmin(mean)), mean.shape)
    return LRS[i], STEP_GRID[j], mean


def gate_weights(params, data):
    S, F = data.feats.shape[2:]
    return np.asarray(Gate(S, F).apply({"params": params}, data.feats, data.probs, data.mask))

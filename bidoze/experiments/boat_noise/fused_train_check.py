"""Check ``fused_train``: its clean input against the training cache, its speed, and a short fit.

1. Equivalence: with no noise, ``fused_train.model_input`` against ``assemble_batch`` on the
   cache for the same nights; labels, gap rows and the largest spectrum difference.
2. Speed (batch 2, the release's): the cached-input step of ``fit_balanced``, the fused step
   without noise and with noise; first call (compile) and median of 20 later calls.
3. A short fine-tune from one adult release seed with noise on every step: loss per step, and
   4-stage kappa on the held-out nights before and after, clean and with fixed noise draws.

Writes ``equivalence.csv``, ``speed.csv``, ``fit_loss.csv`` and ``fit_kappa.csv`` to ``<out>``.

Usage: python fused_train_check.py <data root> <cache dir> <seed bundle> <out> [steps]
"""
from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import fused_train
from arcasleep.artifacts.publish import load_bundle_model
from arcasleep.data.batching import assemble_batch
from arcasleep.data.cache import load_bundles
from arcasleep.data.constants import GAP_LABEL
from arcasleep.data.prepare import load_dataset
from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion
from arcasleep.training.balanced import selection_confusion

COHORT = "SleepAccel-Clinical"
N_EQUIVALENCE = 4
N_HELD_OUT = 8
BATCH = 2
LR = 1e-4
BALANCE_BETA = 0.999
NOISE_DRAWS = 3


def timed(fn, *args, repeats=20):
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return first, float(np.median(times))


def main(root: Path, cache: Path, seed_bundle: Path, out: Path, steps: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(cache.glob(f"{COHORT}_subject_*_features.npz"))
    ids = [p.name.removeprefix(f"{COHORT}_subject_").removesuffix("_features.npz") for p in paths]
    data_set = load_dataset(root / COHORT)
    start = time.perf_counter()
    nights = [fused_train.load_night(data_set, i) for i in ids]
    print(f"{len(nights)} nights loaded in {time.perf_counter() - start:.1f} s", flush=True)
    stacked = fused_train.stack(nights)
    arrays, width = stacked["arrays"], stacked["width"]
    print(f"stack: {arrays['xyz'].shape[1]} samples, {arrays['first'].shape[1]} frames, width {width}, "
          f"fs {sorted(set(np.round(np.asarray(arrays['fs']), 1)))}", flush=True)

    # 1. Equivalence on the first nights, no noise.
    bundles = load_bundles(paths[:N_EQUIVALENCE])
    refs = sorted(bundles)
    X_ref, y_ref = assemble_batch(bundles, refs, fused_train.EPOCHS, fused_train.EPOCHS, 4)
    rows = []
    clean_input = jax.jit(lambda night: fused_train.model_input(night, jax.random.PRNGKey(0), width, False))
    for k, ref in enumerate(refs):
        i = ids.index(str(ref.subject_id))
        got = np.asarray(clean_input(jax.tree.map(lambda a: a[i], arrays)))
        y = np.asarray(arrays["y"][i])
        spectrum = np.abs(got[:, :-1] - X_ref[k][:, :-1])
        rows.append({"night": nights[i].name, "same_labels": bool(np.array_equal(y, y_ref[k])),
                     "same_gap_row": bool(np.array_equal(got[:, -1], X_ref[k][:, -1])),
                     "max_abs_diff": float(spectrum.max()), "p999_abs_diff": float(np.percentile(spectrum, 99.9))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "equivalence.csv", index=False, float_format="%.6g")

    # 2. Speed.
    model = load_bundle_model(seed_bundle)
    module, cfg = model.module, model.config
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=LR, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
    params = model.params
    opt_state = tx.init(params)
    kwargs = dict(width=width, output_classes=cfg.output_classes, num_classes=cfg.num_classes,
                  balance_beta=BALANCE_BETA)
    clean_step = fused_train.make_train_step(module, tx, noise=False, **kwargs)
    noisy_step = fused_train.make_train_step(module, tx, noise=True, **kwargs)

    @jax.jit
    def cached_step(params, opt_state, xb, mb, yb, key):
        def loss_fn(p):
            logits = module.apply({"params": p}, xb, mb, train=True, rngs={"dropout": key, "augment": key})
            return fused_train.balanced_cross_entropy(logits, yb, cfg.output_classes, BALANCE_BETA)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    xb = jnp.asarray(X_ref[:BATCH], jnp.float32)
    yb = jnp.asarray(y_ref[:BATCH], jnp.int32)
    mb = ((yb >= 0) | (yb == GAP_LABEL)).astype(jnp.float32)
    yb = jnp.where(yb <= GAP_LABEL, cfg.num_classes, yb)
    rows_idx = jnp.arange(BATCH)
    key = jax.random.PRNGKey(1)
    speed = []
    for name, fn, args in [("cached input", cached_step, (params, opt_state, xb, mb, yb, key)),
                           ("fused, no noise", clean_step, (params, opt_state, arrays, rows_idx, key)),
                           ("fused, noise", noisy_step, (params, opt_state, arrays, rows_idx, key))]:
        first, call = timed(fn, *args)
        speed.append({"step": name, "first_call_s": first, "call_s": call})
        print(speed[-1], flush=True)
    pd.DataFrame(speed).to_csv(out / "speed.csv", index=False, float_format="%.4g")

    # 3. Short fine-tune with noise.
    n = len(nights)
    held = np.arange(n - N_HELD_OUT, n)
    train = np.arange(n - N_HELD_OUT)

    @partial(jax.jit, static_argnames="noise")
    def eval_inputs(rows, key, noise):
        keys = jax.random.split(key, rows.shape[0])
        batch = jax.tree.map(lambda a: a[rows], arrays)
        return jax.lax.map(lambda a: fused_train.model_input(a[0], a[1], width, noise), (batch, keys))

    @jax.jit
    def eval_logits(params, x, mask):
        return module.apply({"params": params}, x, mask, train=False)

    def kappa(params, noise: bool, draw: int) -> float:
        cm = np.zeros((cfg.num_classes, cfg.num_classes + 1), dtype=np.int64)
        for i in held:
            rows = jnp.array([i])
            x = eval_inputs(rows, jax.random.PRNGKey(1000 * draw + int(i)), noise=noise)
            y = np.asarray(arrays["y"][rows])
            logits = eval_logits(params, x, jnp.ones(y.shape, jnp.float32))
            cm += selection_confusion(y, np.argmax(np.asarray(logits), axis=-1), cfg.num_classes)
        return float(cohen_kappa_from_confusion(cm))

    def kappas(params, label):
        out_rows = [{"params": label, "input": "clean", "draw": 0, "kappa": kappa(params, False, 0)}]
        out_rows += [{"params": label, "input": "noise", "draw": d, "kappa": kappa(params, True, d)}
                     for d in range(1, NOISE_DRAWS + 1)]
        for r in out_rows:
            print(r, flush=True)
        return out_rows

    kappa_rows = kappas(params, "release seed")
    rng = np.random.default_rng(0)
    loss_rows = []
    start = time.perf_counter()
    for s in range(steps):
        rows = jnp.asarray(rng.choice(train, BATCH, replace=False))
        key, sub = jax.random.split(key)
        params, opt_state, loss = noisy_step(params, opt_state, arrays, rows, sub)
        loss_rows.append({"step": s + 1, "loss": float(loss)})
        if (s + 1) % 50 == 0:
            recent = np.mean([r["loss"] for r in loss_rows[-50:]])
            print(f"step {s + 1}  mean loss (last 50) {recent:.4f}  {time.perf_counter() - start:.1f} s", flush=True)
    pd.DataFrame(loss_rows).to_csv(out / "fit_loss.csv", index=False, float_format="%.6g")
    kappa_rows += kappas(params, f"after {steps} noisy steps")
    pd.DataFrame(kappa_rows).to_csv(out / "fit_kappa.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
         int(sys.argv[5]) if len(sys.argv) > 5 else 300)

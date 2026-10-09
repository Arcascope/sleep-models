"""The balanced recipe with random boat-like noise and a consistency term, on cached model inputs.

Computing the model input on the device every step (``fused_train``) costs about 4× a cached
step. Here the input is computed once per night per noise draw and kept in host memory:

1. Cache every night's clean input, and one noisy input per night (``fused_train.model_input``;
   ``--noise`` picks ``boat``, a ``jax_generator`` draw, ``floor``, white noise at the Galaxy Watch
   floor, or ``none``, which trains on the clean input only). ``--lowpass`` low-passes each axis
   at 6 Hz before the jerk (``fused.channels``), in every input.
2. Train on the cache. Each step scores a batch clean and noisy:
   ``loss = (CE(clean) + CE(noisy)) / 2 + λ · KL(p_clean ‖ p_noisy)``, the KL with the clean
   prediction held fixed (no gradient) and weighted per epoch like the class-balanced CE.
3. After each epoch, the deviation (mean KL over scored epochs, eval mode) on the cached noisy
   inputs is measured. When it has fallen to ``deviation_ratio`` of its value at the last
   refresh, or the round has run ``round_steps`` steps, ``refresh_fraction`` of the fit nights
   get a new draw (least recently drawn first). The deviation on the new draws is measured
   with the current weights *before* they are trained on: this is the deviation on noise the
   model has not seen.
4. The run stops as the recipe does: ``Patience`` on the eval kappa, here the mean of the
   clean and noisy kappa on the held-out nights (one fixed draw per night), or with
   ``--select clean`` (and always with ``--noise none``) the clean kappa alone. With
   ``--noise none``, the noisy eval inputs, used only for the logged deviation, are ``floor``.

Nights, the fit/eval split and the recipe's hyperparameters are the adult release's (the
``matched`` arm); rows are visited in a fresh permutation each epoch.

Writes to ``<out>``: ``reference.csv`` (the seed bundle on the eval nights), ``epochs.csv`` (per epoch), ``refreshes.csv`` (per refresh) and
``bundle/`` (the best weights, with the seed bundle's ``config.json`` and
``preprocessing.json``, and ``arm.json``: the noise, the low-pass flag and the seed).

Usage: python amortized.py <data root> <release cache dir> <seed bundle> <out> [options]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import fused_train
from arcasleep.data.constants import GAP_LABEL
from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion
from arcasleep.training.balanced import LR_DROP_FACTOR, Patience, Plateau, selection_confusion, split_train_eval
from pisces_lite.cv import SubjectRef

# Cache-file cohort prefix → raw dataset directory and the name ``load_dataset`` selects.
SOURCES = {"weaver_revised": ("weaver-2026-10-revised", "weaver_revised"), "SleepAccel": ("SleepAccel", None)}
TRAIN_SETS = ("weaverMatched", "BiDozeSleepAccelFold")


@dataclass(frozen=True)
class Recipe:
    lr: float = 1e-3
    batch_size: int = 2
    balance_beta: float = 0.999
    eval_fraction: float = 0.3
    max_epochs: int = 3000
    lr_drop_factor: float = LR_DROP_FACTOR
    patience: Patience = field(default_factory=lambda: Patience(lr_drop=700, early_stop=1750, unit="steps", smoothing_steps=350))
    consistency: float = 1.0
    deviation_ratio: float = 0.5
    round_steps: int = 1000
    refresh_fraction: float = 0.25
    seed: int = 0
    noise: str = "boat"
    lowpass: bool = False
    select: str = "mean"


def training_refs(cache: Path) -> dict[SubjectRef, tuple[str, str, str | None]]:
    """The release's training nights: ref → (raw cohort dir, subject id, dataset name)."""
    refs = {}
    for link in sorted(cache.glob("*_features.npz")):
        dataset, subject = link.name.removesuffix("_features.npz").split("_subject_")
        if not dataset.startswith(TRAIN_SETS):
            continue
        source = os.readlink(link).rsplit("/", 1)[-1].split("_subject_")[0]
        directory, name = SOURCES[source]
        refs[SubjectRef(dataset, subject)] = (directory, subject, name)
    return refs


_DATASETS: dict = {}


def _load(root: str, directory: str, subject: str, name: str | None) -> fused_train.Night:
    from arcasleep.data.prepare import load_dataset

    if directory not in _DATASETS:
        _DATASETS[directory] = load_dataset(Path(root) / directory, name)
    return fused_train.load_night(_DATASETS[directory], subject)


def ingest(root: Path, sources: list[tuple[str, str, str | None]], workers: int) -> list[fused_train.Night]:
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_load, [str(root)] * len(sources), *zip(*sources)))


class Inputs:
    """Model inputs of single nights, padded to one sample count and one frame count."""

    def __init__(self, nights: list[fused_train.Night], lowpass: bool):
        self.n_total = int(max(len(n.t) for n in nights))
        self.data_epochs = int(max(len(n.y) for n in nights))
        self.lowpass = lowpass

    def __call__(self, night: fused_train.Night, key: jax.Array, noise: str) -> np.ndarray:
        stacked = fused_train.stack([night], self.n_total, self.data_epochs)
        return np.asarray(_one_input(stacked["arrays"], key, width=stacked["width"], noise=noise, lowpass=self.lowpass))


@partial(jax.jit, static_argnames=("width", "noise", "lowpass"))
def _one_input(arrays, key, width, noise, lowpass):
    return fused_train.model_input(jax.tree.map(lambda a: a[0], arrays), key, width, noise, lowpass)


def epoch_weights(y: jax.Array, output_classes: int, beta: float) -> jax.Array:
    """``balanced_cross_entropy``'s per-epoch weight: the class weight, 0 where masked."""
    valid = (y >= 0) & (y < output_classes)
    safe = jnp.where(valid, y, 0)
    counts = jnp.zeros(output_classes).at[safe].add(valid.astype(jnp.float32))
    class_weights = jnp.where(counts > 0, (1.0 - beta) / (1.0 - jnp.power(beta, counts)), 0.0)
    return class_weights[safe] * valid


def kl(clean_logits: jax.Array, noisy_logits: jax.Array) -> jax.Array:
    p = jax.nn.softmax(clean_logits, axis=-1)
    return jnp.sum(p * (jax.nn.log_softmax(clean_logits, axis=-1) - jax.nn.log_softmax(noisy_logits, axis=-1)), axis=-1)


def train_targets(y: jax.Array, num_classes: int) -> tuple[jax.Array, jax.Array]:
    mask = ((y >= 0) | (y == GAP_LABEL)).astype(jnp.float32)
    return mask, jnp.where(y <= GAP_LABEL, num_classes, y)


def main(root: Path, cache: Path, seed_bundle: Path, out: Path, recipe: Recipe, workers: int) -> None:
    from arcasleep.artifacts.publish import load_bundle_model
    from arcasleep_bidoze.model import BiDozeNetModel

    out.mkdir(parents=True, exist_ok=True)
    refs = training_refs(cache)
    fit_refs, eval_refs = split_train_eval(sorted(refs), recipe.eval_fraction, recipe.seed)
    order = fit_refs + eval_refs
    start = time.perf_counter()
    nights = ingest(root, [refs[r] for r in order], workers)
    print(f"{len(fit_refs)} fit and {len(eval_refs)} eval nights loaded in {time.perf_counter() - start:.0f} s; "
          f"{sum(len(n.t) for n in nights) / 1e6:.0f} M samples", flush=True)
    fit_nights, eval_nights = nights[: len(fit_refs)], nights[len(fit_refs):]

    make = Inputs(nights, recipe.lowpass)
    key = jax.random.PRNGKey(recipe.seed)
    start = time.perf_counter()
    clean = np.stack([make(n, key, "none") for n in fit_nights])
    eval_clean = np.stack([make(n, key, "none") for n in eval_nights])
    noisy_eval_kind = "floor" if recipe.noise == "none" else recipe.noise
    eval_noisy = np.stack([make(n, jax.random.PRNGKey(10_000 + i), noisy_eval_kind) for i, n in enumerate(eval_nights)])
    draws = 0
    if recipe.noise == "none":
        noisy = clean
    else:
        noisy = np.stack([make(n, jax.random.fold_in(key, i), recipe.noise) for i, n in enumerate(fit_nights)])
    drawn_at = np.zeros(len(fit_nights), int)
    print(f"caches built in {time.perf_counter() - start:.0f} s: {clean.nbytes * 2 / 1e9:.1f} GB fit, "
          f"{eval_clean.nbytes * 2 / 1e9:.1f} GB eval", flush=True)
    y_fit = np.stack([np.pad(n.y, (0, fused_train.EPOCHS - len(n.y)), constant_values=GAP_LABEL) for n in fit_nights])
    y_eval = np.stack([np.pad(n.y, (0, fused_train.EPOCHS - len(n.y)), constant_values=GAP_LABEL) for n in eval_nights])

    reference = load_bundle_model(seed_bundle)
    cfg = reference.config
    model = BiDozeNetModel(cfg)
    model._ensure_built(tuple(clean.shape[1:]), seed=recipe.seed)
    module, params = model.module, model.params
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=recipe.lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
    opt_state = tx.init(params)
    num_classes, output_classes = cfg.num_classes, cfg.output_classes

    @jax.jit
    def train_step(params, opt_state, xc, xn, y, key):
        mask, target = train_targets(y, num_classes)
        kc, kn = jax.random.split(key)

        def loss_fn(p):
            lc = module.apply({"params": p}, xc, mask, train=True, rngs={"dropout": kc, "augment": kc})
            ln = module.apply({"params": p}, xn, mask, train=True, rngs={"dropout": kn, "augment": kn})
            w = epoch_weights(target, output_classes, recipe.balance_beta)
            ce = fused_train.balanced_cross_entropy(lc, target, output_classes, recipe.balance_beta)
            ce = ce + fused_train.balanced_cross_entropy(ln, target, output_classes, recipe.balance_beta)
            consistency = jnp.sum(w * kl(jax.lax.stop_gradient(lc), ln)) / target.size
            return ce / 2 + recipe.consistency * consistency, consistency

        (loss, consistency), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, consistency

    @jax.jit
    def logits(params, x):
        return module.apply({"params": params}, x, jnp.ones(x.shape[:2], jnp.float32), train=False)

    def all_logits(params, x, chunk=4):
        return np.concatenate([np.asarray(logits(params, jnp.asarray(x[i:i + chunk]))) for i in range(0, len(x), chunk)])

    def kappa(lg, y) -> float:
        return float(cohen_kappa_from_confusion(selection_confusion(y, np.argmax(lg, axis=-1), num_classes)))

    def deviation(params, xc, xn, y) -> tuple[float, float]:
        """Mean KL(clean ‖ noisy) over scored epochs, and the share of them whose wake/sleep call flips."""
        lc, ln = all_logits(params, xc), all_logits(params, xn)
        scored = y >= 0
        d = np.asarray(kl(jnp.asarray(lc), jnp.asarray(ln)))[scored]
        wake_c = np.argmax(lc[..., :num_classes], axis=-1) == 0
        wake_n = np.argmax(ln[..., :num_classes], axis=-1) == 0
        return float(d.mean()), float((wake_c != wake_n)[scored].mean())

    ref_clean, ref_noisy = all_logits(reference.params, eval_clean), all_logits(reference.params, eval_noisy)
    ref_dev, ref_flip = deviation(reference.params, eval_clean, eval_noisy, y_eval)
    reference_row = {"kappa_clean": kappa(ref_clean, y_eval), "kappa_noisy": kappa(ref_noisy, y_eval),
                     "deviation_eval": ref_dev, "flip_eval": ref_flip}
    pd.DataFrame([reference_row]).to_csv(out / "reference.csv", index=False, float_format="%.6g")
    print(f"seed bundle on the eval nights: {reference_row}", flush=True)

    rng = np.random.default_rng(recipe.seed)
    steps_per_epoch = math.ceil(len(fit_nights) / recipe.batch_size)
    plateau = Plateau(recipe.patience, steps_per_epoch)
    lr, best, best_params, best_epoch = recipe.lr, -np.inf, None, -1
    round_start, round_steps = deviation(params, clean, noisy, y_fit)[0], 0
    epoch_rows, refresh_rows = [], []
    for epoch in range(1, recipe.max_epochs + 1):
        t0 = time.perf_counter()
        order_rows = rng.permutation(len(fit_nights))
        losses, consistencies = [], []
        for s in range(steps_per_epoch):
            rows = order_rows[s * recipe.batch_size:(s + 1) * recipe.batch_size]
            if len(rows) < recipe.batch_size:
                rows = np.concatenate([rows, rng.choice(len(fit_nights), recipe.batch_size - len(rows), replace=False)])
            key, sub = jax.random.split(key)
            params, opt_state, loss, consistency = train_step(
                params, opt_state, jnp.asarray(clean[rows]), jnp.asarray(noisy[rows]), jnp.asarray(y_fit[rows]), sub)
            losses.append(float(loss))
            consistencies.append(float(consistency))
        round_steps += steps_per_epoch
        train_s = time.perf_counter() - t0

        lc, ln = all_logits(params, eval_clean), all_logits(params, eval_noisy)
        kc, kn = kappa(lc, y_eval), kappa(ln, y_eval)
        selection = kc if recipe.select == "clean" or recipe.noise == "none" else (kc + kn) / 2
        decision = plateau.update(selection)
        if np.isfinite(selection) and selection > best:
            best, best_params, best_epoch = selection, params, epoch
        cached_dev, cached_flip = deviation(params, clean, noisy, y_fit)
        eval_dev, eval_flip = deviation(params, eval_clean, eval_noisy, y_eval)

        refreshed = False
        if recipe.noise != "none" and (cached_dev <= recipe.deviation_ratio * round_start
                                       or round_steps >= recipe.round_steps):
            t1 = time.perf_counter()
            draws += 1
            count = max(1, round(recipe.refresh_fraction * len(fit_nights)))
            chosen = np.argsort(drawn_at, kind="stable")[:count]
            fresh = np.stack([make(fit_nights[i], jax.random.fold_in(key, 1_000_000 * draws + int(i)), recipe.noise)
                              for i in chosen])
            fresh_dev, fresh_flip = deviation(params, clean[chosen], fresh, y_fit[chosen])
            seen_dev, seen_flip = deviation(params, clean[chosen], noisy[chosen], y_fit[chosen])
            noisy[chosen] = fresh
            drawn_at[chosen] = draws
            refresh_rows.append({"epoch": epoch, "step": plateau.step, "draw": draws, "nights": count,
                                 "round_steps": round_steps, "deviation_round_start": round_start,
                                 "deviation_cached": cached_dev, "deviation_fresh": fresh_dev,
                                 "deviation_replaced": seen_dev, "flip_fresh": fresh_flip, "flip_replaced": seen_flip,
                                 "seconds": time.perf_counter() - t1})
            print(f"    refresh {draws}: {count} nights, deviation on them {seen_dev:.3e} cached → {fresh_dev:.3e} fresh "
                  f"(wake/sleep flips {seen_flip:.2%} → {fresh_flip:.2%}), {time.perf_counter() - t1:.1f} s", flush=True)
            round_start, round_steps, refreshed = deviation(params, clean, noisy, y_fit)[0], 0, True

        epoch_rows.append({"epoch": epoch, "step": plateau.step, "lr": lr, "loss": np.mean(losses),
                           "consistency": np.mean(consistencies), "kappa_clean": kc, "kappa_noisy": kn,
                           "smoothed_kappa": decision.smoothed_kappa, "deviation_cached": cached_dev,
                           "flip_cached": cached_flip, "deviation_eval": eval_dev, "flip_eval": eval_flip,
                           "refreshed": refreshed, "train_s": train_s, "epoch_s": time.perf_counter() - t0})
        print(f"epoch {epoch:3d} step {plateau.step:5d} loss {np.mean(losses):.2e} κ clean {kc:.4f} noisy {kn:.4f} "
              f"best {best:.4f}@{best_epoch} dev cached {cached_dev:.3e} eval {eval_dev:.3e} "
              f"flips eval {eval_flip:.2%} lr {lr:.0e} {time.perf_counter() - t0:.1f} s", flush=True)
        pd.DataFrame(epoch_rows).to_csv(out / "epochs.csv", index=False, float_format="%.6g")
        pd.DataFrame(refresh_rows).to_csv(out / "refreshes.csv", index=False, float_format="%.6g")
        if decision.stop:
            print(f"early stop: {plateau.describe_stop()}", flush=True)
            break
        if decision.drop_lr:
            lr /= recipe.lr_drop_factor
            opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)

    bundle = out / "bundle"
    bundle.mkdir(exist_ok=True)
    for name in ("config.json", "preprocessing.json"):
        shutil.copy(seed_bundle / name, bundle / name)
    (bundle / "arm.json").write_text(json.dumps({"noise": recipe.noise, "lowpass": recipe.lowpass,
                                                  "consistency": recipe.consistency, "seed": recipe.seed}))
    model.params = best_params
    model.model = {"params": best_params}
    model.save_weights_safetensors(bundle / "model.safetensors")
    print(f"best mean eval kappa {best:.4f} at epoch {best_epoch}; weights in {bundle}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("root", "cache", "seed_bundle", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=Recipe.max_epochs)
    parser.add_argument("--consistency", type=float, default=Recipe.consistency)
    parser.add_argument("--deviation-ratio", type=float, default=Recipe.deviation_ratio)
    parser.add_argument("--round-steps", type=int, default=Recipe.round_steps)
    parser.add_argument("--refresh-fraction", type=float, default=Recipe.refresh_fraction)
    parser.add_argument("--seed", type=int, default=Recipe.seed)
    parser.add_argument("--noise", choices=("boat", "floor", "none"), default=Recipe.noise)
    parser.add_argument("--lowpass", action="store_true")
    parser.add_argument("--select", choices=("mean", "clean"), default=Recipe.select)
    a = parser.parse_args()
    main(a.root, a.cache, a.seed_bundle, a.out,
         Recipe(max_epochs=a.max_epochs, consistency=a.consistency, deviation_ratio=a.deviation_ratio,
                round_steps=a.round_steps, refresh_fraction=a.refresh_fraction, seed=a.seed,
                noise=a.noise, lowpass=a.lowpass, select=a.select), a.workers)

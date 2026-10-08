"""Participant-disjoint SeqSmoother methodological replication in Flax/JAX."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import flax.linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from prepare import FEATURES


class SeqSmoother(nn.Module):
    """26 inputs, 40 tokens, two 64-wide pre-norm encoder blocks; 106625 params.

    The paper omits head details; a 32-wide GELU head plus final layer norm
    matches its parameter count. The bounded output uses 30 + 90*sigmoid.
    No causal mask: the final token sees preceding context; every token is
    supervised as in the paper. This supports retrospective inference.
    """
    dropout: float = 0.15

    @nn.compact
    def __call__(self, x, *, training=False):
        x = nn.Dense(64, name="projection")(x)
        x = nn.LayerNorm(name="input_norm")(x)
        position = self.param("position", nn.initializers.normal(0.02), (40, 64))
        x = x + position[None]
        for i in range(2):
            h = nn.LayerNorm(name=f"attention_norm_{i}")(x)
            h = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=64,
                out_features=64, dropout_rate=self.dropout,
                name=f"attention_{i}")(h, deterministic=not training)
            x = x + nn.Dropout(self.dropout, name=f"attention_dropout_{i}")(h, deterministic=not training)
            h = nn.LayerNorm(name=f"ff_norm_{i}")(x)
            h = nn.Dense(256, name=f"ff_expand_{i}")(h)
            h = nn.gelu(h)
            h = nn.Dropout(self.dropout, name=f"ff_inner_dropout_{i}")(h, deterministic=not training)
            h = nn.Dense(64, name=f"ff_contract_{i}")(h)
            x = x + nn.Dropout(self.dropout, name=f"ff_dropout_{i}")(h, deterministic=not training)
        x = nn.LayerNorm(name="output_norm")(x)
        x = nn.gelu(nn.Dense(32, name="head_hidden")(x))
        return 30 + 90 * nn.sigmoid(nn.Dense(1, name="head_out")(x)[..., 0])


MODEL = SeqSmoother()


def sequence_ends(times, stride=1):
    if len(times) < 40:
        return np.array([], dtype=int)
    bad = np.r_[0, np.diff(times) > 45]
    cumulative = np.cumsum(bad)
    ends = np.arange(39, len(times), stride)
    return ends[cumulative[ends] == cumulative[ends - 39]]


def load_recordings(cache, reference_source="kubios_none_mean_hr"):
    recordings = []
    for path in sorted(cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
            if meta["reference_source"] != reference_source:
                continue
            if not meta["nightbeat"]:
                raise ValueError("Formal model run requires the Nightbeat-derived anchor: " + str(path))
            recordings.append(dict(path=path, meta=meta,
                arrays={k: d[k] for k in d.files if k != "metadata"}))
    return recordings


def normalization(recordings, ids):
    arrays = [r["arrays"]["X"] for r in recordings if r["meta"]["participant"] in ids]
    x = np.concatenate(arrays)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    return x.mean(axis=0).astype(np.float32), np.maximum(x.std(axis=0), 1e-5).astype(np.float32)


def dataset(recordings, ids, stride):
    xs, ys, qs, masks, info = [], [], [], [], []
    for r in recordings:
        if r["meta"]["participant"] not in ids:
            continue
        a = r["arrays"]
        for end in sequence_ends(a["endpoint"], stride):
            sl = slice(end - 39, end + 1)
            mask = a["eligible"][sl].astype(np.float32)
            if not mask.any():
                continue
            xs.append(a["X"][sl])
            ys.append(np.nan_to_num(a["reference"][sl], nan=75).astype(np.float32))
            qs.append(a["reference_quality"][sl].astype(np.float32))
            masks.append(mask)
            info.append((r, end))
    if not xs:
        raise ValueError("No 40-window sequences")
    return dict(X=np.asarray(xs, dtype=np.float32), y=np.asarray(ys, dtype=np.float32),
        q=np.asarray(qs, dtype=np.float32), mask=np.asarray(masks, dtype=np.float32), info=info)


def augment(key, x):
    """Jitter in raw feature units, then standardize with training statistics."""
    keys = jax.random.split(key, 10)
    for col, sigma, probability, k in [(0, 2.5, 0.6, 0), (2, 1.5, 0.5, 2), (5, 0.05, 0.3, 4)]:
        gate = jax.random.bernoulli(keys[k], probability, (x.shape[0], 1))
        noise = jax.random.normal(keys[k + 1], x.shape[:2]) * sigma * gate
        x = x.at[:, :, col].add(noise)
    # Half-anchor and delta were derived before feature augmentation, as unspecified in paper.
    gate = jax.random.bernoulli(keys[6], 0.4, (x.shape[0], 1, 1))
    x = x.at[:, :, 7:25].add(jax.random.normal(keys[7], x[:, :, 7:25].shape) * 0.05 * gate)
    start = jax.random.randint(keys[8], (x.shape[0], 1, 1), 0, 18)
    width = jax.random.randint(keys[9], (x.shape[0], 1, 1), 1, 6)
    bins = jnp.arange(18)[None, None]
    # A separate PRNG key for the masking decision.
    mask_key = jax.random.fold_in(key, 100)
    mask_gate = jax.random.bernoulli(mask_key, 0.3, (x.shape[0], 1, 1))
    masked = ((bins >= start) & (bins < start + width)) & mask_gate
    return x.at[:, :, 7:25].set(jnp.where(masked, 0, x[:, :, 7:25]))


def objective(pred, y, q, mask):
    error = pred - y
    absolute = jnp.abs(error)
    huber = jnp.where(absolute <= 1.5, 0.5 * error ** 2, 1.5 * (absolute - 0.75))
    denom = jnp.maximum(mask.sum(), 1)
    regression = (((0.3 + 0.7 * q) * huber + 0.08 * jnp.maximum(absolute - 3, 0) ** 2) * mask).sum() / denom
    pm = (pred * mask).sum() / denom
    ym = (y * mask).sum() / denom
    pc, yc = (pred - pm) * mask, (y - ym) * mask
    correlation = (pc * yc).sum() / jnp.sqrt(jnp.maximum((pc ** 2).sum() * (yc ** 2).sum(), 1e-8))
    return regression + 0.15 * (1 - correlation)


@jax.jit
def update(state, x, y, q, mask, mean, std, key):
    aug_key, dropout_key = jax.random.split(key)
    x = (augment(aug_key, x) - mean) / std
    def loss(params):
        pred = MODEL.apply({"params": params}, x, training=True, rngs={"dropout": dropout_key})
        return objective(pred, y, q, mask)
    loss, grads = jax.value_and_grad(loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def inference(params, x, mean, std):
    return MODEL.apply({"params": params}, (x - mean) / std, training=False)


def predict_dataset(params, d, mean, std, batch_size=128):
    out = []
    for lo in range(0, len(d["X"]), batch_size):
        x = d["X"][lo:lo + batch_size]
        n = len(x)
        if n < batch_size:
            x = np.pad(x, ((0, batch_size - n), (0, 0), (0, 0)))
        out.append(np.asarray(inference(params, x, mean, std))[:n, -1])
    return np.concatenate(out)


def macro_mae(d, predictions):
    rows = []
    for i, (r, end) in enumerate(d["info"]):
        if d["mask"][i, -1]:
            rows.append(dict(participant=r["meta"]["participant"],
                recording=r["path"].stem, error=abs(predictions[i] - d["y"][i, -1])))
    frame = pd.DataFrame(rows)
    # Equal nights/devices within child, then equal children.
    return float(frame.groupby(["participant", "recording"]).error.mean().groupby("participant").mean().mean())


def checkpoint(path, params, mean, std, config):
    np.savez_compressed(path.with_suffix(".normalization.npz"), mean=mean, std=std,
        feature_names=np.array(FEATURES))
    path.write_bytes(serialization.to_bytes(params))
    path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")


def fit_fold(recordings, fold, output, args, split_hash):
    ids = set(fold["train"]) | set(fold["validation"]) | set(fold["test"])
    available = {r["meta"]["participant"] for r in recordings}
    if not ids <= available:
        raise ValueError("Missing frozen participant caches: " + str(sorted(ids - available)))
    if any(set(fold[a]) & set(fold[b]) for a, b in [("train", "test"), ("validation", "test"), ("train", "validation")]):
        raise ValueError("Participant leakage")
    k = fold["fold"]
    directory = output / f"fold{k:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists() and not args.replace:
        print("Skipping completed fold " + str(k), flush=True)
        return
    mean, std = normalization(recordings, fold["train"])
    train = dataset(recordings, fold["train"], args.train_stride)
    validation = dataset(recordings, fold["validation"], 1)
    test = dataset(recordings, fold["test"], 1)
    training_mean = float(train["y"][train["mask"].astype(bool)].mean())
    steps_per_epoch = int(np.ceil(len(train["X"]) / args.batch_size))
    schedule = optax.warmup_cosine_decay_schedule(init_value=0, peak_value=3e-4,
        warmup_steps=5 * steps_per_epoch,
        decay_steps=max(args.epochs, 6) * steps_per_epoch, end_value=1e-6)
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adamw(schedule, weight_decay=1e-3))
    key = jax.random.PRNGKey(args.seed + k)
    key, init_key = jax.random.split(key)
    params = MODEL.init(init_key, jnp.zeros((1, 40, 26), dtype=jnp.float32))["params"]
    count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert count == 106625, count
    state = train_state.TrainState.create(apply_fn=MODEL.apply, params=params, tx=tx)
    config = dict(fold=fold, epochs_cap=args.epochs, minimum_epochs=min(15, args.epochs),
        patience=30, batch_size=args.batch_size, train_stride=args.train_stride,
        parameter_count=count, seed=args.seed + k, feature_names=FEATURES, training_mean=training_mean,
        output_range=[30, 120], split_sha256=split_hash,
        deviations=["Weaver/Kubios proxy reference instead of NightbeatDB beat-derived ECG",
            "Flax reimplementation; omitted head details inferred to match parameter count",
            "Training sequence stride explicit; source paper does not specify stride",
            "Validation selection uses participant-macro final-token MAE"],
        train_sequences=len(train["X"]), validation_sequences=len(validation["X"]),
        test_sequences=len(test["X"]), backend=str(jax.devices()))
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rng = np.random.default_rng(args.seed + k)
    best, best_epoch, best_params, history = float("inf"), 0, None, []
    begun = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train["X"]))
        losses = []
        for lo in range(0, len(order), args.batch_size):
            idx = order[lo:lo + args.batch_size]
            n = len(idx)
            idx = np.pad(idx, (0, args.batch_size - n), mode="edge")
            mask = train["mask"][idx].copy()
            mask[n:] = 0
            key, step_key = jax.random.split(key)
            state, loss = update(state, train["X"][idx], train["y"][idx], train["q"][idx],
                mask, mean, std, step_key)
            losses.append(float(loss))
        pred = predict_dataset(state.params, validation, mean, std, args.batch_size)
        score = macro_mae(validation, pred)
        if not np.isfinite(score) or not np.isfinite(losses).all():
            raise ValueError("Nonfinite training/validation")
        if score < best:
            best, best_epoch, best_params = score, epoch, state.params
            checkpoint(directory / "best.msgpack", best_params, mean, std,
                       {**config, "best_epoch":epoch, "validation_macro_mae":best})
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
            validation_macro_mae=score, best_validation_macro_mae=best,
            elapsed_seconds=time.monotonic() - begun))
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        if epoch == 1 or epoch % 5 == 0:
            print(f"fold {k} epoch {epoch}: val {score:.3f}, best {best:.3f}, "
                  f"elapsed {time.monotonic() - begun:.0f}s", flush=True)
        if epoch >= min(15, args.epochs) and epoch - best_epoch >= 30:
            break
    predictions = predict_dataset(best_params, test, mean, std, args.batch_size)
    rows = []
    methods = ["peak_narrow", "peak_wide", "peak_median", "peak_screened",
               "nb_anchor", "nightbeat", "anchor_filled", "anchor_median"]
    for i, (r, end) in enumerate(test["info"]):
        a = r["arrays"]
        if not a["eligible"][end]:
            continue
        row = dict(fold=k, participant=r["meta"]["participant"],
            recording=r["path"].stem, device=r["meta"]["device"],
            endpoint=float(a["endpoint"][end]), reference=float(a["reference"][end]),
            quality=float(a["reference_quality"][end]), psg_stage=str(a["psg_stage"][end]),
            seqsmoother=float(predictions[i]), training_mean=training_mean, subharmonic=float(a["X"][end, 5]),
            motion=float(a["motion"][end]), clarity=float(a["peak_wide_clarity"][end]))
        row.update({name:float(a[name][end]) for name in methods})
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "test_predictions.csv.gz", index=False)
    completion = dict(fold=k, test_participants=fold["test"], epochs_run=epoch,
        best_epoch=best_epoch, validation_macro_mae=best, test_windows=len(rows),
        elapsed_seconds=time.monotonic() - begun, status="complete",
        split_sha256=split_hash, smoke=args.epochs < 15)
    (directory / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    print("Fold complete: " + json.dumps(completion), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--reference-source", default="kubios_none_mean_hr")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-stride", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()
    split_bytes = args.splits.read_bytes()
    protocol = json.loads(split_bytes)
    recordings = load_recordings(args.cache, args.reference_source)
    for fold in protocol["folds"]:
        if args.folds is None or fold["fold"] in args.folds:
            fit_fold(recordings, fold, args.output, args, hashlib.sha256(split_bytes).hexdigest())


if __name__ == "__main__":
    main()

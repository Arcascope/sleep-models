"""A one-layer temporal CNN readout: light/deep/REM from k epochs of the release's output and HR.

Between the logistic readout of ``readout.py`` (kernel 1) and an HR input channel in BiDoze:
one centred Conv1d over a night's 30-s epochs, inputs → 3 logits (light, deep, REM).

Inputs per epoch:
- ``none``: the release's log P(wake), log P(light), log P(deep), log P(REM);
- ``<source> raw``: those plus HR relative to the night's median and a missing-HR flag
  (missing HR is 0); the convolution learns level and variability itself;
- ``<source> hand``: those plus the 7 hand features of ``readout.py``.
Sources: ``seqsmoother``, ``jerk``, ``watch`` (SleepAccel only).

Kernel k ∈ {1, 9, 21, 41} epochs. k = 21 (10.5 min) is the primary comparison, named before
running; the other kernels are secondary.

Each night is padded by replicating its first and last epoch, never with zeros: zero padding
would tell the head where the recording starts and ends, a position-in-night cue.

Training: cross-entropy on PSG light/deep/REM epochs, no class weights, L2 penalty on the
kernel, Adam, full batch. Inputs are standardized on the training nights' PSG sleep epochs.
Number of steps: early stopping on held-out log-loss in an inner split (20% of the training
subjects, drawn per seed), then a refit on all training subjects for that many steps. The test
fold is never seen. 5 seeds per configuration; their probabilities are averaged.

Decoding, cross-validation and cohorts are those of ``readout.py``: the release's sleep/wake
calls are kept; on its called-sleep epochs the head's most probable stage is the call. SleepAccel
is cross-validated on the BiDoze folds; SleepAccel-Clinical uses the head trained on all
SleepAccel.

Writes ``<run>/cnn/``: subjects.csv, summary.csv, logloss.csv, deep_tracking.csv, paired.csv.

Usage: python cnn_readout.py <run>
"""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.stats import wilcoxon

from readout import BASE, SOURCES, bootstrap_ci, build, release_calls, subject_metrics

KERNELS = [1, 9, 21, 41]
PRIMARY = 21
SEEDS = range(5)
L2 = 1e-4
LR = 1e-2
MAX_STEPS = 3000
EVAL_EVERY = 25
INNER = 0.2
T_MAX = 1000
HAND = ["hr_rel", "hr_rel_5min", "hr_sd_2min", "hr_sd_5min", "hr_change", "hr_abs_change_2min", "hr_missing"]


def arms() -> dict[str, list[str]]:
    out = {"none": BASE}
    for s in SOURCES:
        out[f"{s} raw"] = BASE + [f"{s}:hr_rel", f"{s}:hr_missing"]
        out[f"{s} hand"] = BASE + [f"{s}:{n}" for n in HAND]
    return out


def nights(d: pd.DataFrame) -> list[dict]:
    out = []
    for (cohort, sid), g in d.groupby(["cohort", "subject"], sort=False):
        g = g.sort_values("epoch")
        assert len(g) <= T_MAX
        y = g.psg.to_numpy()
        out.append(dict(cohort=cohort, subject=sid, fold=int(g.fold.iloc[0]), index=g.index.to_numpy(),
                        frame=g, target=np.where((y >= 1) & (y <= 3), y - 1, -1)))
    return out


def stack(group: list[dict], features: list[str], mean, std, k: int, size: int):
    """(size, T_MAX + k − 1, C) inputs, edge-padded per night; (size, T_MAX) targets (−1 unscored)."""
    half = (k - 1) // 2
    x = np.zeros((size, T_MAX + k - 1, len(features)), np.float32)
    y = np.full((size, T_MAX), -1, np.int32)
    for i, n in enumerate(group):
        z = (n["frame"][features].to_numpy(float) - mean) / std
        t = len(z)
        x[i] = np.pad(z, ((half, T_MAX - t + half), (0, 0)), mode="edge")
        y[i, :t] = n["target"]
    return jnp.asarray(x), jnp.asarray(y)


def logits(params, x):
    out = jax.lax.conv_general_dilated(x, params["w"], (1,), "VALID", dimension_numbers=("NWC", "WIO", "NWC"))
    return out + params["b"]


def nll(params, x, y):
    lp = jax.nn.log_softmax(logits(params, x))
    mask = y >= 0
    picked = jnp.take_along_axis(lp, jnp.maximum(y, 0)[..., None], axis=-1)[..., 0]
    return -(picked * mask).sum() / jnp.maximum(mask.sum(), 1)


def loss(params, x, y):
    return nll(params, x, y) + L2 * jnp.sum(params["w"] ** 2)


@partial(jax.jit, static_argnums=(4,))
def train_steps(params, opt_state, x, y, steps: int):
    tx = optax.adam(LR)

    def body(carry, _):
        p, s = carry
        g = jax.grad(loss)(p, x, y)
        u, s = tx.update(g, s, p)
        return (optax.apply_updates(p, u), s), None

    (params, opt_state), _ = jax.lax.scan(body, (params, opt_state), None, length=steps)
    return params, opt_state


nll_jit = jax.jit(nll)


def init(key, k: int, c: int):
    return {"w": 0.01 * jax.random.normal(key, (k, c, 3)), "b": jnp.zeros(3)}


def train(group: list[dict], features: list[str], k: int, seed: int, fold_key: int):
    rows = pd.concat([n["frame"] for n in group])
    sleep = rows[rows.psg.between(1, 3)][features]
    mean, std = sleep.mean().to_numpy(), sleep.std().replace(0, 1).fillna(1).to_numpy()
    rng = np.random.default_rng(1000 * seed + fold_key)
    order = rng.permutation(len(group))
    nval = max(1, round(INNER * len(group)))
    inner_val = [group[i] for i in order[:nval]]
    inner_train = [group[i] for i in order[nval:]]
    xt, yt = stack(inner_train, features, mean, std, k, 32)
    xv, yv = stack(inner_val, features, mean, std, k, 32)
    key = jax.random.PRNGKey(seed)
    params = init(key, k, len(features))
    opt_state = optax.adam(LR).init(params)
    best, best_steps = np.inf, EVAL_EVERY
    for step in range(EVAL_EVERY, MAX_STEPS + 1, EVAL_EVERY):
        params, opt_state = train_steps(params, opt_state, xt, yt, EVAL_EVERY)
        v = float(nll_jit(params, xv, yv))
        if v < best - 1e-5:
            best, best_steps = v, step
        elif step - best_steps >= 500:
            break
    xa, ya = stack(group, features, mean, std, k, 32)
    params = init(key, k, len(features))
    opt_state = optax.adam(LR).init(params)
    params, _ = train_steps(params, opt_state, xa, ya, best_steps)
    return params, mean, std, best_steps


def predict(params, mean, std, group: list[dict], features: list[str], k: int) -> list[np.ndarray]:
    x, _ = stack(group, features, mean, std, k, len(group))
    p = np.asarray(jax.nn.softmax(logits(params, x)))
    return [p[i, :len(n["index"])] for i, n in enumerate(group)]


def main(run: Path) -> None:
    out = run / "cnn"
    out.mkdir(exist_ok=True)
    d, _ = build(pd.read_csv(run / "epochs.csv.gz", dtype={"subject": str}))
    divisors = json.loads((run / "release_divisors.json").read_text())
    base = release_calls(d, divisors)
    y = d.psg.to_numpy()
    all_nights = nights(d)
    sa = [n for n in all_nights if n["cohort"] == "SleepAccel"]
    cl = [n for n in all_nights if n["cohort"] == "SleepAccel-Clinical"]
    called_sleep = (base >= 1) & (base <= 3)

    calls, losses, steps_log = {"release": base}, [], []
    for k in KERNELS:
        for arm, features in arms().items():
            name = f"k{k} {arm}"
            proba = np.full((len(d), 3), np.nan)
            for fold in list(range(5)) + [-1]:
                train_group = [n for n in sa if n["fold"] != fold] if fold >= 0 else sa
                test_group = [n for n in sa if n["fold"] == fold] if fold >= 0 else cl
                if arm.startswith("watch") and fold < 0:
                    continue
                per_seed = []
                for seed in SEEDS:
                    params, mean, std, steps = train(train_group, features, k, seed, fold + 1)
                    per_seed.append(predict(params, mean, std, test_group, features, k))
                    steps_log.append({"config": name, "fold": fold, "seed": seed, "steps": steps})
                for i, n in enumerate(test_group):
                    proba[n["index"]] = np.mean([s[i] for s in per_seed], axis=0)
            c = base.copy()
            has = np.isfinite(proba).all(axis=1)
            pick = called_sleep & has
            c[pick] = 1 + proba[pick].argmax(axis=1)
            c[~has] = -9
            calls[name] = c
            for cohort in ("SleepAccel", "SleepAccel-Clinical"):
                rows = d.cohort.eq(cohort).to_numpy() & (y >= 1) & (y <= 3) & has
                if rows.any():
                    p = np.clip(proba[rows], 1e-12, 1)
                    losses.append({"cohort": cohort, "config": name, "k": k, "arm": arm,
                                   "logloss_pooled": float(-np.mean(np.log(p[np.arange(rows.sum()), y[rows] - 1])))})
            print(name, "done", flush=True)
    pd.DataFrame(losses).to_csv(out / "logloss.csv", index=False)
    pd.DataFrame(steps_log).to_csv(out / "steps.csv", index=False)

    rows = []
    for name, c in calls.items():
        for n in all_nights:
            idx = n["index"]
            if (c[idx] == -9).any():
                continue
            rows.append({"cohort": n["cohort"], "subject": n["subject"], "config": name,
                         **subject_metrics(y[idx], c[idx])})
    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(out / "subjects.csv", index=False)
    metrics = [m for m in per_subject.columns if m not in ("cohort", "subject", "config")]
    summary = per_subject.groupby(["cohort", "config"], sort=False)[metrics].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    tracking = per_subject.groupby(["cohort", "config"], sort=False).apply(lambda g: pd.Series({
        "r_deep_vs_psg": g["deep%"].corr(g["psg_deep%"]),
        "mean_abs_deep_error_points": (g["deep%"] - g["psg_deep%"]).abs().mean()})).reset_index()
    tracking.to_csv(out / "deep_tracking.csv", index=False)

    # Paired: HR − no HR at the same k; no HR at k − no HR at k = 1; each config − release.
    rng = np.random.default_rng(20261008)
    pairs = []
    for k in KERNELS:
        for arm in arms():
            if arm != "none":
                pairs.append((f"k{k} {arm}", f"k{k} none"))
            pairs.append((f"k{k} {arm}", "release"))
        if k != 1:
            pairs.append((f"k{k} none", "k1 none"))
    paired = []
    for cohort in ("SleepAccel", "SleepAccel-Clinical"):
        sub = per_subject[per_subject.cohort.eq(cohort)].pivot(index="subject", columns="config", values="kappa_4stage")
        for a, b in pairs:
            if a not in sub or b not in sub:
                continue
            diff = (sub[a] - sub[b]).dropna().to_numpy()
            lo, hi = bootstrap_ci(diff, rng)
            paired.append({"cohort": cohort, "a": a, "b": b, "n": len(diff), "diff": diff.mean(),
                           "ci_low": lo, "ci_high": hi,
                           "p": float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0})
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

"""SeqSmoother HR models on the BiDoze SleepAccel folds, and HR for every window.

Folds 0-4 hold out the subjects of ``sleepaccel_5fold.json`` fold k, so a SleepAccel subject's
HR comes from a model that never trained on its fold. Fold 5 (``all``) trains on every
SleepAccel subject and scores SleepAccel-Clinical. In each fold, 20% of the training subjects
(seeded) are the early-stopping holdout. Fold 5 has no test subjects; ``train.py`` scores a test
set after training, so for fold 5 it scores the holdout, whose predictions are not used.

Inference covers every window, not only windows with reference HR: each window gets the
final-token prediction of the 40-window sequence ending at it, and the first 39 windows of a
contiguous run take their position in the run's first sequence.

Writes ``<run>/splits.json``, ``<run>/models/fold0k/`` and ``<run>/hr_windows.csv.gz``.

Usage: python hr_models.py --plan <sleepaccel_5fold.json> --sleepaccel-cache <dir>
                           --clinical-cache <dir> --output <run>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import serialization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seqsmoother_weaver"))
from prepare import FEATURES  # noqa: E402
import train  # noqa: E402
from train import MODEL, fit_fold, inference, load_recordings  # noqa: E402

SOURCE = "apple_optical_hr_median"
SEED = 20261008
ALL = 5


def make_splits(plan: dict) -> dict:
    names = sorted(plan["folds"])
    everyone = sorted(s for n in names for s in plan["folds"][n])
    folds = []
    for k, name in enumerate(names + ["all"]):
        test = sorted(plan["folds"][name]) if name != "all" else []
        rest = [s for s in everyone if s not in set(test)]
        shuffled = list(np.random.default_rng(SEED + k).permutation(rest))
        nval = max(1, round(0.2 * len(shuffled)))
        validation, train = sorted(shuffled[:nval]), sorted(shuffled[nval:])
        folds.append(dict(fold=k, name=name, train=train, validation=validation,
                          test=test))
    return dict(seed=SEED, reference_source=SOURCE, plan=plan, participants=everyone, folds=folds,
                note="fold 5 trains on all SleepAccel and has no test subjects")


def load_model(directory: Path):
    template = MODEL.init(jax.random.PRNGKey(0), jnp.zeros((1, 40, 26)))["params"]
    params = serialization.from_bytes(template, (directory / "best.msgpack").read_bytes())
    with np.load(directory / "best.normalization.npz") as n:
        assert n["feature_names"].tolist() == FEATURES
        return params, n["mean"], n["std"]


def recording_hr(params, mean, std, X: np.ndarray, endpoint: np.ndarray, batch: int = 256) -> np.ndarray:
    """HR for every window of one recording (NaN in contiguous runs shorter than 40 windows)."""
    out = np.full(len(endpoint), np.nan)
    run = np.r_[0, np.cumsum(np.diff(endpoint) > 45)]
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        if len(idx) < 40:
            continue
        starts = idx[: len(idx) - 39]
        seqs = np.stack([X[s:s + 40] for s in starts]).astype(np.float32)
        preds = []
        for lo in range(0, len(seqs), batch):
            x = seqs[lo:lo + batch]
            n = len(x)
            if n < batch:
                x = np.pad(x, ((0, batch - n), (0, 0), (0, 0)))
            preds.append(np.asarray(inference(params, x, mean, std))[:n])
        preds = np.concatenate(preds)
        out[idx[:40]] = preds[0]
        out[idx[39:]] = preds[:, -1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--sleepaccel-cache", type=Path, required=True)
    ap.add_argument("--clinical-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    splits = make_splits(json.loads(a.plan.read_text()))
    split_path = a.output / "splits.json"
    split_path.write_text(json.dumps(splits, indent=2) + "\n")
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    recordings = load_recordings(a.sleepaccel_cache, SOURCE)
    args = SimpleNamespace(epochs=120, batch_size=128, train_stride=40, seed=20261007, replace=False)
    dataset = train.dataset
    for fold in splits["folds"]:
        # An empty test list (fold 5) scores the holdout instead.
        train.dataset = lambda recs, ids, stride, v=fold["validation"]: dataset(recs, ids or v, stride)
        fit_fold(recordings, fold, a.output / "models", args, split_hash)
    train.dataset = dataset

    rows = []
    fold_of = {s: f["fold"] for f in splits["folds"][:ALL] for s in f["test"]}
    models = {k: load_model(a.output / "models" / f"fold{k:02d}") for k in range(ALL + 1)}
    for r in recordings:
        sid = r["meta"]["participant"]
        k = fold_of[sid]
        hr = recording_hr(*models[k], r["arrays"]["X"], r["arrays"]["endpoint"])
        rows.append(pd.DataFrame(dict(cohort="SleepAccel", participant=sid, model_fold=k,
                                      endpoint=r["arrays"]["endpoint"], seqsmoother=hr,
                                      reference=r["arrays"]["reference"])))
    for path in sorted(a.clinical_cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            hr = recording_hr(*models[ALL], d["X"], d["endpoint"])
            rows.append(pd.DataFrame(dict(cohort="SleepAccel-Clinical", participant=path.stem, model_fold=ALL,
                                          endpoint=d["endpoint"], seqsmoother=hr, reference=np.nan)))
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(a.output / "hr_windows.csv.gz", index=False)
    sa = frame[frame.cohort.eq("SleepAccel") & frame.reference.between(30, 120)]
    err = (sa.seqsmoother - sa.reference).abs()
    print("SleepAccel windows with reference:", len(sa), "covered:", int(err.notna().sum()),
          "macro MAE:", round(err.groupby(sa.participant).mean().mean(), 3), flush=True)


if __name__ == "__main__":
    main()

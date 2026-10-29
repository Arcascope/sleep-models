"""Per-epoch calls of the out-of-fold adult BiDoze models on the PSG span of each Wearanize+ night.

    python -m arcasleep_watchtree.comparison_calls \
        --training ../bidoze/results/adult-f2-2026-10-23/training \
        --cache-root ../feature_cache --out results/<run>/cv/bidoze_epochs.parquet

Each night is scored by the fold model that held it out, with that fold's stage divisor. The
input is the PSG span alone, cut from the uncropped feature cache. Nights with a PSG span over
3000 epochs are left out.

Writes one row per epoch: ``subject``, ``fold``, ``epoch`` (index within the PSG span), ``y_raw``
(the cache label) and ``call`` (0 wake, 1 light, 2 deep, 3 REM, 4 gap).
:mod:`arcasleep_watchtree.report` compares these calls with WatchTree on the same epochs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.data.batching import pad_pooled, pool_frames_to_epochs, split_gap_row
from arcasleep.divisor import hybrid_calls
from arcasleep.predict import _stage_and_gap_probabilities
from arcasleep_bidoze.adult import StackedModel, adult_stage_divisor, stage_counts
from arcasleep_bidoze.model import BiDozeNetModel

MAX_EPOCHS = 3000
FRAMES_PER_EPOCH = 15
SEEDS = 3
FOLDS = 5
UNCROPPED = "12Hz_log_PSD_uncropped"
ALIGNED = "12Hz_log_PSD_aligned"
#: Cache file prefix of each training dataset name.
ALIGNED_PREFIX = {
    "weaverAll": "weaver_revised",
    "BiDozeSleepAccel": "SleepAccel",
    "BiDozeWearanize": "wearanize",
}
ADULT_DATASETS = ("BiDozeSleepAccel", "BiDozeWearanize")


def aligned_path(cache_root: Path, dataset: str, subject: str) -> Path:
    for name, prefix in ALIGNED_PREFIX.items():
        if dataset.startswith(name):
            return cache_root / ALIGNED / f"{prefix}_subject_{subject}_features.npz"
    raise KeyError(dataset)


def fold_divisor(training: Path, cache_root: Path, fold: int) -> np.ndarray:
    """The adult stage divisor of one fold, from the stage counts of its training nights."""
    splits = json.loads((training / f"fold{fold}/fit_eval_splits/fold{fold}.json").read_text())
    counts = {
        f"{s['dataset']}/{s['subject_id']}": stage_counts(
            np.load(aligned_path(cache_root, s["dataset"], s["subject_id"]))["y"]
        )
        for s in splits["fit"] + splits["eval"]
    }
    train = np.array(list(counts.values()))
    adult = np.array([v for key, v in counts.items() if key.startswith(ADULT_DATASETS)])
    config = json.loads((training / f"fold{fold}/config/training.json").read_text())
    divisor = adult_stage_divisor(train, adult, config["balance_beta"], config["batch_size"])
    return np.concatenate([[1.0], divisor[1:]])


def load_fold_model(training: Path, fold: int) -> StackedModel:
    outputs = training / f"fold{fold}/model_outputs"
    return StackedModel(
        [BiDozeNetModel.loadz(str(outputs / f"model_fold{fold}_seed_{s}.pisces")) for s in range(SEEDS)]
    )


def calls_of(model: StackedModel, divisor: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    """Stage call per epoch of ``pooled`` (n, 61, 2); 4 where the gap row says absent."""
    n = pooled.shape[0]
    logits = model.infer(pad_pooled(pooled, MAX_EPOCHS)[None])[0][:n]
    probs, _ = _stage_and_gap_probabilities(logits, 4)
    calls = hybrid_calls(probs, probs, divisor)
    return np.where(split_gap_row(pooled)[1] >= 1.0, 4, calls)


def held_out_fold(training: Path) -> dict[str, int]:
    """Wearanize+ subject id -> the fold that held it out, from the saved predictions."""
    fold_of = {}
    for k in range(FOLDS):
        prefix = f"BiDozeWearanizeFold{k}_"
        for path in (training / f"fold{k}/predictions/fold{k}_seed_0").glob(f"{prefix}*.csv"):
            fold_of[path.stem[len(prefix):]] = k
    return fold_of


def bidoze_epochs(training: Path, cache_root: Path) -> pd.DataFrame:
    fold_of = held_out_fold(training)
    rows = []
    for k in range(FOLDS):
        model, divisor = load_fold_model(training, k), fold_divisor(training, cache_root, k)
        for subject in sorted(s for s, fold in fold_of.items() if fold == k):
            path = cache_root / UNCROPPED / f"wearanize_subject_{subject}_features.npz"
            if not path.exists():
                continue
            cached = np.load(path)
            x, y = cached["X"][0], cached["y"][0]
            in_psg = np.flatnonzero(y != -3)
            a, b = int(in_psg[0]), int(in_psg[-1]) + 1
            if b - a > MAX_EPOCHS:
                continue
            frames = x[None, a * FRAMES_PER_EPOCH : b * FRAMES_PER_EPOCH]
            pooled = pool_frames_to_epochs(frames, np.zeros((1, b - a)))[0]
            rows.append(
                pd.DataFrame(
                    {
                        "subject": subject,
                        "fold": k,
                        "epoch": np.arange(b - a),
                        "y_raw": y[a:b],
                        "call": calls_of(model, divisor, pooled),
                    }
                )
            )
        print("fold", k, flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training", required=True, type=Path, help="training/ of the BiDoze run")
    parser.add_argument("--cache-root", required=True, type=Path, help="the feature_cache directory")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bidoze_epochs(args.training, args.cache_root).to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()

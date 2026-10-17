"""Score checkpoints on a draw's training recordings, for boosting weights, out-of-bag scores and stacking.

The draw's training recordings are split as training splits them (``split_train_eval`` over the
cv's train sets, eval fraction and split seed 0 as the arms use). Each checkpoint scores every
training recording, fit and eval share alike; the output records which share each recording is in.

Writes ``<arm>/draw<d>/train_predictions/<checkpoint stem>.npz``, keyed ``<dataset>/<subject_id>``:
the ``(epochs, 5)`` probabilities (wake, light, deep, REM, gap) over the recording's own epochs, and
``__labels__/<key>`` the 4-stage labels (−1 unscored, the gap label for gap). ``__fit__`` and
``__eval__`` list the keys of the two shares. A checkpoint whose file exists is skipped.

With a fifth argument, the checkpoints and training config are read from that arm directory (an arm
of another run) and the output still goes under ``<run dir>/<arm>``.

Usage: python train_predictions.py <run dir> <arm> <draw> [checkpoint glob, default *.pisces] [source arm dir]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from arcasleep.data.cache import discover_cached_subjects, load_cached_features
from arcasleep.training.balanced import split_train_eval
from arcasleep.training.driver import evaluate_subject
from arcasleep_bidoze.model import BiDozeNetModel

CACHE_NAME = "12Hz_log_PSD_aligned"


def training_split(run: Path, d: int, eval_fraction: float = 0.3, split_seed: int = 0):
    cv = json.loads((run / f"setup/cv/draw{d}.json").read_text())
    cached = discover_cached_subjects(cv["train_sets"], run / f"setup/cache/draw{d}" / CACHE_NAME)
    # Training splits the fold's sorted subject list; split_train_eval's split depends on the order.
    refs = sorted(s.ref for subjects in cached.values() for s in subjects)
    fit_refs, eval_refs = split_train_eval(refs, eval_fraction, split_seed)
    return cached, fit_refs, eval_refs


def key(ref) -> str:
    return f"{ref.dataset_name}/{ref.subject_id}"


def main(run: Path, arm: str, d: int, pattern: str = "*.pisces", source: str | None = None) -> None:
    out = run / arm / f"draw{d}"
    src = Path(source) / f"draw{d}" if source else out
    training = json.loads((src / "config/training.json").read_text())
    cached, fit_refs, eval_refs = training_split(run, d, training.get("eval_fraction", 0.3))
    features = load_cached_features(s for subjects in cached.values() for s in subjects)
    target = out / "train_predictions"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted((src / "model_outputs").glob(pattern)):
        dest = target / f"{path.stem.removeprefix('model_')}.npz"
        if dest.exists():
            print(f"{dest.name}: exists")
            continue
        start = time.time()
        model = BiDozeNetModel.loadz(path)
        arrays = {"__fit__": np.array([key(r) for r in fit_refs]), "__eval__": np.array([key(r) for r in eval_refs])}
        for ref in [*fit_refs, *eval_refs]:
            bundle = features[ref]
            y_true, proba = evaluate_subject(
                model, bundle, target_x_length=training["x_length"], target_y_length=training["y_length"],
                num_classes=model.config.num_classes,
            )
            n = min(bundle.y.shape[-1], proba.shape[0])
            arrays[key(ref)] = proba[:n].astype(np.float32)
            arrays[f"__labels__/{key(ref)}"] = np.asarray(y_true[:n], dtype=np.int16)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(dest)
        print(f"{dest.name}: {len(fit_refs)} fit + {len(eval_refs)} eval recordings, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), *(sys.argv[4:6]))

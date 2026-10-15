"""Score each member of a joint fit by itself on the draw's test subjects.

Training writes the joint fit's mean output to ``predictions/draw<d>_seed_0`` and one checkpoint
per member to ``model_outputs/model_draw<d>_seed_0_member<m>.pisces``. This writes each member's
own output to ``predictions/draw<d>_seed_0_member<m>``, in the same format.

Usage: python member_predictions.py <run dir> <arm> <draw>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from arcasleep.data.cache import discover_cached_subjects, load_cached_features
from arcasleep.evaluation.predictions import write_scored_prediction
from arcasleep.training.driver import evaluate_subject
from arcasleep_bidoze.model import BiDozeNetModel

CACHE_NAME = "12Hz_log_PSD_aligned"


def main(run: Path, arm: str, d: int) -> None:
    out = run / arm / f"draw{d}"
    cv = json.loads((run / f"setup/cv/draw{d}.json").read_text())
    training = json.loads((out / "config/training.json").read_text())
    cached = discover_cached_subjects(cv["test_sets"], run / f"setup/cache/draw{d}" / CACHE_NAME)
    features = load_cached_features(s for subjects in cached.values() for s in subjects)
    for path in sorted((out / "model_outputs").glob(f"model_draw{d}_seed_*_member*.pisces")):
        name = path.stem.removeprefix("model_")
        model = BiDozeNetModel.loadz(path)
        directory = out / "predictions" / name
        for ref, bundle in sorted(features.items()):
            y_true, proba = evaluate_subject(
                model, bundle, target_x_length=training["x_length"], target_y_length=training["y_length"],
                num_classes=model.config.num_classes,
            )
            write_scored_prediction(directory, ref, bundle, y_true, proba, model.config.num_classes)
        print(f"{name}: {len(features)} subjects -> {directory}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]))

"""Build the adult bundle from an arm's final phase: the stage source that is released.

- Each seed's checkpoint in ``<run>/<arm>/final/model_outputs/`` becomes a release bundle.
- The divisor comes from the final phase's training nights (the arm's Weaver recordings and
  all of SleepAccel) and BIDsleep, the adult reference.
- The pediatric release bundle is the wake source.

Usage: python bundle.py <run dir> <arm> <adult cache root> <pediatric bundle> <output dir> <version>
(the adult cache root holds ``12Hz_log_PSD_aligned/`` with BIDsleep).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from arcasleep.artifacts.publish import build_release_bundle
from arcasleep.data.cache import discover_cached_subjects
from arcasleep_bidoze.adult import adult_stage_divisor, build_adult_bundle, stage_counts

CACHE_NAME = "12Hz_log_PSD_aligned"


def night_counts(paths: list[Path]) -> np.ndarray:
    return np.array([stage_counts(np.load(p)["y"]) for p in paths])


def main(run: Path, arm: str, adult_root: Path, pediatric: Path, output: Path, version: str) -> None:
    final = run / arm / "final"
    cv = json.loads((final / "config/cv.json").read_text())
    training = json.loads((final / "config/training.json").read_text())
    cache = run / "setup/cache" / arm / CACHE_NAME
    train = discover_cached_subjects(cv["train_sets"], cache)
    train_paths = [s.path for subjects in train.values() for s in subjects]
    bidsleep = discover_cached_subjects(["bidsleep"], adult_root / CACHE_NAME)["bidsleep"]
    divisor = adult_stage_divisor(
        night_counts(train_paths),
        night_counts([s.path for s in bidsleep]),
        training["balance_beta"],
        training["batch_size"],
    )

    stages = []
    staging = output.parent / f"{output.name}-stage"
    for checkpoint in sorted((final / "model_outputs").glob("*.pisces")):
        bundle = build_release_bundle(final, staging / checkpoint.stem, version=version,
                                      checkpoint=checkpoint, family="bidoze", overwrite=True)
        stages.append(bundle.directory)
    build_adult_bundle(
        output, pediatric, stages, divisor, version=version, overwrite=True,
        provenance={
            "arm": arm,
            "train_sets": cv["train_sets"],
            "training_nights": len(train_paths),
            "adult_reference": "bidsleep",
            "adult_reference_nights": len(bidsleep),
            "stage_checkpoints": [p.name for p in stages],
        },
    )
    print(f"{output}: {len(stages)} stage sources; divisor light/deep/REM "
          + " ".join(f"{v:.3f}" for v in divisor[1:]))


if __name__ == "__main__":
    run, arm, adult_root, pediatric, output, version = sys.argv[1:7]
    main(Path(run).resolve(), arm, Path(adult_root).resolve(), Path(pediatric).resolve(),
         Path(output).resolve(), version)

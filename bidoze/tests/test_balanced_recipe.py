"""The BiDoze balanced recipe through the shared driver: split, fit, checkpoint, score."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from arcasleep.data.bundle import CacheBundle  # noqa: E402
from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR  # noqa: E402
from arcasleep.models import checkpoint_family, get_family  # noqa: E402
from arcasleep.training.config import RunConfig  # noqa: E402
from arcasleep.training.driver import RunPaths, run_training  # noqa: E402
from arcasleep_bidoze.training import TrainingConfig  # noqa: E402

BIDOZE = get_family("bidoze")
FRAMES_PER_EPOCH = 3
BALANCED_TRAINING = {
    "recipe": "balanced",
    "run_seed": 0,
    "seed_count": 1,
    "epochs": 2,
    "batch_size": 2,
    "lr": 0.001,
    "weight_decay": 0.0,
    "x_length": 16,
    "y_length": 16,
}


def _write_cache(cache_dir: Path, dataset: str, subjects: list[str], epochs: int) -> None:
    rng = np.random.default_rng(len(dataset))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        # Raw 0-5 PSG labels: wake, N1..N3, REM.
        y = rng.choice([0, 1, 2, 3, 5], size=(1, epochs))
        X = rng.normal(size=(1, epochs * FRAMES_PER_EPOCH, 5, 2))
        CacheBundle(
            X=X, y=y, datasets=np.array([dataset]), subject_ids=np.array([subject])
        ).save_npz(cache_dir / f"{dataset}{SUBJECT_SEPARATOR}{subject}{CACHE_SUFFIX}")


def _configs(directory: Path) -> RunConfig:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "inputs.json": {
            "type": "nufft",
            "cache_prefix": "test",
            "window_seconds": 10,
            "window_step_seconds": 10,
        },
        "model.json": {
            "num_classes": 4,
            "score_gap": True,
            "d_model": 16,
            "n_heads": 2,
            "n_transformer_layers": 1,
            "d_ff": 32,
            "conv_kernels": [3],
            "dropout": 0.0,
        },
        "cv.json": {
            "mode": "transfer",
            "train_sets": ["CohortA"],
            "test_sets": ["CohortB"],
            "deblip_sets": ["CohortA"],
        },
        "training.json": BALANCED_TRAINING,
        "metrics.json": {
            "num_classes": 4,
            "class_names": ["wake", "light", "deep", "rem", "gap"],
            "psg_dt_minutes": 0.5,
            "min_stage_minutes": 5.0,
            "metrics": ["sleep_sensitivity", "wake_specificity"],
        },
    }
    for name, payload in files.items():
        (directory / name).write_text(json.dumps(payload))
    return RunConfig.from_paths(
        family=BIDOZE,
        inputs=directory / "inputs.json",
        model=directory / "model.json",
        cv=directory / "cv.json",
        training=directory / "training.json",
        metrics=directory / "metrics.json",
    )


def test_balanced_recipe_trains_and_scores_end_to_end(tmp_path: Path) -> None:
    config = _configs(tmp_path / "config")
    cache_root = tmp_path / "feature_cache"
    _write_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(6)], epochs=12)
    _write_cache(cache_root / "test", "CohortB", ["b0", "b1"], epochs=9)
    results = tmp_path / "results" / "training"

    run_training(config, RunPaths(results=results, feature_cache=cache_root), run_id="test")

    with (results / "cv_results_4_stages.csv").open(newline="") as handle:
        scored = {row["subject_id"] for row in csv.DictReader(handle)}
    assert scored == {"b0", "b1"}
    (checkpoint,) = (results / "model_outputs").glob("*.pisces")
    assert checkpoint_family(checkpoint) == "bidoze"


@pytest.mark.parametrize(
    "overrides",
    [
        {"deblip_adjustment_epochs": 20, "deblip_logit_weight": 0.5},
        {"sliding_copies": 2},
        {"recipe": "curriculum", "early_stop_patience": 5},
    ],
)
def test_options_the_chosen_recipe_does_not_use_are_refused(overrides: dict) -> None:
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({**BALANCED_TRAINING, **overrides})

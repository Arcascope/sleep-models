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
from arcasleep.training.balanced import Patience  # noqa: E402
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


def _configs(
    directory: Path, model_overrides: dict | None = None, cv_overrides: dict | None = None
) -> RunConfig:
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
            **(model_overrides or {}),
        },
        "cv.json": {
            "mode": "transfer",
            "train_sets": ["CohortA"],
            "test_sets": ["CohortB"],
            **(cv_overrides or {}),
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


@pytest.mark.parametrize("model_frames_per_epoch", [1, FRAMES_PER_EPOCH])
def test_balanced_recipe_trains_and_scores_end_to_end(
    tmp_path: Path, model_frames_per_epoch: int
) -> None:
    config = _configs(tmp_path / "config", {"frames_per_epoch": model_frames_per_epoch})
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

    (fold_dir,) = (results / "predictions").iterdir()
    for subject in ("b0", "b1"):
        with (fold_dir / f"CohortB_{subject}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 9
        assert {row["psg"] for row in rows} <= {"wake", "light", "deep", "rem"}


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


@pytest.mark.parametrize(
    "cv_overrides",
    [{"deblip_sets": ["CohortA"]}, {"validation_sets": ["CohortC"]}],
)
def test_cv_options_the_balanced_recipe_does_not_use_are_refused(
    tmp_path: Path, cv_overrides: dict
) -> None:
    from pisces_lite.cv import Fold, SubjectRef

    config = _configs(tmp_path / "config", cv_overrides=cv_overrides)
    fold = Fold(
        fold_id=0,
        total_folds=1,
        mode="transfer",
        train_subjects=[SubjectRef("CohortA", f"a{i}") for i in range(4)],
        test_subjects=[],
        val_subjects=[],
        description="",
    )

    with pytest.raises(ValueError):
        BIDOZE.train_fold(config, fold=fold, fit_seed=0, split_seed=0, features={}, datasets={})


def test_a_fit_with_no_finite_eval_kappa_refuses_to_pick_an_epoch(tmp_path: Path) -> None:
    from arcasleep.data.constants import GAP_LABEL

    from arcasleep_bidoze.model import BiDozeNetModel

    model = BiDozeNetModel(_configs(tmp_path / "config").model)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 16, 6, 2)).astype(np.float32)
    y = rng.integers(0, 4, size=(4, 16)).astype(np.int32)
    no_scored_epochs = np.full((2, 16), GAP_LABEL, dtype=np.int32)

    with pytest.raises(ValueError):
        model.fit_balanced(
            x, y, x[:2], no_scored_epochs,
            batch_size=2, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
            patience=Patience(lr_drop=1, early_stop=2), max_epochs=3, fit_seed=0,
        )

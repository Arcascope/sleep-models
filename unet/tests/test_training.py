"""The UNet recipe through the shared driver: split, fit, checkpoint, score."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from pisces_lite.cv import SubjectRef  # noqa: E402

from arcasleep.data.bundle import CacheBundle  # noqa: E402
from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR, load_bundles  # noqa: E402
from arcasleep.evaluation.run import RunIdentity, run_evaluation  # noqa: E402
from arcasleep.models import checkpoint_family, get_family  # noqa: E402
from arcasleep.training.config import RunConfig  # noqa: E402
from arcasleep.training.driver import FIT_EVAL_SPLITS_DIRNAME, RunPaths, run_training  # noqa: E402
from arcasleep_unet.training import split_train_eval  # noqa: E402

UNET = get_family("unet")
FRAMES_PER_EPOCH = 3


def _write_cache(cache_dir: Path, dataset: str, subjects: list[str], epochs: int) -> list[Path]:
    rng = np.random.default_rng(len(dataset))
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for subject in subjects:
        # Raw 0-5 PSG labels: wake, N1..N3, REM.
        y = rng.choice([0, 1, 2, 3, 5], size=(1, epochs))
        X = rng.normal(size=(1, epochs * FRAMES_PER_EPOCH, 5, 2))
        path = cache_dir / f"{dataset}{SUBJECT_SEPARATOR}{subject}{CACHE_SUFFIX}"
        CacheBundle(
            X=X, y=y, datasets=np.array([dataset]), subject_ids=np.array([subject])
        ).save_npz(path)
        paths.append(path)
    return paths


def _configs(directory: Path, cv: dict | None = None) -> RunConfig:
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
            "frames_per_epoch": FRAMES_PER_EPOCH,
            "c_u": 4,
            "kernel_temporal": 4,
            "kernel_freq": 3,
            "m_blocks": 2,
            "head_hidden": 8,
        },
        "cv.json": cv or {"mode": "transfer", "train_sets": ["CohortA"], "test_sets": ["CohortB"]},
        "training.json": {
            "run_seed": 0,
            "seed_count": 1,
            "max_epochs": 2,
            "batch_size": 2,
            "lr": 0.001,
            "segment_epochs": 4,
        },
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
        family=UNET,
        inputs=directory / "inputs.json",
        model=directory / "model.json",
        cv=directory / "cv.json",
        training=directory / "training.json",
        metrics=directory / "metrics.json",
    )


def test_eval_split_keeps_a_subject_on_one_side_and_is_seeded() -> None:
    refs = [SubjectRef("a", f"s{i}") for i in range(10)] + [SubjectRef("b", "s0")]

    fit, held = split_train_eval(refs, 0.3, split_seed=0)

    assert sorted(fit + held) == sorted(refs)
    held_ids = {ref.subject_id for ref in held}
    assert not held_ids & {ref.subject_id for ref in fit}
    assert len(held_ids) == 3
    assert split_train_eval(refs, 0.3, split_seed=0) == (fit, held)


def test_training_then_evaluation_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    config = _configs(tmp_path / "config")
    cache_root = tmp_path / "feature_cache"
    _write_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(6)], epochs=10)
    test_paths = _write_cache(cache_root / "test", "CohortB", ["b0", "b1"], epochs=7)
    results = tmp_path / "results" / "training"

    run_training(config, RunPaths(results=results, feature_cache=cache_root), run_id="test")

    with (results / "cv_results_4_stages.csv").open(newline="") as handle:
        scored = {row["subject_id"] for row in csv.DictReader(handle)}
    assert scored == {"b0", "b1"}
    (checkpoint,) = (results / "model_outputs").glob("*.pisces")
    assert checkpoint_family(checkpoint) == "unet"
    split = json.loads((results / FIT_EVAL_SPLITS_DIRNAME / "0.json").read_text())
    assert tuple(
        [SubjectRef(row["dataset"], row["subject_id"]) for row in split[side]]
        for side in ("fit", "eval")
    ) == split_train_eval([SubjectRef("CohortA", f"a{i}") for i in range(6)], 0.3, split_seed=0)

    output_dir = tmp_path / "results" / "evaluation"
    run_evaluation(
        config,
        subjects=load_bundles(test_paths),
        checkpoint=checkpoint,
        output_dir=output_dir,
        identity=RunIdentity("test", "0_seed_0", 2, "CohortA"),
        render_hypnograms=True,
    )
    assert (output_dir / "cv_results_4_stages.csv").is_file()
    assert len(list((output_dir / "hypnograms").glob("*.png"))) == 2


def test_cv_options_the_recipe_does_not_use_are_refused(tmp_path: Path) -> None:
    from pisces_lite.cv import Fold

    config = _configs(
        tmp_path / "config",
        cv={
            "mode": "transfer",
            "train_sets": ["CohortA"],
            "test_sets": ["CohortB"],
            "deblip_sets": ["CohortA"],
        },
    )
    fold = Fold(
        fold_id=0,
        total_folds=1,
        mode="transfer",
        train_subjects=[SubjectRef("CohortA", "a0")],
        test_subjects=[],
        val_subjects=[],
        description="",
    )

    with pytest.raises(ValueError):
        UNET.train_fold(config, fold=fold, fit_seed=0, split_seed=0, features={}, datasets={})

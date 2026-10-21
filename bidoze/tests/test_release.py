"""Retraining the releases: the adult phases, and a final phase's release bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from arcasleep.data.bundle import CacheBundle  # noqa: E402
from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR  # noqa: E402
from arcasleep.models import get_family  # noqa: E402
from arcasleep.training.config import RunConfig  # noqa: E402
from arcasleep.training.driver import RunPaths, run_training  # noqa: E402
from arcasleep_bidoze.release import (  # noqa: E402
    CACHE_NAME,
    build_release,
    prepare_adult,
    weaver_dataset_name,
)

FRAMES_PER_EPOCH = 3
EPOCHS = 12
TINY_TRAINING = {
    "recipe": "balanced", "run_seed": 0, "seed_count": 2, "epochs": 2, "batch_size": 2, "lr": 0.001,
    "weight_decay": 0.0, "x_length": 16, "y_length": 16, "balance_beta": 0.999, "eval_fraction": 0.3,
}
# Revised-Weaver recordings: three children, two with a second device on the same night.
REVISED = ["kid1_fitbit_n1", "kid1_apple_n1", "kid2_fitbit_n1", "kid2_actigraph_n1", "kid3_apple_n1"]
EARLIER_NIGHTS = {"early1": "kid1_fitbit_n1", "early2": "kid2_fitbit_n1", "early3": "kid3_apple_n1"}
SLEEPACCEL = ["sa0", "sa1", "sa2", "sa3"]
CLINICAL = ["cl0", "cl1"]


def _save(directory: Path, dataset: str, subject: str, y: np.ndarray, X: np.ndarray) -> None:
    CacheBundle(
        X=X[None], y=y[None], datasets=np.array([dataset]), subject_ids=np.array([subject])
    ).save_npz(directory / f"{dataset}{SUBJECT_SEPARATOR}{subject}{CACHE_SUFFIX}")


def _feature_root(tmp_path: Path) -> Path:
    cache = tmp_path / "root" / CACHE_NAME
    cache.mkdir(parents=True)
    rng = np.random.default_rng(0)
    nights = {}
    for sid in REVISED:
        night = sid.split("_")[0]
        if night not in nights:
            nights[night] = (rng.choice([0, 1, 2, 3, 5], size=EPOCHS), rng.normal(size=(EPOCHS * FRAMES_PER_EPOCH, 5, 2)))
        y, X = nights[night]
        # Another device of the same night has the same labels and slightly different spectra.
        offset = 0.0 if "fitbit" in sid or sid.startswith("kid3") else 0.5
        _save(cache, "weaver_revised", sid, y, X + offset)
    for early, revised in EARLIER_NIGHTS.items():
        y, X = nights[revised.split("_")[0]]
        _save(cache, "weaver", early, y, X)
    for sid in SLEEPACCEL:
        _save(cache, "SleepAccel", sid, rng.choice([0, 1, 2, 3, 5], size=EPOCHS), rng.normal(size=(EPOCHS * FRAMES_PER_EPOCH, 5, 2)))
    for sid in CLINICAL + ["ref0", "ref1"]:
        dataset = "SleepAccel-Clinical" if sid.startswith("cl") else "bidsleep"
        _save(cache, dataset, sid, rng.choice([0, 1, 2, 3, 5], size=EPOCHS), rng.normal(size=(EPOCHS * FRAMES_PER_EPOCH, 5, 2)))
    return tmp_path / "root"


def _config_dir(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    (config / "splits").mkdir(parents=True)
    plan = {
        "schema_version": 1, "cohort": "SleepAccel", "n_splits": 2, "seed": 0,
        "folds": {"BiDozeSleepAccelFold0": SLEEPACCEL[:2], "BiDozeSleepAccelFold1": SLEEPACCEL[2:]},
    }
    (config / "splits/sleepaccel_5fold.json").write_text(json.dumps(plan))
    (config / "training-adult.json").write_text(json.dumps(TINY_TRAINING))
    return config


@pytest.mark.parametrize("weaver_set, expected", [("matched", sorted(EARLIER_NIGHTS.values())), ("all", sorted(REVISED))])
def test_the_adult_phases_train_on_the_chosen_weaver_set(tmp_path: Path, weaver_set: str, expected: list[str]) -> None:
    setup = tmp_path / "setup"

    chosen = prepare_adult(setup, _feature_root(tmp_path), weaver_set, _config_dir(tmp_path))

    assert chosen == expected
    name = weaver_dataset_name(weaver_set)
    linked = {p.name.split(SUBJECT_SEPARATOR)[1][: -len(CACHE_SUFFIX)] for p in (setup / "cache" / CACHE_NAME).glob(f"{name}*")}
    assert linked == set(expected)
    final = json.loads((setup / "cv/cv_final.json").read_text())
    assert final["train_sets"][0] == name
    assert final["test_sets"] == ["SleepAccel-Clinical"]
    fold0 = json.loads((setup / "cv/cv_fold0.json").read_text())
    assert fold0["test_sets"][0] == "BiDozeSleepAccelFold0"
    assert "BiDozeSleepAccelFold0" not in fold0["train_sets"]


def test_an_unknown_weaver_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prepare_adult(tmp_path / "setup", _feature_root(tmp_path), "some", _config_dir(tmp_path))


def _final_phase(tmp_path: Path, setup: Path, training: dict) -> Path:
    config = tmp_path / "run-config"
    config.mkdir()
    files = {
        "inputs.json": {"type": "nufft", "cache_prefix": CACHE_NAME, "window_seconds": 10, "window_step_seconds": 10},
        "model.json": {"num_classes": 4, "score_gap": True, "gap_from_row": True, "d_model": 16, "n_heads": 2,
                       "n_transformer_layers": 1, "d_ff": 32, "conv_kernels": [3], "dropout": 0.0},
        "training.json": training,
        "metrics.json": {"num_classes": 4, "class_names": ["wake", "light", "deep", "rem", "gap"],
                         "psg_dt_minutes": 0.5, "min_stage_minutes": 5.0, "metrics": ["sleep_sensitivity"]},
    }
    for name, payload in files.items():
        (config / name).write_text(json.dumps(payload))
    run = RunConfig.from_paths(
        family=get_family("bidoze"), inputs=config / "inputs.json", model=config / "model.json",
        cv=setup / "cv/cv_final.json", training=config / "training.json", metrics=config / "metrics.json",
    )
    results = tmp_path / "results"
    run_training(run, RunPaths(results=results, feature_cache=setup / "cache"), run_id="final")
    return results


def test_a_final_phase_builds_the_adult_bundle_with_one_bundle_per_seed(tmp_path: Path) -> None:
    root = _feature_root(tmp_path)
    setup = tmp_path / "setup"
    prepare_adult(setup, root, "all", _config_dir(tmp_path))
    final = _final_phase(tmp_path, setup, TINY_TRAINING)

    bundle = build_release("adult", final, tmp_path / "bundle", "v0.0.0",
                           train_cache=setup / "cache", cache_root=root)

    composite = json.loads((bundle / "composite.json").read_text())
    assert composite["seeds"] == ["seed0", "seed1"]
    assert all(v > 0 for v in composite["stage_divisor"].values())
    from arcasleep.predict import load_predictor

    assert load_predictor(bundle) is not None


def test_a_final_phase_builds_the_pediatric_bundle(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    prepare_adult(setup, _feature_root(tmp_path), "all", _config_dir(tmp_path))
    final = _final_phase(tmp_path, setup, {**TINY_TRAINING, "seed_count": 1})

    bundle = build_release("pediatric", final, tmp_path / "bundle", "v0.0.0")

    assert (bundle / "model.safetensors").is_file()
    assert json.loads((bundle / "config.json").read_text())["gap_from_row"] is True

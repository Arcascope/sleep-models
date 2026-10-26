"""Retraining the releases: the adult phases, and a final phase's release bundle."""
from __future__ import annotations

import json
import shutil
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
    ADULT_REFERENCE_PREFIXES,
    CACHE_NAME,
    adult_card_metrics,
    build_release,
    phase_divisor,
    prepare_adult,
    weaver_dataset_name,
    write_adult_card,
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
WEARANIZE = ["w0", "w1", "w2", "w3"]
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
    for sid in WEARANIZE:
        _save(cache, "wearanize", sid, rng.choice([0, 1, 2, 3, 5], size=EPOCHS), rng.normal(size=(EPOCHS * FRAMES_PER_EPOCH, 5, 2)))
    for sid in SLEEPACCEL:
        _save(cache, "SleepAccel", sid, rng.choice([0, 1, 2, 3, 5], size=EPOCHS), rng.normal(size=(EPOCHS * FRAMES_PER_EPOCH, 5, 2)))
    for sid in CLINICAL:
        _save(cache, "SleepAccel-Clinical", sid, rng.choice([0, 1, 2, 3, 5], size=EPOCHS), rng.normal(size=(EPOCHS * FRAMES_PER_EPOCH, 5, 2)))
    return tmp_path / "root"


def _config_dir(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    (config / "splits").mkdir(parents=True)
    plan = {
        "schema_version": 1, "cohort": "SleepAccel", "n_splits": 2, "seed": 0,
        "folds": {"BiDozeSleepAccelFold0": SLEEPACCEL[:2], "BiDozeSleepAccelFold1": SLEEPACCEL[2:]},
    }
    (config / "splits/sleepaccel_5fold.json").write_text(json.dumps(plan))
    wearanize = {
        "schema_version": 1, "cohort": "wearanize", "n_splits": 2, "seed": 0,
        "folds": {"BiDozeWearanizeFold0": WEARANIZE[:2], "BiDozeWearanizeFold1": WEARANIZE[2:]},
    }
    (config / "splits/wearanize_5fold.json").write_text(json.dumps(wearanize))
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
    assert {"BiDozeWearanizeFold0", "BiDozeWearanizeFold1"} <= set(final["train_sets"])
    assert final["test_sets"] == ["SleepAccel-Clinical"]
    fold0 = json.loads((setup / "cv/cv_fold0.json").read_text())
    assert fold0["test_sets"] == ["BiDozeSleepAccelFold0", "BiDozeWearanizeFold0", "SleepAccel-Clinical"]
    assert fold0["train_sets"] == [name, "BiDozeSleepAccelFold1", "BiDozeWearanizeFold1"]
    wearanize_links = {p.name for p in (setup / "cache" / CACHE_NAME).glob("BiDozeWearanize*")}
    assert len(wearanize_links) == len(WEARANIZE)


def test_an_unknown_weaver_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prepare_adult(tmp_path / "setup", _feature_root(tmp_path), "some", _config_dir(tmp_path))


F2_MODEL = {"num_classes": 4, "score_gap": True, "gap_from_row": True, "n_heads": 2, "n_transformer_layers": 1,
            "conv_kernels": [3], "dropout": 0.0, "strands": 2, "strand_width": 8, "strand_ffn": 8,
            "fusion_proj": 4, "fusion_width": 8, "fusion_layers": 1, "fusion_heads": 2, "fusion_ffn": 8}


def _final_phase(
    tmp_path: Path, setup: Path, training: dict, model: dict | None = None, phase: str = "final"
) -> Path:
    config = tmp_path / f"run-config-{phase}"
    config.mkdir()
    files = {
        "inputs.json": {"type": "nufft", "cache_prefix": CACHE_NAME, "window_seconds": 10, "window_step_seconds": 10},
        "model.json": model or {"num_classes": 4, "score_gap": True, "gap_from_row": True, "d_model": 16,
                                "n_heads": 2, "n_transformer_layers": 1, "d_ff": 32, "conv_kernels": [3],
                                "dropout": 0.0},
        "training.json": training,
        "metrics.json": {"num_classes": 4, "class_names": ["wake", "light", "deep", "rem", "gap"],
                         "psg_dt_minutes": 0.5, "min_stage_minutes": 5.0, "metrics": ["sleep_sensitivity", "wake_specificity", "kappa_sleep_wake"]},
    }
    for name, payload in files.items():
        (config / name).write_text(json.dumps(payload))
    run = RunConfig.from_paths(
        family=get_family("bidoze"), inputs=config / "inputs.json", model=config / "model.json",
        cv=setup / f"cv/cv_{phase}.json", training=config / "training.json", metrics=config / "metrics.json",
    )
    results = tmp_path / "results" / phase
    run_training(run, RunPaths(results=results, feature_cache=setup / "cache"), run_id=phase)
    return results


def _members(bundle: Path) -> list[dict]:
    return json.loads((bundle / "provenance.json").read_text())["members"]


def _tensor_prefixes(bundle: Path) -> set[str]:
    from safetensors import safe_open

    with safe_open(str(bundle / "model.safetensors"), framework="np") as handle:
        return {key.split("/", 1)[0] for key in handle.keys()}


def test_a_final_phase_builds_the_flat_adult_bundle_of_its_seeds(tmp_path: Path) -> None:
    root = _feature_root(tmp_path)
    setup = tmp_path / "setup"
    prepare_adult(setup, root, "all", _config_dir(tmp_path))
    final = _final_phase(tmp_path, setup, TINY_TRAINING)

    bundle = build_release("adult", final, tmp_path / "bundle", "v0.0.0",
                           train_cache=setup / "cache")

    config = json.loads((bundle / "config.json").read_text())
    assert config["members"] == 2
    assert all(v > 0 for v in config["stage_divisor"].values())
    assert [m["seed"] for m in _members(bundle)] == [0, 1]
    assert _tensor_prefixes(bundle) == {"member0", "member1"}
    assert not (bundle / "composite.json").exists()
    from arcasleep.artifacts.publish import verify_bundle
    from arcasleep.predict import Predictor

    verify_bundle(bundle)
    assert Predictor(bundle) is not None


@pytest.mark.parametrize("seeds, expected", [(None, [0, 1]), ([1], [1])])
def test_an_f2_final_phase_builds_the_adult_bundle_of_all_or_chosen_seeds(
    tmp_path: Path, seeds: list[int] | None, expected: list[int]
) -> None:
    root = _feature_root(tmp_path)
    setup = tmp_path / "setup"
    prepare_adult(setup, root, "all", _config_dir(tmp_path))
    final = _final_phase(tmp_path, setup, TINY_TRAINING, model=F2_MODEL)

    bundle = build_release("adult", final, tmp_path / "bundle", "v0.0.0",
                           train_cache=setup / "cache", seeds=seeds)

    config = json.loads((bundle / "config.json").read_text())
    assert [m["seed"] for m in _members(bundle)] == expected
    assert config["members"] == len(expected)
    assert config["strands"] == 2
    from arcasleep.predict import Predictor

    assert Predictor(bundle) is not None


def _adult_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A tiny adult run: ``training/fold0``, ``fold1`` and ``final`` phases, the setup cache and the bundle."""
    root = _feature_root(tmp_path)
    setup = tmp_path / "setup"
    prepare_adult(setup, root, "all", _config_dir(tmp_path))
    training = tmp_path / "training"
    for phase in ("fold0", "fold1", "final"):
        results = _final_phase(tmp_path, setup, TINY_TRAINING, model=F2_MODEL, phase=phase)
        shutil.move(results, training / phase)
    bundle = build_release("adult", training / "final", tmp_path / "bundle", "v0.0.0",
                           train_cache=setup / "cache")
    return training, setup / "cache", bundle


def test_the_adult_divisor_comes_from_the_adult_training_sets(tmp_path: Path) -> None:
    training, cache, bundle = _adult_run(tmp_path)

    provenance = json.loads((bundle / "provenance.json").read_text())

    assert all(name.startswith(ADULT_REFERENCE_PREFIXES) for name in provenance["stage_divisor_reference_sets"])
    assert provenance["stage_divisor_reference_nights"] == len(SLEEPACCEL) + len(WEARANIZE)
    divisor, train, adult = phase_divisor(training / "fold0", cache)
    assert len(adult) == len(SLEEPACCEL) // 2 + len(WEARANIZE) // 2
    assert len(train) == len(REVISED) + len(adult)
    assert (divisor > 0).all()


def test_the_adult_card_is_scored_per_cohort_and_written_into_a_verified_bundle(tmp_path: Path) -> None:
    import csv

    from arcasleep.artifacts.publish import verify_bundle

    training, cache, bundle = _adult_run(tmp_path)
    info = tmp_path / "info.toml"
    info.write_text('repo = "org/adult"\naudience = "Adults"\nheadline = "SleepAccel"\n'
                    '[cohorts]\nSleepAccel = "a"\n')

    summary = adult_card_metrics(training, bundle, tmp_path / "summary.csv", train_cache=cache)
    readme = write_adult_card(bundle, summary, info)

    with summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["cohort"] for r in rows} == {"SleepAccel", "Wearanize+", "SleepAccel-Clinical"}
    counts = {r["cohort"]: int(float(r["n"])) for r in rows if r["metric"] == "sleep_sensitivity"}
    assert counts == {"SleepAccel": len(SLEEPACCEL), "Wearanize+": len(WEARANIZE),
                      "SleepAccel-Clinical": len(CLINICAL)}
    assert readme == bundle / "README.md" and readme.is_file()
    verify_bundle(bundle)


def test_a_seed_the_phase_did_not_train_is_refused(tmp_path: Path) -> None:
    root = _feature_root(tmp_path)
    setup = tmp_path / "setup"
    prepare_adult(setup, root, "all", _config_dir(tmp_path))
    final = _final_phase(tmp_path, setup, TINY_TRAINING, model=F2_MODEL)

    with pytest.raises(ValueError):
        build_release("adult", final, tmp_path / "bundle", "v0.0.0",
                      train_cache=setup / "cache", seeds=[7])


def test_a_final_phase_builds_the_pediatric_bundle(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    prepare_adult(setup, _feature_root(tmp_path), "all", _config_dir(tmp_path))
    final = _final_phase(tmp_path, setup, {**TINY_TRAINING, "seed_count": 1})

    bundle = build_release("pediatric", final, tmp_path / "bundle", "v0.0.0")

    assert (bundle / "model.safetensors").is_file()
    assert json.loads((bundle / "config.json").read_text())["gap_from_row"] is True


def test_the_local_entry_point_trains_a_tiny_final_phase_and_writes_the_bundle(tmp_path: Path) -> None:
    import os
    import shutil
    import sys

    from arcasleep_bidoze.release import run_release

    bidoze = Path(__file__).resolve().parents[1]
    local = tmp_path / "bidoze"
    (local / "scripts").mkdir(parents=True)
    shutil.copy(bidoze / "scripts/harness.sh", local / "scripts/harness.sh")
    config = _config_dir(tmp_path)
    shutil.copytree(config, local / "config")
    (local / "config/inputs.json").write_text(json.dumps(
        {"type": "nufft", "cache_prefix": CACHE_NAME, "window_seconds": 10, "window_step_seconds": 10}))
    (local / "config/metrics.json").write_text(json.dumps(
        {"num_classes": 4, "class_names": ["wake", "light", "deep", "rem", "gap"],
         "psg_dt_minutes": 0.5, "min_stage_minutes": 5.0, "metrics": ["sleep_sensitivity"]}))
    (local / "config/model-gap-rule.json").write_text(json.dumps(
        {"num_classes": 4, "score_gap": True, "gap_from_row": True, "d_model": 16, "n_heads": 2,
         "n_transformer_layers": 1, "d_ff": 32, "conv_kernels": [3], "dropout": 0.0}))
    (local / "config/model-f2.json").write_text(json.dumps(F2_MODEL))
    (local / "config/cv-balanced.json").write_text("{}")
    path = os.environ["PATH"]
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{path}"
    os.environ["BIDOZE_WORK_DIR"] = str(tmp_path / "work")
    try:
        status = run_release(
            "adult", run_id="tiny", cache_root=_feature_root(tmp_path), weaver_set="all",
            phases=["final"], results_dir=tmp_path / "results", bidoze_dir=local,
        )
    finally:
        os.environ["PATH"] = path
        del os.environ["BIDOZE_WORK_DIR"]

    assert status == 0
    bundle = tmp_path / "results/tiny/training/final/bundle"
    assert json.loads((bundle / "config.json").read_text())["members"] == 2


def test_the_training_image_installs_the_extras_a_release_bundle_needs() -> None:
    import re

    dockerfile = (Path(__file__).resolve().parents[1] / "docker/train.Dockerfile").read_text()
    (extras,) = re.findall(r'pip install[^\n]*"\.\[([^\]]+)\]"', dockerfile)

    assert {"bidoze", "train", "aws", "publish"} <= set(extras.split(","))

"""The Wearanize+ training experiment: the split plan, each arm's fit setup and the paired comparison."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR  # noqa: E402
from arcasleep.data.splits import SplitPlan, fold_dataset_name, read_plan, write_plan  # noqa: E402
from test_release import CACHE_NAME, EPOCHS, FRAMES_PER_EPOCH, TINY_TRAINING, _feature_root, _save  # noqa: E402

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/wearanize_training"
sys.path.insert(0, str(EXPERIMENT))
import make_plan  # noqa: E402
import prepare  # noqa: E402


def _load_summarize():
    sys.path.insert(0, str(EXPERIMENT.parents[0] / "adult_devices"))
    spec = importlib.util.spec_from_file_location("wearanize_summarize", EXPERIMENT / "analysis/summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRAW = {"trainSleepAccel": ["sa0", "sa1"], "trainClinical": ["cl0"], "testSleepAccel": ["sa2", "sa3"], "testClinical": ["cl1"]}
WEARANIZE = [f"sub{i:03d}_s1" for i in range(10)]
AAUWSS = ["s01", "s02", "s03"]


def _root(tmp_path: Path) -> tuple[Path, Path]:
    root = _feature_root(tmp_path)
    rng = np.random.default_rng(1)
    for dataset, ids in (("wearanize", WEARANIZE), ("aauwss", AAUWSS)):
        for sid in ids:
            _save(root / CACHE_NAME, dataset, sid, rng.choice([0, 1, 2, 3, 5], size=EPOCHS),
                  rng.normal(size=(EPOCHS * FRAMES_PER_EPOCH, 5, 2)))
    folds = {fold_dataset_name("wearanize", i): WEARANIZE[i::5] for i in range(5)}
    plan = tmp_path / "plan.json"
    write_plan(SplitPlan(cohort="wearanize", n_splits=5, folds=folds, seed=0), plan)
    return root, plan


def _linked(fit: Path, prefix: str) -> set[str]:
    return {p.name.split(SUBJECT_SEPARATOR)[1][: -len(CACHE_SUFFIX)]
            for p in (fit / "cache" / CACHE_NAME).glob(f"{prefix}{SUBJECT_SEPARATOR}*")}


def test_the_committed_plan_puts_each_of_92_nights_in_one_of_5_folds() -> None:
    plan = read_plan(prepare.PLAN)
    sizes = sorted(len(ids) for ids in plan.folds.values())
    assert plan.cohort == "wearanize" and plan.n_splits == 5
    assert len(plan.subjects()) == 92 and sizes[-1] - sizes[0] <= 1


def test_the_stratified_folds_have_near_equal_size_and_wake_share() -> None:
    shares = {f"s{i:03d}": i / 92 for i in range(92)}
    folds = make_plan.stratified_folds(shares, 5, 0)
    assert sorted(sid for fold in folds for sid in fold) == sorted(shares)
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    means = [np.mean([shares[s] for s in fold]) for fold in folds]
    assert max(means) - min(means) < 0.05
    assert folds == make_plan.stratified_folds(shares, 5, 0)


@pytest.mark.parametrize("arm, with_wearanize", [
    ("g-base", False), ("g-wearanize", True), ("f2-base", False), ("f2-wearanize", True)])
def test_each_arm_trains_on_the_other_folds_and_is_tested_on_the_held_out_ones(
    tmp_path: Path, arm: str, with_wearanize: bool
) -> None:
    root, plan = _root(tmp_path)
    training = tmp_path / "training.json"
    training.write_text(json.dumps(TINY_TRAINING))
    fit = tmp_path / "fit"

    prepare.prepare(fit, root, arm, DRAW, 2, training, plan)

    held_out = set(WEARANIZE[2::5])
    assert _linked(fit, "testWearanize") == held_out
    assert _linked(fit, "testAAUWSS") == set(AAUWSS)
    assert _linked(fit, "trainWearanize") == (set(WEARANIZE) - held_out if with_wearanize else set())
    assert len(_linked(fit, "weaverAll")) == 5
    cv = json.loads((fit / "cv.json").read_text())
    expected_train = ["weaverAll", "trainSleepAccel", "trainClinical"] + (["trainWearanize"] if with_wearanize else [])
    assert cv["train_sets"] == expected_train
    assert set(cv["train_sets"]).isdisjoint(cv["test_sets"])
    assert {"testSleepAccel", "testClinical", "testWearanize", "testAAUWSS"} <= set(cv["test_sets"])
    assert prepare.model_config(arm) == ("config/model-f2.json" if arm.startswith("f2") else "config/model-gap-rule.json")


def test_an_unknown_arm_is_refused(tmp_path: Path) -> None:
    root, plan = _root(tmp_path)
    with pytest.raises(ValueError):
        prepare.prepare(tmp_path / "fit", root, "all", DRAW, 0, None, plan)


def test_the_paired_comparison_gives_the_difference_of_the_draw_means() -> None:
    summarize = _load_summarize()
    rows = []
    for draw in range(5):
        for arm, kappa in (("g-base", 0.50), ("g-wearanize", 0.52 + 0.01 * draw), ("f2-base", 0.51), ("f2-wearanize", 0.51)):
            rows.append({"draw": draw, "arm": arm, "cohort": "AAUWSS", "decoding": "hybrid",
                         **{m: 0.0 for m in summarize.METRICS}, "kappa_4stage": kappa})
    result = summarize.paired(pd.DataFrame(rows), "draw")
    row = result[(result.comparison == "g-wearanize - g-base") & (result.metric == "kappa_4stage")].iloc[0]
    assert row["n"] == 5 and row["higher"] == 5
    assert row["mean"] == pytest.approx(0.04) and row["ci_low"] < row["mean"] < row["ci_high"]
    flat = result[(result.comparison == "f2-wearanize - f2-base") & (result.metric == "kappa_4stage")].iloc[0]
    assert flat["higher"] == 0 and flat["sd"] == 0

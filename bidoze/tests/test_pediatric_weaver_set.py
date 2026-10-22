"""The pediatric release's Weaver set: the revised-Weaver split plan and the recipe that reads it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from arcasleep.data.cache import participant_id
from arcasleep.data.splits import SplitPlan, cv_configs_for_plan, fold_dataset_name, read_plan
from arcasleep_bidoze.release import WEAVER_SETS, pediatric_recipe

BIDOZE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BIDOZE / "experiments/pediatric_weaver_set"))
import plan as revised_plan  # noqa: E402


def _labels(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice([0, 1, 2, 3], size=40)


def _revised_cohort(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path], SplitPlan]:
    """Six children with two devices each; earlier nights for the first four, in two folds."""
    revised, earlier = {}, {}
    for c in range(6):
        for device in ("fitbit", "apple"):
            sid = f"kid{c}_{device}_n1"
            revised[sid] = tmp_path / f"{sid}.npz"
            np.savez(revised[sid], y=_labels(c)[None])
    for c in range(4):
        earlier[f"early{c}"] = tmp_path / f"early{c}.npz"
        np.savez(earlier[f"early{c}"], y=_labels(c)[None])
    folds = {fold_dataset_name("weaver", 0): ["early0", "early1"], fold_dataset_name("weaver", 1): ["early2", "early3"]}
    return revised, earlier, SplitPlan(cohort="weaver", n_splits=2, folds=folds, seed=0)


def test_children_of_the_earlier_plan_keep_their_fold(tmp_path: Path) -> None:
    revised, earlier, earlier_plan = _revised_cohort(tmp_path)
    pinned, split = revised_plan.earlier_children(earlier_plan, earlier, revised)
    assert pinned == {"kid0": 0, "kid1": 0, "kid2": 1, "kid3": 1}
    assert split == []
    folds = revised_plan.extend_plan(list(revised), pinned, 2, seed=0)
    for child, fold in pinned.items():
        assert all(participant_id(s) != child or s in folds[fold] for s in revised)
    assert sorted(s for sids in folds.values() for s in sids) == sorted(revised)


def test_a_child_with_nights_in_two_earlier_folds_is_reported(tmp_path: Path) -> None:
    revised, earlier, earlier_plan = _revised_cohort(tmp_path)
    np.savez(earlier["early3"], y=_labels(0)[None])
    pinned, split = revised_plan.earlier_children(earlier_plan, earlier, revised)
    assert split == ["kid0"]
    assert pinned["kid0"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_new_children_balance_the_recordings_and_stay_whole(seed: int) -> None:
    recordings = [f"kid{c}_{d}_n1" for c in range(30) for d in ("a", "b", "c")[: 1 + c % 3]]
    pinned = {f"kid{c}": c % 5 for c in range(10)}
    folds = revised_plan.extend_plan(recordings, pinned, 5, seed)
    fold_of = {}
    for fold, sids in folds.items():
        for sid in sids:
            assert fold_of.setdefault(participant_id(sid), fold) == fold
    assert len(fold_of) == 30
    assert all(fold_of[child] == fold for child, fold in pinned.items())
    sizes = [len(sids) for sids in folds.values()]
    assert max(sizes) - min(sizes) <= 3


def test_the_committed_plan_has_no_child_in_two_folds() -> None:
    plan = read_plan(BIDOZE / revised_plan.OUTPUT)
    fold_of = {}
    for name in plan.fold_names:
        for sid in plan.folds[name]:
            assert fold_of.setdefault(participant_id(sid), name) == name
    assert plan.cohort == "weaver_revised"
    assert len(plan.subjects()) == plan.provenance["subject_count"]


def test_the_pediatric_recipe_of_each_weaver_set_reads_its_own_cache() -> None:
    for weaver_set in WEAVER_SETS:
        plan_path, cv_path = pediatric_recipe(weaver_set)
        plan = read_plan(BIDOZE / plan_path)
        template = json.loads((BIDOZE / cv_path).read_text())
        assert template["train_sets"] == [plan.cohort]
        configs = cv_configs_for_plan(plan, template, external_test_sets=["SleepAccel", "SleepAccel-Clinical"])
        assert sorted(configs) == sorted([f"fold{k}" for k in range(5)] + ["final"])
        assert configs["final"]["train_sets"] == plan.fold_names
    assert read_plan(BIDOZE / pediatric_recipe("all")[0]).cohort == "weaver_revised"
    assert read_plan(BIDOZE / pediatric_recipe("matched")[0]).cohort == "weaver"


def test_an_unknown_weaver_set_has_no_pediatric_recipe() -> None:
    with pytest.raises(ValueError):
        pediatric_recipe("some")


def test_the_harness_refuses_the_replaced_adult_weaver_set_variable() -> None:
    import os
    import subprocess

    env = {**os.environ, "BIDOZE_RELEASE": "adult", "BIDOZE_ADULT_WEAVER_SET": "all"}
    done = subprocess.run(["bash", str(BIDOZE / "scripts/harness.sh"), "--dry-run"], env=env,
                          capture_output=True)
    assert done.returncode != 0
    env.pop("BIDOZE_ADULT_WEAVER_SET")
    env["BIDOZE_WEAVER_SET"] = "all"
    assert subprocess.run(["bash", str(BIDOZE / "scripts/harness.sh"), "--dry-run"], env=env,
                          capture_output=True).returncode == 0

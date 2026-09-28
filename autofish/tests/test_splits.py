import json

import pytest

from autofish.data.splits import (
    DATA_DIGEST_KEY,
    SplitPlan,
    cv_configs_for_plan,
    data_drift_warning,
    fold_dataset_name,
    plan_splits,
    read_plan,
    write_plan,
)

IDS = [f"subject_{i:03d}" for i in range(23)]
TEMPLATE = {
    "mode": "transfer",
    "train_sets": ["weaver"],
    "deblip_sets": ["weaver"],
    "test_sets": ["SleepAccel", "weaver"],
    "validation_sets": [],
    "n_splits": 5,
}


def test_fold_names_capitalise_the_cohort() -> None:
    assert fold_dataset_name("weaver", 0) == "AutofishWeaverFold0"


def test_plan_partitions_the_cohort_into_near_equal_folds() -> None:
    plan = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    sizes = [len(plan.folds[n]) for n in plan.fold_names]
    assert sum(sizes) == len(IDS)
    assert max(sizes) - min(sizes) <= 1
    assert plan.subjects() == sorted(IDS)  # every subject exactly once


def test_plan_is_reproducible_from_the_same_seed_and_cohort() -> None:
    a = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    b = plan_splits(list(reversed(IDS)), cohort="weaver", n_splits=5, seed=0)
    assert a.folds == b.folds  # input order must not matter
    assert plan_splits(IDS, cohort="weaver", n_splits=5, seed=1).folds != a.folds


def test_plan_rejects_a_subject_in_two_folds() -> None:
    bad = SplitPlan(
        cohort="weaver",
        n_splits=2,
        folds={"AutofishWeaverFold0": ["a", "b"], "AutofishWeaverFold1": ["b"]},
        seed=0,
    )
    with pytest.raises(ValueError, match="in both"):
        bad.validate()


def test_plan_round_trips_through_json(tmp_path) -> None:
    plan = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    path = write_plan(plan, tmp_path / "plan.json")
    assert read_plan(path).folds == plan.folds


def test_plan_rejects_an_unknown_schema_version(tmp_path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema_version": 99, "cohort": "w", "n_splits": 2, "folds": {}}))
    with pytest.raises(ValueError, match="schema_version"):
        read_plan(path)


def test_fold_configs_hold_out_one_fold_and_never_train_on_it() -> None:
    plan = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    configs = cv_configs_for_plan(plan, TEMPLATE, external_test_sets=["SleepAccel"])
    assert sorted(configs) == ["final", "fold0", "fold1", "fold2", "fold3", "fold4"]
    for i in range(5):
        cfg = configs[f"fold{i}"]
        held = fold_dataset_name("weaver", i)
        assert cfg["test_sets"] == [held]
        assert held not in cfg["train_sets"]
        assert held not in cfg["deblip_sets"]  # deblip is exposure too
        assert not set(cfg["train_sets"]) & set(cfg["test_sets"])
        assert "n_splits" not in cfg  # k is split policy, not a training input
        # Each phase is a separate run whose single fold is pisces-lite's
        # fold 0; the label is what keeps their results distinguishable.
        assert cfg["fold_label"] == f"fold{i}"


def test_final_config_trains_on_every_fold_and_scores_only_external_sets() -> None:
    plan = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    final = cv_configs_for_plan(plan, TEMPLATE, external_test_sets=["SleepAccel"])["final"]
    assert final["train_sets"] == plan.fold_names
    assert final["test_sets"] == ["SleepAccel"]
    assert final["fold_label"] == "final"
    assert not set(final["train_sets"]) & set(final["test_sets"])


def test_deblip_only_datasets_ride_along_in_every_phase() -> None:
    plan = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    template = dict(TEMPLATE, deblip_sets=["weaver", "FoundryTakeda"])
    configs = cv_configs_for_plan(plan, template, external_test_sets=["SleepAccel"])
    for name, cfg in configs.items():
        assert "FoundryTakeda" in cfg["deblip_sets"], name
        assert "weaver" not in cfg["deblip_sets"]
        assert "FoundryTakeda" not in cfg["test_sets"]


def test_materialise_rejects_a_passthrough_dataset_with_no_cached_subjects(tmp_path) -> None:
    from autofish.data.splits import materialise

    plan = SplitPlan(cohort="weaver", n_splits=0, folds={}, seed=0)
    with pytest.raises(FileNotFoundError, match="Foundry"):
        materialise(plan, {"weaver": []}, tmp_path, passthrough=["Foundry"])


def test_fold_label_reaches_the_id_used_for_checkpoints_and_results():
    from pisces_lite.cv import Fold

    from autofish.training.config import AutofishCVConfig
    from autofish.training.driver import _fold_label

    plan = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    configs = cv_configs_for_plan(plan, TEMPLATE, external_test_sets=["SleepAccel"])

    def _fold(total_folds=1, fold_id=0):
        return Fold(
            fold_id=fold_id,
            total_folds=total_folds,
            mode="transfer",
            description="",
            train_subjects=[],
            val_subjects=[],
            test_subjects=[],
        )

    class _Run:
        def __init__(self, cv):
            self.cv = cv

    cv2 = AutofishCVConfig.from_dict(dict(configs["fold2"]))
    assert _fold_label(_Run(cv2), _fold()) == "fold2"

    final = AutofishCVConfig.from_dict(dict(configs["final"]))
    assert _fold_label(_Run(final), _fold()) == "final"

    # Unlabelled configs keep the historical bare fold id.
    bare = AutofishCVConfig.from_dict(
        {"mode": "transfer", "train_sets": ["a"], "test_sets": ["b"]}
    )
    assert _fold_label(_Run(bare), _fold()) == "0"

    # A multi-fold phase still needs the fold number to stay unique.
    assert _fold_label(_Run(cv2), _fold(total_folds=3, fold_id=1)) == "fold2_1"


def _plan_with_digest(digest):
    plan = plan_splits(IDS, cohort="weaver", n_splits=5, seed=0)
    return SplitPlan(
        cohort=plan.cohort,
        n_splits=plan.n_splits,
        folds=plan.folds,
        seed=plan.seed,
        provenance={DATA_DIGEST_KEY: digest} if digest else {},
    )


def test_no_warning_when_the_plan_matches_the_cache_bytes() -> None:
    assert data_drift_warning(_plan_with_digest("a" * 64), "a" * 64) is None


def test_warns_when_the_plan_was_built_from_different_bytes() -> None:
    warning = data_drift_warning(_plan_with_digest("a" * 64), "b" * 64)
    assert warning is not None
    assert "aaaaaaaaaaaa" in warning and "bbbbbbbbbbbb" in warning


def test_warns_when_the_plan_predates_digest_recording() -> None:
    warning = data_drift_warning(_plan_with_digest(None), "b" * 64)
    assert warning is not None and "records no data digest" in warning


def test_warns_when_there_is_no_manifest_to_compare_against() -> None:
    warning = data_drift_warning(_plan_with_digest("a" * 64), None)
    assert warning is not None and "no manifest" in warning


def test_digest_survives_a_json_round_trip(tmp_path) -> None:
    plan = _plan_with_digest("c" * 64)
    assert read_plan(write_plan(plan, tmp_path / "p.json")).provenance[DATA_DIGEST_KEY] == "c" * 64

"""The Weaver-set experiment's fit setup: each arm's recordings, cv config and sampling weights."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR  # noqa: E402
from arcasleep_bidoze.training import TrainingConfig  # noqa: E402
from test_release import CACHE_NAME, EARLIER_NIGHTS, REVISED, TINY_TRAINING, _feature_root, _save  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/weaver_set_q1"))
import prepare  # noqa: E402

DRAW = {"trainSleepAccel": ["sa0", "sa1"], "trainClinical": ["cl0"], "testSleepAccel": ["sa2", "sa3"], "testClinical": ["cl1"]}


def _linked(fit: Path, prefix: str) -> set[str]:
    return {p.name.split(SUBJECT_SEPARATOR)[1][: -len(CACHE_SUFFIX)]
            for p in (fit / "cache" / CACHE_NAME).glob(f"{prefix}{SUBJECT_SEPARATOR}*")}


@pytest.mark.parametrize("arm, dataset, expected", [
    ("matched", "weaverMatched", sorted(EARLIER_NIGHTS.values())),
    ("all", "weaverAll", sorted(REVISED)),
    ("all-adult-share", "weaverAll", sorted(REVISED)),
])
def test_each_arm_trains_on_its_weaver_set_and_the_draws_adults(
    tmp_path: Path, arm: str, dataset: str, expected: list[str]
) -> None:
    root = _feature_root(tmp_path)
    training = tmp_path / "training.json"
    training.write_text(json.dumps(TINY_TRAINING))
    fit = tmp_path / "fit"

    prepare.prepare(fit, root, arm, DRAW, 3, training)

    assert _linked(fit, dataset) == set(expected)
    assert _linked(fit, "trainSleepAccel") == {"sa0", "sa1"}
    assert _linked(fit, "testClinical") == {"cl1"}
    cv = json.loads((fit / "cv.json").read_text())
    assert cv["train_sets"] == [dataset, "trainSleepAccel", "trainClinical"]
    assert cv["test_sets"] == ["testSleepAccel", "testClinical"]
    assert cv["fold_label"] == "draw3"
    config = TrainingConfig.from_json(fit / "training.json")
    if arm == "all-adult-share":
        assert config.dataset_sampling_weights == {
            "trainSleepAccel": 2.0, "trainClinical": 1.0, "weaverAll": float(len(EARLIER_NIGHTS))}
    else:
        assert config.dataset_sampling_weights is None


def test_an_unknown_arm_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prepare.prepare(tmp_path / "fit", _feature_root(tmp_path), "some", DRAW, 0)

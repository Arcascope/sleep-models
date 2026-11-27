import numpy as np
import pytest

from arcasleep.data import excluded
from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import SplitPlan, materialise

KEPT = ["child01_apple_n1", "child01_fitbit_n2", "child02_fitbit_n1"]
DROPPED = "child01_fitbit_n1"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(excluded, "EXCLUDED_RECORDINGS", frozenset({DROPPED}))
    root = tmp_path / "cache"
    root.mkdir()
    for sid in KEPT + [DROPPED]:
        np.savez(root / f"weaver_revised_subject_{sid}_features.npz", y=np.zeros((1, 2)))
    return root


def test_discovery_skips_excluded_recordings(cache) -> None:
    found = discover_cached_subjects(["weaver_revised"], cache)["weaver_revised"]
    assert sorted(s.subject_id for s in found) == sorted(KEPT)


def test_materialise_links_a_plan_without_its_excluded_recordings(cache, tmp_path) -> None:
    cached = discover_cached_subjects(["weaver_revised"], cache)
    plan = SplitPlan(
        cohort="weaver_revised",
        n_splits=2,
        folds={
            "BiDozeWeaver_revisedFold0": [DROPPED, KEPT[0]],
            "BiDozeWeaver_revisedFold1": KEPT[1:],
        },
        seed=0,
    )
    written = materialise(plan, cached, tmp_path / "links")
    assert len(written) == len(KEPT)
    assert not any(DROPPED in p.name for p in written)


def test_every_listed_recording_is_a_weaver_fitbit_recording() -> None:
    assert excluded.EXCLUDED_RECORDINGS
    assert all("_fitbit_n" in sid for sid in excluded.EXCLUDED_RECORDINGS)

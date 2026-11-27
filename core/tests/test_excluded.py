import numpy as np
import pytest

from arcasleep.data import excluded
from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import SplitPlan, fold_dataset_name, materialise

KEPT = ["child01_apple_n1", "child01_fitbit_n2", "child02_fitbit_n1"]
DROPPED = "child01_fitbit_n1"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(excluded, "EXCLUDED_RECORDINGS", {"weaver_revised": frozenset({DROPPED})})
    root = tmp_path / "cache"
    root.mkdir()
    for sid in KEPT + [DROPPED]:
        np.savez(root / f"weaver_revised_subject_{sid}_features.npz", y=np.zeros((1, 2)))
    np.savez(root / f"other_subject_{DROPPED}_features.npz", y=np.zeros((1, 2)))
    return root


def ids(found):
    return sorted(s.subject_id for s in found)


def test_discovery_skips_excluded_recordings(cache) -> None:
    found = discover_cached_subjects(["weaver_revised"], cache)
    assert ids(found["weaver_revised"]) == sorted(KEPT)


def test_discovery_keeps_the_same_id_in_another_dataset(cache) -> None:
    assert ids(discover_cached_subjects(["other"], cache)["other"]) == [DROPPED]


def test_discovery_skips_a_link_to_an_excluded_recording(cache, tmp_path) -> None:
    links = tmp_path / "links"
    links.mkdir()
    for sid in KEPT + [DROPPED]:
        (links / f"weaverAll_subject_{sid}_features.npz").symlink_to(
            cache / f"weaver_revised_subject_{sid}_features.npz"
        )
    other = cache / f"other_subject_{DROPPED}_features.npz"
    (links / f"otherAll_subject_{DROPPED}_features.npz").symlink_to(other)
    found = discover_cached_subjects(["weaverAll", "otherAll"], links)
    assert ids(found["weaverAll"]) == sorted(KEPT)
    assert ids(found["otherAll"]) == [DROPPED]


def plan(cohort):
    return SplitPlan(
        cohort=cohort,
        n_splits=2,
        folds={
            fold_dataset_name(cohort, 0): [DROPPED, KEPT[0]],
            fold_dataset_name(cohort, 1): KEPT[1:],
        },
        seed=0,
    )


def test_materialise_links_a_plan_without_its_excluded_recordings(cache, tmp_path) -> None:
    cached = discover_cached_subjects(["weaver_revised"], cache)
    written = materialise(plan("weaver_revised"), cached, tmp_path / "links")
    assert len(written) == len(KEPT)
    assert not any(DROPPED in p.name for p in written)


def test_materialise_keeps_the_same_id_in_another_cohort(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(excluded, "EXCLUDED_RECORDINGS", {"weaver_revised": frozenset({DROPPED})})
    root = tmp_path / "cache"
    root.mkdir()
    for sid in KEPT + [DROPPED]:
        np.savez(root / f"other_subject_{sid}_features.npz", y=np.zeros((1, 2)))
    cached = discover_cached_subjects(["other"], root)
    written = materialise(plan("other"), cached, tmp_path / "links")
    assert len(written) == len(KEPT) + 1


def test_every_listed_recording_is_a_weaver_fitbit_recording() -> None:
    assert list(excluded.EXCLUDED_RECORDINGS) == ["weaver_revised"]
    assert all("_fitbit_n" in sid for sid in excluded.EXCLUDED_RECORDINGS["weaver_revised"])

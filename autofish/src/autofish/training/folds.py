"""Deterministic participant-level split generation.

Splits are always taken at the subject level and recorded as an artifact, so a
run's train/test assignment is reproducible from ``split_seed`` alone.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pisces_lite.cv import Fold, SubjectRef, iter_folds

from autofish.data.cache import CachedSubject
from autofish.training.config import AutofishCVConfig


class DatasetIndex:
    """Adapts the cache listing to the ``.name``/``.ids`` shape iter_folds wants."""

    def __init__(self, name: str, subjects: Sequence[CachedSubject]) -> None:
        self.name = name
        self.ids = [subject.subject_id for subject in subjects]


def build_dataset_index(
    cached: Dict[str, List[CachedSubject]],
) -> Dict[str, DatasetIndex]:
    return {name: DatasetIndex(name, subjects) for name, subjects in cached.items()}


def subjects_from_datasets(
    datasets: Dict[str, DatasetIndex],
    set_names: Sequence[str],
) -> List[SubjectRef]:
    refs: List[SubjectRef] = []
    for name in set_names:
        refs.extend(SubjectRef(name, sid) for sid in datasets[name].ids)
    return sorted(refs)


def _overlap_rng(split_seed: int, dataset_name: str) -> np.random.Generator:
    """A per-dataset stream, so adding a dataset cannot reshuffle the others."""
    digest = hashlib.sha1(dataset_name.encode("utf-8")).hexdigest()[:8]
    return np.random.default_rng(split_seed + int(digest, 16))


def deblip_subjects_for_fold(
    datasets: Dict[str, DatasetIndex],
    deblip_sets: Sequence[str],
    fold: Fold,
) -> List[SubjectRef]:
    """Deblip subjects for one fold, never touching its test participants.

    A cohort may legitimately sit in both ``deblip_sets`` and ``test_sets`` —
    deblip supplies no labels, so it is not label leakage — but adapting on a
    recording and then scoring that same participant is still test exposure.
    The dataset-level overlap is allowed; the participant-level one is not.
    """
    held_out = set(fold.test_subjects) | set(fold.val_subjects)
    return [
        ref for ref in subjects_from_datasets(datasets, deblip_sets) if ref not in held_out
    ]


def build_folds(
    cv_config: AutofishCVConfig,
    datasets: Dict[str, DatasetIndex],
    *,
    split_seed: int,
) -> Tuple[List[Fold], dict]:
    """Build the run's folds and the split artifact describing them.

    In ``transfer`` mode a dataset may appear in both ``train_sets`` and
    ``test_sets``; the split is then taken by subject, half to each side, never
    by recording. Validation subjects are excluded from both sides first.
    """
    overlap_sets = sorted(set(cv_config.train_sets) & set(cv_config.test_sets))
    if cv_config.mode != "transfer" or not overlap_sets:
        folds = list(iter_folds(cv_config.to_pisces_config(), datasets))
        return folds, {"enabled": False, "overlap_sets": overlap_sets}

    val_subjects = subjects_from_datasets(datasets, cv_config.validation_sets)
    excluded = set(val_subjects)
    train_subjects = subjects_from_datasets(
        datasets, [n for n in cv_config.train_sets if n not in overlap_sets]
    )
    test_subjects = subjects_from_datasets(
        datasets, [n for n in cv_config.test_sets if n not in overlap_sets]
    )

    summary: dict = {
        "enabled": True,
        "seed": split_seed,
        "test_fraction": 0.5,
        "overlap_sets": overlap_sets,
        "splits": {},
    }
    for name in overlap_sets:
        candidates = [
            ref
            for sid in datasets[name].ids
            if (ref := SubjectRef(name, sid)) not in excluded
        ]
        if len(candidates) < 2:
            raise ValueError(
                f"cv.json: dataset {name!r} is in both train_sets and test_sets but has "
                f"{len(candidates)} non-validation subjects; at least 2 are needed to split"
            )
        rng = _overlap_rng(split_seed, name)
        shuffled = [candidates[i] for i in rng.permutation(len(candidates))]
        test_count = max(1, len(shuffled) // 2)
        split_test = sorted(shuffled[:test_count])
        split_train = sorted(shuffled[test_count:])
        train_subjects.extend(split_train)
        test_subjects.extend(split_test)
        summary["splits"][name] = {
            "train_subjects": [ref.subject_id for ref in split_train],
            "test_subjects": [ref.subject_id for ref in split_test],
        }

    train_subjects = sorted(ref for ref in train_subjects if ref not in excluded)
    test_subjects = sorted(test_subjects)
    leaked = set(train_subjects) & set(test_subjects)
    if leaked:
        raise RuntimeError(f"train/test subject overlap after split: {sorted(leaked)}")

    fold = Fold(
        fold_id=0,
        total_folds=1,
        mode="transfer",
        description=(
            f"transfer: 1 model on {len(train_subjects)} train subjects from "
            f"{list(cv_config.train_sets)}, participant split of {overlap_sets} -> "
            f"{len(test_subjects)} test subjects from {list(cv_config.test_sets)} "
            f"(val: {len(val_subjects)}, split_seed: {split_seed})"
        ),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    return [fold], summary

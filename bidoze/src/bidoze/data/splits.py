"""Materialised cross-validation splits.

The split is an *artifact*, not a computation. A plan file lists, per fold, the
literal subject ids that belong to it; it is generated once, committed, and
from then on read rather than recomputed. A seeded shuffle is reproducible only
while the seed, the RNG, the cohort membership and the sort order all hold
still -- four things that drift independently. Writing the ids down removes all
four from the reproducibility argument.

Folds are materialised as symlinks whose *filenames* carry a per-fold dataset
name, because :func:`bidoze.data.cache.discover_cached_subjects` takes the
dataset from the filename prefix. A fold is therefore an ordinary dataset as
far as the rest of the pipeline is concerned, which means k-fold needs no new
CV mode: for any fold, ``train_sets`` and ``test_sets`` are disjoint, and that
is exactly what ``transfer`` already expresses.

The links live in their own root rather than beside the real cache, so the
content-addressed feature cache is never written to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from pisces_lite.model_io import get_features_cache_name

from bidoze.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR, CachedSubject

SPLIT_SCHEMA_VERSION = 1
FINAL_FOLD_NAME = "final"


def fold_dataset_name(cohort: str, index: int) -> str:
    """``weaver`` + 0 -> ``BiDozeWeaverFold0``: a dataset name, not a path."""
    return f"BiDoze{cohort[:1].upper()}{cohort[1:]}Fold{index}"


@dataclass(frozen=True)
class SplitPlan:
    """Which subjects are in which fold, plus how the assignment was made."""

    cohort: str
    n_splits: int
    folds: Dict[str, List[str]]
    seed: int
    provenance: dict = field(default_factory=dict)

    @property
    def fold_names(self) -> List[str]:
        return [fold_dataset_name(self.cohort, i) for i in range(self.n_splits)]

    def subjects(self) -> List[str]:
        return sorted(sid for ids in self.folds.values() for sid in ids)

    def validate(self) -> None:
        expected = self.fold_names
        if sorted(self.folds) != sorted(expected):
            raise ValueError(
                f"split plan: folds {sorted(self.folds)} do not match "
                f"n_splits={self.n_splits} for cohort {self.cohort!r}"
            )
        seen: Dict[str, str] = {}
        for name in expected:
            if not self.folds[name]:
                raise ValueError(f"split plan: fold {name} is empty")
            for sid in self.folds[name]:
                if sid in seen:
                    raise ValueError(f"split plan: subject {sid} is in both {seen[sid]} and {name}")
                seen[sid] = name

    def to_dict(self) -> dict:
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "cohort": self.cohort,
            "n_splits": self.n_splits,
            "seed": self.seed,
            "provenance": self.provenance,
            # Sorted so the committed file is stable under regeneration.
            "folds": {name: sorted(self.folds[name]) for name in self.fold_names},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SplitPlan":
        version = data.get("schema_version")
        if version != SPLIT_SCHEMA_VERSION:
            raise ValueError(
                f"split plan: schema_version {version!r}, expected {SPLIT_SCHEMA_VERSION}"
            )
        plan = cls(
            cohort=data["cohort"],
            n_splits=int(data["n_splits"]),
            folds={k: list(v) for k, v in data["folds"].items()},
            seed=int(data.get("seed", 0)),
            provenance=dict(data.get("provenance", {})),
        )
        plan.validate()
        return plan


def plan_splits(
    subject_ids: Sequence[str],
    *,
    cohort: str,
    n_splits: int,
    seed: int = 0,
    provenance: dict | None = None,
) -> SplitPlan:
    """Assign subjects to ``n_splits`` near-equal folds, once.

    The shuffle is seeded and taken over *sorted* ids so that regenerating the
    plan from the same cohort reproduces it. That reproducibility is a
    convenience for review, not the guarantee -- the guarantee is the committed
    file.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    ordered = sorted(set(subject_ids))
    if len(ordered) < n_splits:
        raise ValueError(
            f"cohort {cohort!r} has {len(ordered)} subjects; cannot make {n_splits} folds"
        )
    shuffled = [ordered[i] for i in np.random.default_rng(seed).permutation(len(ordered))]
    chunks = np.array_split(np.arange(len(shuffled)), n_splits)
    plan = SplitPlan(
        cohort=cohort,
        n_splits=n_splits,
        folds={
            fold_dataset_name(cohort, i): sorted(shuffled[j] for j in chunk)
            for i, chunk in enumerate(chunks)
        },
        seed=seed,
        provenance=dict(provenance or {}),
    )
    plan.validate()
    return plan


DATA_DIGEST_KEY = "data_digest"


def data_drift_warning(plan: SplitPlan, current_digest: str | None) -> str | None:
    """Why this plan may not belong to the cache in front of us, or None.

    Membership is checked separately and fails hard. This is the softer signal:
    the plan can name exactly the right subjects while having been built from
    different bytes -- a cache rebuilt from repaired source files, say. That is
    usually fine and occasionally not, so it is reported rather than enforced.
    """
    recorded = plan.provenance.get(DATA_DIGEST_KEY)
    if not recorded:
        return (
            "this plan records no data digest, so it cannot be checked against "
            "the cache; regenerate it with splits-plan to record one"
        )
    if not current_digest:
        return (
            f"plan was built against data {recorded[:12]}, but no manifest is "
            "available here to compare against"
        )
    if recorded != current_digest:
        return (
            f"plan was built against data {recorded[:12]}, cache manifest is "
            f"{current_digest[:12]} -- different bytes, same subject list is "
            "still possible and is checked separately"
        )
    return None


def write_plan(plan: SplitPlan, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=False) + "\n")
    return path


def read_plan(path: Path | str) -> SplitPlan:
    return SplitPlan.from_dict(json.loads(Path(path).read_text()))


def materialise(
    plan: SplitPlan,
    cached: Dict[str, List[CachedSubject]],
    link_dir: Path | str,
    *,
    passthrough: Sequence[str] = (),
) -> List[Path]:
    """Build a symlink-only cache in which each fold is its own dataset.

    ``passthrough`` datasets (an external test set, say) are linked under their
    own names so one directory holds everything a run needs.
    """
    link_dir = Path(link_dir)
    link_dir.mkdir(parents=True, exist_ok=True)

    by_id = {subject.subject_id: subject for subject in cached.get(plan.cohort, [])}
    missing = [sid for sid in plan.subjects() if sid not in by_id]
    if missing:
        raise FileNotFoundError(
            f"split plan names {len(missing)} subject(s) absent from the cache: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    written: List[Path] = []
    for fold_name in plan.fold_names:
        for sid in plan.folds[fold_name]:
            written.append(
                _link(by_id[sid].path, link_dir / get_features_cache_name(fold_name, sid))
            )
    for name in passthrough:
        if not cached.get(name):
            raise FileNotFoundError(f"no cached subjects for dataset {name!r}")
        for subject in cached[name]:
            written.append(_link(subject.path, link_dir / subject.path.name))
    return written


def _link(source: Path, destination: Path) -> Path:
    source = source.resolve()
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source)
    return destination


def extra_deblip_sets(template: dict, cohort: str) -> List[str]:
    """Deblip datasets in the cv template other than the cohort being split."""
    return [name for name in template.get("deblip_sets", []) if name != cohort]


def cv_configs_for_plan(
    plan: SplitPlan,
    template: dict,
    *,
    external_test_sets: Sequence[str],
) -> Dict[str, dict]:
    """One cv config per fold, plus ``final``.

    Per fold: train on the other folds, test on the held-out one. The final
    model trains on every fold and is scored only on the external sets -- it
    has no held-out cohort subjects by construction, so scoring it on them
    would report training performance.
    """
    names = plan.fold_names
    extra_deblip = extra_deblip_sets(template, plan.cohort)
    configs: Dict[str, dict] = {}
    for index, held_out in enumerate(names):
        train = [n for n in names if n != held_out]
        config = dict(template)
        config.pop("n_splits", None)
        config["train_sets"] = train
        config["test_sets"] = [held_out]
        # Deblip is unlabelled adaptation, but adapting on a participant and
        # then scoring them is still exposure; restricting it to the training
        # folds keeps that impossible rather than merely filtered. Deblip-only
        # datasets outside the cohort hold no scored subjects and ride along.
        config["deblip_sets"] = train + extra_deblip
        config["fold_label"] = f"fold{index}"
        configs[f"fold{index}"] = config

    final = dict(template)
    final.pop("n_splits", None)
    final["train_sets"] = list(names)
    final["test_sets"] = list(external_test_sets)
    final["deblip_sets"] = list(names) + extra_deblip
    final["fold_label"] = FINAL_FOLD_NAME
    configs[FINAL_FOLD_NAME] = final
    return configs


def write_cv_configs(configs: Dict[str, dict], out_dir: Path | str) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, config in configs.items():
        path = out_dir / f"cv_{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        written.append(path)
    return written

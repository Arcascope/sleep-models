"""Write ``config/splits/weaver_revised_pediatric_5fold.json``: the revised-Weaver folds that keep
the earlier ``weaver`` plan's children where they were.

- A child is the part of a recording id before the first ``_``; a child's recordings (devices and
  nights) are in one fold.
- Each earlier ``weaver`` night is found in revised Weaver by its PSG labels
  (``arcasleep_bidoze.release.holding_recordings``). That child keeps the fold the night has in
  ``config/splits/weaver_5fold.json``. A child whose earlier nights are in different folds keeps
  the fold of the first of them (by recording id) and is listed in ``split_earlier_children`` of
  the plan's provenance: the earlier model of the other fold trained on one of its nights.
- The other children are shuffled with the seed, then each goes to the fold holding the fewest
  recordings so far (counting the kept children's recordings).

Needs only PSG labels: the caches may be labels-only.

Usage: python plan.py <cache root with weaver> <cache root with weaver_revised> [--seed 0] [--output PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from arcasleep.data.cache import discover_cached_subjects, participant_id
from arcasleep.data.splits import SplitPlan, fold_dataset_name, read_plan, write_plan
from arcasleep_bidoze.release import CACHE_NAME, holding_recordings

BIDOZE = Path(__file__).resolve().parents[2]
EARLIER_PLAN = "config/splits/weaver_5fold.json"
OUTPUT = "config/splits/weaver_revised_pediatric_5fold.json"
COHORT = "weaver_revised"


def extend_plan(
    recordings: list[str], pinned: dict[str, int], n_splits: int, seed: int
) -> dict[int, list[str]]:
    """Fold index -> recordings; a child in ``pinned`` goes to its fold, the others are balanced."""
    children: dict[str, list[str]] = {}
    for sid in sorted(recordings):
        children.setdefault(participant_id(sid), []).append(sid)
    unknown = sorted(set(pinned) - set(children))
    if unknown:
        raise ValueError(f"pinned children without recordings: {unknown[:5]}")
    folds: dict[int, list[str]] = {i: [] for i in range(n_splits)}
    for child, fold in pinned.items():
        folds[fold].extend(children[child])
    free = sorted(set(children) - set(pinned))
    for i in np.random.default_rng(seed).permutation(len(free)):
        min(folds.values(), key=len).extend(children[free[i]])
    return {fold: sorted(sids) for fold, sids in folds.items()}


def earlier_children(
    plan: SplitPlan, earlier: dict[str, Path], revised: dict[str, Path]
) -> tuple[dict[str, int], list[str]]:
    """The fold of each child whose night is in the earlier plan, and the children whose earlier
    nights are in more than one fold (these keep the fold of their first night)."""
    labels = {sid: np.load(path)["y"].ravel() for sid, path in revised.items()}
    fold_of = {sid: i for i, name in enumerate(plan.fold_names) for sid in plan.folds[name]}
    pinned: dict[str, int] = {}
    split: set[str] = set()
    for sid, path in sorted(earlier.items()):
        child = participant_id(holding_recordings(np.load(path)["y"].ravel(), labels)[0])
        if pinned.setdefault(child, fold_of[sid]) != fold_of[sid]:
            split.add(child)
    return pinned, sorted(split)


def main(earlier_root: Path, revised_root: Path, seed: int, output: Path) -> None:
    earlier = {s.subject_id: s.path
               for s in discover_cached_subjects(["weaver"], earlier_root / CACHE_NAME)["weaver"]}
    revised = {s.subject_id: s.path
               for s in discover_cached_subjects([COHORT], revised_root / CACHE_NAME)[COHORT]}
    earlier_plan = read_plan(BIDOZE / EARLIER_PLAN)
    pinned, split = earlier_children(earlier_plan, earlier, revised)
    folds = extend_plan(list(revised), pinned, earlier_plan.n_splits, seed)
    plan = SplitPlan(
        cohort=COHORT,
        n_splits=earlier_plan.n_splits,
        folds={fold_dataset_name(COHORT, i): sids for i, sids in folds.items()},
        seed=seed,
        provenance={
            "cache_dir": earlier_plan.provenance["cache_dir"],
            "external_test_sets": earlier_plan.provenance["external_test_sets"],
            "subject_count": len(revised),
            "group_separator": "_",
            "data_digest": None,
            "kept_from": EARLIER_PLAN,
            "kept_children": len(pinned),
            "split_earlier_children": split,
            "assignment": "kept children stay in their earlier fold; the others, shuffled with the "
                          "seed, each go to the fold with the fewest recordings",
        },
    )
    plan.validate()
    write_plan(plan, output)
    for i, name in enumerate(plan.fold_names):
        children = {participant_id(s) for s in plan.folds[name]}
        print(f"fold{i}: {len(plan.folds[name])} recordings, {len(children)} children, "
              f"{len(children & set(pinned))} earlier children")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("earlier_root", type=Path)
    parser.add_argument("revised_root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=BIDOZE / OUTPUT)
    args = parser.parse_args()
    main(args.earlier_root.resolve(), args.revised_root.resolve(), args.seed, args.output)

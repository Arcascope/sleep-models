"""Write the setup of the free-living experiment: where a night sits in the context, and Foundry.

The adult release puts every training recording at epoch 0 of its 3,000-epoch context, so the
context past about epoch 1,100 only ever held padding; on 24 h recordings it then calls late
nights gap or wake (research log 2026-10-15 §2). Every arm trains the adult stage source's
recipe on the release's training nights: the matched revised Weaver nights (``matched.json``)
plus all of SleepAccel.

Arms:
- random:        the release's nights, each placed at a random epoch of the context
                 (``segment_placement: random``).
- foundry:       as random, plus the FoundryApple windows (Apple Watch sleep stages over
                 09:00–09:00) of the people the phase does not hold out.
- foundry-start: as foundry, with every recording at epoch 0, as the release trained.
- circular:      as random, with each row rotated by a uniformly drawn number of epochs
                 (``segment_placement: circular``), so a recording can wrap from the
                 context's end to its start.
- foundry-circular: as foundry, with circular placement.

Phases are ``<arm>/<held-out person>`` (eric, olivia, doug), training without that person's
windows, and ``<arm>/final``, training on every person. ``random`` and ``circular`` have only
``final``: they never train on Foundry. Every phase scores SleepAccel-Clinical and SENSE,
which no arm trains on; a person phase also scores that person's windows, and the ``final``
phase of an arm without Foundry scores all of them.

Writes, under ``<run>/setup/``:
- cache/<arm>/12Hz_log_PSD_aligned/  symlinks only: the matched nights as ``weaverMatched``,
                                      SleepAccel by fold, each person's FoundryApple windows as
                                      ``Foundry<Person>`` with subject id ``<Person>_<window>``
                                      (so the early-stopping holdout keeps a person on one
                                      side), SleepAccel-Clinical, SENSE
- cv/<arm>/cv_<phase>.json
- training-<arm>.json                 the repo's balanced recipe, run_seed 0, seed_count 3, and
                                      the arm's segment_placement

Usage: python prepare.py <run dir> <cache root>
(the cache root holds ``12Hz_log_PSD_aligned/`` with weaver_revised, SleepAccel,
SleepAccel-Clinical, FoundryApple and SENSE).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import read_plan

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
SEEDS = 3
WEAVER = "weaverMatched"
OUTSIDE = ["SleepAccel-Clinical", "SENSE"]
#: arm → (segment_placement, trains on Foundry)
ARMS = {
    "random": ("random", False),
    "foundry": ("random", True),
    "foundry-start": ("start", True),
    "circular": ("circular", False),
    "foundry-circular": ("circular", True),
}


def person_of(window: str) -> str:
    """``Eric-AppleWatch-2026-10-11-09.00--2026-10-12`` → ``Eric``."""
    return window.split("-", 1)[0]


def foundry_dataset(person: str) -> str:
    return f"Foundry{person}"


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(run_dir: Path, cache_root: Path) -> None:
    setup = run_dir / "setup"
    cached = discover_cached_subjects(
        ["weaver_revised", "SleepAccel", "FoundryApple", *OUTSIDE], cache_root / CACHE_NAME
    )
    weaver = {s.subject_id: s.path for s in cached["weaver_revised"]}
    sleepaccel = {s.subject_id: s.path for s in cached["SleepAccel"]}
    matched = json.loads((HERE / "matched.json").read_text())["weaver_revised"]
    missing = sorted(set(matched) - set(weaver))
    if missing:
        raise ValueError(f"{len(missing)} matched nights are not in the cache, e.g. {missing[0]}")
    plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")
    windows: dict[str, list] = defaultdict(list)
    for subject in cached["FoundryApple"]:
        windows[person_of(subject.subject_id)].append(subject)
    people = sorted(windows)

    recipe = json.loads((BIDOZE / "config/training-balanced.json").read_text())
    for arm, (placement, with_foundry) in ARMS.items():
        write_json(setup / f"training-{arm}.json",
                   {**recipe, "run_seed": 0, "seed_count": SEEDS, "segment_placement": placement})

        cache = setup / "cache" / arm / CACHE_NAME
        for sid in matched:
            link(weaver[sid], cache / f"{WEAVER}_subject_{sid}_features.npz")
        for fold, sids in plan.folds.items():
            for sid in sids:
                link(sleepaccel[sid], cache / f"{fold}_subject_{sid}_features.npz")
        for person in people:
            for subject in windows[person]:
                name = f"{foundry_dataset(person)}_subject_{person}_{subject.subject_id}_features.npz"
                link(subject.path, cache / name)
        for dataset in OUTSIDE:
            for subject in cached[dataset]:
                link(subject.path, cache / subject.path.name)

        base = [WEAVER, *plan.fold_names]
        phases = {"final": ([*base, *map(foundry_dataset, people)], OUTSIDE)}
        if not with_foundry:
            phases["final"] = (base, [*map(foundry_dataset, people), *OUTSIDE])
        else:
            for held in people:
                others = [foundry_dataset(p) for p in people if p != held]
                phases[held.lower()] = ([*base, *others], [foundry_dataset(held), *OUTSIDE])
        for phase, (train_sets, test_sets) in phases.items():
            write_json(setup / "cv" / arm / f"cv_{phase}.json", {
                "mode": "transfer",
                "train_sets": train_sets,
                "test_sets": test_sets,
                "validation_sets": [],
                "fold_label": phase,
            })
        print(f"{arm}: {placement}, phases {sorted(phases)}")
    counts = ", ".join(f"{p} {len(windows[p])}" for p in people)
    print(f"{len(matched)} matched nights; FoundryApple windows: {counts}")


if __name__ == "__main__":
    main(*(Path(a).resolve() for a in sys.argv[1:3]))

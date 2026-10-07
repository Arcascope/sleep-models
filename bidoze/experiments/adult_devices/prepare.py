"""Write the setup of the revised-Weaver device ablation under the adult recipe.

Every arm trains the stage source of the adult formulation (see README.md): revised
Weaver recordings from the arm's devices plus 4 of the 5 SleepAccel folds, scored on the
held-out SleepAccel fold and on all of SleepAccel-Clinical.

Arms (revised Weaver part of the training set):
- all:           every recording of every device (634 recordings, 237 children)
- actigraph:     every ActiGraph recording (238 recordings, 236 children)
- <device>-119:  119 children drawn per device (seed 0), all of their recordings with that
                 device. 119 is the number of children with a Fitbit recording, so the four
                 device arms differ in device and children, not in training-set size.

Writes, under ``<run>/setup/``:
- cache/<arm>/12Hz_log_PSD_aligned/  symlinks only: the arm's Weaver recordings as dataset
                                      ``weaver<Arm>``, SleepAccel by fold, SleepAccel-Clinical
- cv/<arm>/cv_fold<k>.json            train on weaver<Arm> + the other 4 SleepAccel folds
- arms.json                           the subject ids of every arm
- training.json                       the repo's balanced recipe with run_seed 0, seed_count 3

Usage: python prepare.py <run dir> <weaver_revised cache root> <SleepAccel cache root>
(cache roots are the directories that hold ``12Hz_log_PSD_aligned/``).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from arcasleep.data.cache import discover_cached_subjects, participant_id
from arcasleep.data.splits import read_plan

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
MATCHED_CHILDREN = 119
SEEDS = 3
DEVICES = ["actigraph", "apple", "fitbit", "garmin"]


def device_of(subject_id: str) -> str:
    return subject_id.split("_")[1]


def arms(weaver_ids: list[str]) -> dict[str, list[str]]:
    by_device: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid in weaver_ids:
        by_device[device_of(sid)][participant_id(sid)].append(sid)
    out = {"all": sorted(weaver_ids), "actigraph": sorted(s for c in by_device["actigraph"].values() for s in c)}
    for device in DEVICES:
        children = sorted(by_device[device])
        drawn = np.random.default_rng(0).choice(len(children), MATCHED_CHILDREN, replace=False)
        out[f"{device}-{MATCHED_CHILDREN}"] = sorted(s for i in drawn for s in by_device[device][children[i]])
    return out


def dataset_name(arm: str) -> str:
    return "weaver" + "".join(part.capitalize() for part in arm.split("-"))


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def main(run_dir: Path, weaver_root: Path, adult_root: Path) -> None:
    setup = run_dir / "setup"
    weaver = {s.subject_id: s.path
              for s in discover_cached_subjects(["weaver_revised"], weaver_root / CACHE_NAME)["weaver_revised"]}
    adult = discover_cached_subjects(["SleepAccel", "SleepAccel-Clinical"], adult_root / CACHE_NAME)
    sleepaccel = {s.subject_id: s.path for s in adult["SleepAccel"]}
    clinical = adult["SleepAccel-Clinical"]
    plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")

    chosen = arms(sorted(weaver))
    setup.mkdir(parents=True, exist_ok=True)
    (setup / "arms.json").write_text(json.dumps(chosen, indent=1) + "\n")

    training = json.loads((BIDOZE / "config/training-balanced.json").read_text())
    training.update(run_seed=0, seed_count=SEEDS)
    (setup / "training.json").write_text(json.dumps(training, indent=2) + "\n")

    for arm, ids in chosen.items():
        name = dataset_name(arm)
        cache = setup / "cache" / arm / CACHE_NAME
        for sid in ids:
            link(weaver[sid], cache / f"{name}_subject_{sid}_features.npz")
        for fold, sids in plan.folds.items():
            for sid in sids:
                link(sleepaccel[sid], cache / f"{fold}_subject_{sid}_features.npz")
        for subject in clinical:
            link(subject.path, cache / subject.path.name)
        for k, held_out in enumerate(plan.fold_names):
            cv = {
                "mode": "transfer",
                "train_sets": [name] + [f for f in plan.fold_names if f != held_out],
                "test_sets": [held_out, "SleepAccel-Clinical"],
                "validation_sets": [],
                "fold_label": f"fold{k}",
            }
            path = setup / "cv" / arm / f"cv_fold{k}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cv, indent=2) + "\n")
        children = len({participant_id(s) for s in ids})
        print(f"{arm}: {name}, {len(ids)} recordings, {children} children")


if __name__ == "__main__":
    main(*(Path(a).resolve() for a in sys.argv[1:4]))

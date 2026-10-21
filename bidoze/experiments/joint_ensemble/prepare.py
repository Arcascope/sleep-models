"""Write the setup of the joint-ensemble experiment: random half splits of the adult cohorts.

Each draw d (random generator seed d) trains on
- 101 of the 201 children of the release's revised-Weaver nights (``matched``, 202 recordings),
  all of each drawn child's recordings, as dataset ``weaverDraw``;
- 30 of the 60 adults of SleepAccel (28) and SleepAccel-Clinical (32), drawn from the pooled 60
  without replacement, as ``trainSleepAccel`` and ``trainClinical``;
and is scored on the other 30, as ``testSleepAccel`` and ``testClinical``.

Arms (training recipe: bidoze/config/training-balanced.json):
- independent: seeds 0, 1 and 2 trained separately, as the release is.
- joint3: one fit of 3 networks whose mean output takes the loss (``joint_members`` 3); member
  m starts from the weights of independent seed m.
- joint2: as joint3 with 2 networks.

Writes, under ``<run>/setup/``:
- draws.json                          the training and test ids of every draw
- cache/draw<d>/12Hz_log_PSD_aligned/ symlinks into the feature cache, named by the datasets above
- cv/draw<d>.json                     train on the three training sets, test on the two test sets
- training-<arm>.json                 the arm's recipe

Usage: python prepare.py <run dir> <feature cache root> <matched ids json> [draw count]
(the cache root holds ``12Hz_log_PSD_aligned/`` with SleepAccel, SleepAccel-Clinical and
weaver_revised; the ids json is a run's ``setup/arms.json`` whose ``matched`` list names the
release's revised-Weaver recordings).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from arcasleep.data.cache import discover_cached_subjects, participant_id

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
WEAVER_CHILDREN = 101
ADULT_TRAIN = 30
DRAWS = 5
ARMS = {"independent": {"seed_count": 3}, "joint3": {"joint_members": 3}, "joint2": {"joint_members": 2}}
TRAIN_SETS = {"weaver": "weaverDraw", "SleepAccel": "trainSleepAccel", "SleepAccel-Clinical": "trainClinical"}
TEST_SETS = {"SleepAccel": "testSleepAccel", "SleepAccel-Clinical": "testClinical"}


def draw(seed: int, matched: list[str], adults: list[tuple[str, str]]) -> dict[str, list[str]]:
    """The ids of one draw, by dataset name."""
    rng = np.random.default_rng(seed)
    children: dict[str, list[str]] = defaultdict(list)
    for sid in matched:
        children[participant_id(sid)].append(sid)
    names = sorted(children)
    drawn = rng.choice(len(names), WEAVER_CHILDREN, replace=False)
    train = set(rng.choice(len(adults), ADULT_TRAIN, replace=False).tolist())
    out = {name: [] for name in [*TRAIN_SETS.values(), *TEST_SETS.values()]}
    out[TRAIN_SETS["weaver"]] = sorted(s for i in drawn for s in children[names[i]])
    for i, (cohort, sid) in enumerate(adults):
        out[(TRAIN_SETS if i in train else TEST_SETS)[cohort]].append(sid)
    return {name: sorted(ids) for name, ids in out.items()}


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def main(run_dir: Path, cache_root: Path, matched_json: Path, draws: int = DRAWS) -> None:
    setup = run_dir / "setup"
    cached = discover_cached_subjects(["SleepAccel", "SleepAccel-Clinical", "weaver_revised"], cache_root / CACHE_NAME)
    paths = {(cohort, s.subject_id): s.path for cohort, subjects in cached.items() for s in subjects}
    matched = json.loads(matched_json.read_text())["matched"]
    missing = [sid for sid in matched if ("weaver_revised", sid) not in paths]
    if missing:
        raise ValueError(f"{len(missing)} matched recordings are not in the cache, e.g. {missing[:3]}")
    adults = sorted(key for key in paths if key[0] != "weaver_revised")
    print(f"{len(matched)} matched Weaver recordings; adults: "
          + ", ".join(f"{c} {sum(k[0] == c for k in adults)}" for c in TEST_SETS))

    source_of = {name: cohort for cohort, name in [*TRAIN_SETS.items(), *TEST_SETS.items()]}
    source_of[TRAIN_SETS["weaver"]] = "weaver_revised"
    plans = {}
    for d in range(draws):
        ids = draw(d, matched, adults)
        plans[f"draw{d}"] = ids
        cache = setup / "cache" / f"draw{d}" / CACHE_NAME
        for name, sids in ids.items():
            for sid in sids:
                link(paths[source_of[name], sid], cache / f"{name}_subject_{sid}_features.npz")
        cv = {
            "mode": "transfer",
            "train_sets": list(TRAIN_SETS.values()),
            "test_sets": list(TEST_SETS.values()),
            "validation_sets": [],
            "fold_label": f"draw{d}",
        }
        (setup / "cv").mkdir(parents=True, exist_ok=True)
        (setup / "cv" / f"draw{d}.json").write_text(json.dumps(cv, indent=2) + "\n")
        print(f"draw{d}: " + ", ".join(f"{name} {len(sids)}" for name, sids in ids.items()))
    (setup / "draws.json").write_text(json.dumps(plans, indent=1) + "\n")

    recipe = json.loads((BIDOZE / "config/training-balanced.json").read_text())
    for arm, changes in ARMS.items():
        (setup / f"training-{arm}.json").write_text(json.dumps({**recipe, "run_seed": 0, **changes}, indent=2) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0]).resolve(), Path(args[1]).resolve(), Path(args[2]).resolve(), *(int(a) for a in args[3:4]))

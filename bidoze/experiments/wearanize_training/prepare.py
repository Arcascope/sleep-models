"""Write the setup of one fit of the Wearanize+ training experiment: an arm on a draw.

Every fit trains on all 634 revised-Weaver recordings plus the draw's training adults (SleepAccel and
SleepAccel-Clinical, ``draws.json``, the draws of the Weaver-set experiment Q1). The arms differ in
the architecture and in the Wearanize+ nights:

- g-base, f2-base:           no Wearanize+ nights in training
- g-wearanize, f2-wearanize: also the Wearanize+ nights of the 4 folds other than the draw's
                             (``config/splits/wearanize_5fold.json``)

``g-*`` is BiDoze-G (``config/model-gap-rule.json``), ``f2-*`` the strand-fusion network
(``config/model-f2.json``). Draw d is scored on the draw's test adults, Wearanize+ fold d (held out
in every arm) and all of AAUWSS. Further held-out cohorts are listed in ``EXTRA_TEST_COHORTS``.

Writes, under ``<fit dir>/``:
- cache/12Hz_log_PSD_aligned/  symlinks only
- cv.json                      train on weaverAll + trainSleepAccel + trainClinical (+ trainWearanize);
                               test on testSleepAccel, testClinical, testWearanize, testAAUWSS
- training.json                ``config/training-adult.json`` unless one is given
- recordings.json              the Wearanize+ nights trained on

Usage: python prepare.py <fit dir> <cache root> <arm> <draw> [<training config>]
(``cache root`` holds ``12Hz_log_PSD_aligned/`` with weaver_revised, SleepAccel, SleepAccel-Clinical,
wearanize and aauwss.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import fold_dataset_name, read_plan
from arcasleep_bidoze.release import link, weaver_dataset_name

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
ARMS = ("g-base", "g-wearanize", "f2-base", "f2-wearanize")
MODEL_CONFIGS = {"g": "config/model-gap-rule.json", "f2": "config/model-f2.json"}
PLAN = BIDOZE / "config/splits/wearanize_5fold.json"
ADULT_SETS = {"trainSleepAccel": "SleepAccel", "trainClinical": "SleepAccel-Clinical",
              "testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical"}
#: Held-out cohorts scored whole: test set name -> cache dataset name. EESM19 joins here once built.
EXTRA_TEST_COHORTS = {"testAAUWSS": "aauwss"}


def architecture(arm: str) -> str:
    return arm.split("-", 1)[0]


def with_wearanize(arm: str) -> bool:
    return arm.endswith("-wearanize")


def model_config(arm: str) -> str:
    return MODEL_CONFIGS[architecture(arm)]


def prepare(fit_dir: Path, cache_root: Path, arm: str, draw: dict, draw_index: int,
            training_config: Path | None = None, plan_path: Path = PLAN) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    cache = cache_root / CACHE_NAME
    found = discover_cached_subjects(
        ["weaver_revised", "SleepAccel", "SleepAccel-Clinical", "wearanize", *EXTRA_TEST_COHORTS.values()], cache)
    paths = {cohort: {s.subject_id: s.path for s in subjects} for cohort, subjects in found.items()}

    plan = read_plan(plan_path)
    held_out = plan.folds[fold_dataset_name(plan.cohort, draw_index)]
    trained = sorted(sid for name, ids in plan.folds.items()
                     if name != fold_dataset_name(plan.cohort, draw_index) for sid in ids)

    weaver = weaver_dataset_name("all")
    links = fit_dir / "cache" / CACHE_NAME
    for sid in sorted(paths["weaver_revised"]):
        link(paths["weaver_revised"][sid], links / f"{weaver}_subject_{sid}_features.npz")
    for dataset, cohort in ADULT_SETS.items():
        for sid in draw[dataset]:
            link(paths[cohort][sid], links / f"{dataset}_subject_{sid}_features.npz")
    for sid in held_out:
        link(paths["wearanize"][sid], links / f"testWearanize_subject_{sid}_features.npz")
    for dataset, cohort in EXTRA_TEST_COHORTS.items():
        for sid, path in paths[cohort].items():
            link(path, links / f"{dataset}_subject_{sid}_features.npz")
    recordings = trained if with_wearanize(arm) else []
    for sid in recordings:
        link(paths["wearanize"][sid], links / f"trainWearanize_subject_{sid}_features.npz")

    train_sets = [weaver, "trainSleepAccel", "trainClinical"] + (["trainWearanize"] if recordings else [])
    cv = {"mode": "transfer", "train_sets": train_sets,
          "test_sets": ["testSleepAccel", "testClinical", "testWearanize", *EXTRA_TEST_COHORTS],
          "validation_sets": [], "fold_label": f"draw{draw_index}"}
    training = json.loads((training_config or BIDOZE / "config/training-adult.json").read_text())
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "cv.json").write_text(json.dumps(cv, indent=2) + "\n")
    (fit_dir / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    (fit_dir / "recordings.json").write_text(json.dumps(recordings, indent=1) + "\n")
    print(f"{arm} draw{draw_index}: {len(paths['weaver_revised'])} Weaver recordings, "
          f"{len(draw['trainSleepAccel']) + len(draw['trainClinical'])} training adults, "
          f"{len(recordings)} Wearanize+ nights; tests: "
          f"{len(draw['testSleepAccel']) + len(draw['testClinical'])} adults, "
          f"{len(held_out)} Wearanize+, {len(paths['aauwss'])} AAUWSS")


def main(fit_dir: Path, cache_root: Path, arm: str, draw_index: int, training_config: Path | None = None) -> None:
    draw = json.loads((HERE / "draws.json").read_text())[f"draw{draw_index}"]
    prepare(fit_dir, cache_root, arm, draw, draw_index, training_config)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3], int(sys.argv[4]),
         Path(sys.argv[5]).resolve() if len(sys.argv) > 5 else None)

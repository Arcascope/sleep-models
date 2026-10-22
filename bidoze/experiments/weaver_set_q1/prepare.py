"""Write the setup of one fit of the Weaver-set experiment: an arm on a draw.

Every fit trains on a revised-Weaver recording set plus the draw's training adults (SleepAccel and
SleepAccel-Clinical recordings, ``draws.json``) and is scored on the draw's other adults. The draws are
those of the October 16 TriDoze test; the adult recordings are the same in every arm.

Arms (revised-Weaver part of the training set):
- matched:          the 202 recordings of the earlier ``weaver`` cache's nights (``release.match_weaver``);
                    dataset ``weaverMatched``
- all:              all 634 revised-Weaver recordings; dataset ``weaverAll``
- all-adult-share:  as ``all``, with ``dataset_sampling_weights`` set so that the adult recordings are
                    drawn as often as in ``matched``: a dataset's weight is its number of recordings in
                    ``matched`` (202 for the Weaver set, the draw's counts for the adult sets)

Writes, under ``<fit dir>/``:
- cache/12Hz_log_PSD_aligned/  symlinks only: the arm's Weaver recordings, trainSleepAccel,
                               trainClinical, testSleepAccel and testClinical
- cv.json                      train on the Weaver set + trainSleepAccel + trainClinical; test on the other two
- training.json                the training config (``config/training-adult.json`` unless one is given)
                               with the arm's sampling weights, if any
- recordings.json              the Weaver recordings trained on

Usage: python prepare.py <fit dir> <cache root> <arm> <draw> [<training config>]
(``cache root`` holds ``12Hz_log_PSD_aligned/`` with weaver_revised, SleepAccel and SleepAccel-Clinical;
for ``matched`` also the earlier ``weaver`` cache.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from arcasleep.data.cache import discover_cached_subjects
from arcasleep_bidoze.release import link, match_weaver, weaver_dataset_name

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
ARMS = ("matched", "all", "all-adult-share")
ADULT_SETS = {"trainSleepAccel": "SleepAccel", "trainClinical": "SleepAccel-Clinical",
              "testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical"}


def weaver_set(arm: str) -> str:
    return "matched" if arm == "matched" else "all"


def recordings(arm: str, cache_root: Path) -> list[str]:
    cache = cache_root / CACHE_NAME
    weaver = {s.subject_id: s.path for s in discover_cached_subjects(["weaver_revised"], cache)["weaver_revised"]}
    if weaver_set(arm) == "all":
        return sorted(weaver)
    earlier = discover_cached_subjects(["weaver"], cache)["weaver"]
    return match_weaver([s.path for s in earlier], weaver)["matched"]


def sampling_weights(draw: dict, matched_count: int) -> dict[str, float]:
    return {"trainSleepAccel": float(len(draw["trainSleepAccel"])),
            "trainClinical": float(len(draw["trainClinical"])),
            "weaverAll": float(matched_count)}


def main(fit_dir: Path, cache_root: Path, arm: str, draw_index: int, training_config: Path | None = None) -> None:
    draw = json.loads((HERE / "draws.json").read_text())[f"draw{draw_index}"]
    prepare(fit_dir, cache_root, arm, draw, draw_index, training_config)


def prepare(fit_dir: Path, cache_root: Path, arm: str, draw: dict, draw_index: int,
            training_config: Path | None = None) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    cache = cache_root / CACHE_NAME
    name = weaver_dataset_name(weaver_set(arm))
    chosen = recordings(arm, cache_root)
    sources = {
        "weaver_revised": {s.subject_id: s.path for s in discover_cached_subjects(["weaver_revised"], cache)["weaver_revised"]},
    }
    adult = discover_cached_subjects(["SleepAccel", "SleepAccel-Clinical"], cache)
    sources.update({c: {s.subject_id: s.path for s in subjects} for c, subjects in adult.items()})

    links = fit_dir / "cache" / CACHE_NAME
    for sid in chosen:
        link(sources["weaver_revised"][sid], links / f"{name}_subject_{sid}_features.npz")
    for dataset, cohort in ADULT_SETS.items():
        for sid in draw[dataset]:
            link(sources[cohort][sid], links / f"{dataset}_subject_{sid}_features.npz")

    cv = {"mode": "transfer", "train_sets": [name, "trainSleepAccel", "trainClinical"],
          "test_sets": ["testSleepAccel", "testClinical"], "validation_sets": [], "fold_label": f"draw{draw_index}"}
    training = json.loads((training_config or BIDOZE / "config/training-adult.json").read_text())
    if arm == "all-adult-share":
        matched_count = len(recordings("matched", cache_root))
        training["dataset_sampling_weights"] = sampling_weights(draw, matched_count)
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "cv.json").write_text(json.dumps(cv, indent=2) + "\n")
    (fit_dir / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    (fit_dir / "recordings.json").write_text(json.dumps(chosen, indent=1) + "\n")
    print(f"{arm} draw{draw_index}: {name} {len(chosen)} recordings; "
          f"{len(draw['trainSleepAccel']) + len(draw['trainClinical'])} training adults, "
          f"{len(draw['testSleepAccel']) + len(draw['testClinical'])} test adults")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3], int(sys.argv[4]),
         Path(sys.argv[5]).resolve() if len(sys.argv) > 5 else None)

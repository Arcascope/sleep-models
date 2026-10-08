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
- matched:       the October 6 Weaver training set as it appears in revised Weaver. For each
                 night of the earlier ``weaver`` cache, the revised recording with the same PSG
                 labels and the same spectra (202 recordings, 201 children; see match_weaver).
- matched-actigraph: the same nights, each child's ActiGraph recording instead.
- earlier:       the earlier ``weaver`` cache itself, as dataset ``weaver``: the October 6
                 training set as it was trained then.
- finetune:      no Weaver training data. The pediatric release's weights are trained further on
                 4/5 SleepAccel at a tenth of the recipe's learning rate, early-stopping on the
                 revised Weaver recordings of the children the release never saw (dataset
                 ``weaverUnseen``). Written only when the pediatric bundle is given.

Writes, under ``<run>/setup/``:
- cache/<arm>/12Hz_log_PSD_aligned/  symlinks only: the arm's Weaver recordings as dataset
                                      ``weaver<Arm>`` (``weaver`` for earlier), SleepAccel by
                                      fold, SleepAccel-Clinical
- cv/<arm>/cv_fold<k>.json            train on weaver<Arm> + the other 4 SleepAccel folds
- cv/<arm>/cv_final.json             train on weaver<Arm> + all of SleepAccel: the stage source
                                      that is released; scored on SleepAccel-Clinical only
- arms.json                           the subject ids of every arm (for finetune, its
                                      validation recordings)
- training.json                       the repo's balanced recipe with run_seed 0, seed_count 3
- training-finetune.json              training.json at a tenth of the learning rate, starting
                                      from the pediatric bundle's weights

Usage: python prepare.py <run dir> <weaver_revised cache root> <adult cache root> [pediatric bundle]
(cache roots are the directories that hold ``12Hz_log_PSD_aligned/``; the adult root holds
SleepAccel, SleepAccel-Clinical and the earlier ``weaver`` cache).
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
EARLIER = "earlier"
FINETUNE = "finetune"
FINETUNE_LR_FACTOR = 0.1


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


def epoch_spectra(path: Path) -> np.ndarray:
    """Per-epoch mean of the frame spectra, (epochs, frequencies, channels)."""
    z = np.load(path)
    X, epochs = z["X"][0], z["y"].shape[1]
    frames = X.shape[0] // epochs
    return X[: epochs * frames].reshape(epochs, frames, *X.shape[1:]).mean(axis=1)


def match_weaver(earlier: list[Path], revised: dict[str, Path]) -> dict[str, list[str]]:
    """The revised recordings that hold the earlier ``weaver`` nights.

    A revised recording holds an earlier night when their PSG labels agree epoch for epoch
    from the first epoch over the shorter of the two, and that covers ≥ 90% of the longer.
    Other devices of the same night can match too (one PSG per night), so the recording is the
    candidate whose epoch spectra are closest. Returns ``matched`` (that recording per night)
    and ``matched-actigraph`` (the ActiGraph recording of the same child and night, where there
    is one).
    """
    labels = {sid: np.load(path)["y"].ravel() for sid, path in revised.items()}
    matched, actigraph = [], []
    for path in earlier:
        y = np.load(path)["y"].ravel()
        candidates = [sid for sid, other in labels.items()
                      if min(len(y), len(other)) >= 0.9 * max(len(y), len(other))
                      and np.array_equal(y[: len(other)], other[: len(y)])]
        nights = {sid.split("_", 1)[0] + "_" + sid.rsplit("_", 1)[1] for sid in candidates}
        if len(nights) != 1:
            raise ValueError(f"{path.name}: {len(nights)} revised nights match its labels")
        spectra = epoch_spectra(path)

        def distance(sid: str) -> float:
            other = epoch_spectra(revised[sid])
            n = min(len(spectra), len(other))
            return float(np.mean(np.abs(spectra[:n] - other[:n])))

        best = min(candidates, key=distance)
        matched.append(best)
        child, _, night = best.split("_")
        if f"{child}_actigraph_{night}" in revised:
            actigraph.append(f"{child}_actigraph_{night}")
    return {"matched": sorted(matched), "matched-actigraph": sorted(actigraph)}


def dataset_name(arm: str) -> str:
    if arm == EARLIER:
        return "weaver"
    if arm == FINETUNE:
        return "weaverUnseen"
    return "weaver" + "".join(part.capitalize() for part in arm.split("-"))


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def main(run_dir: Path, weaver_root: Path, adult_root: Path, pediatric: Path | None = None) -> None:
    setup = run_dir / "setup"
    weaver = {s.subject_id: s.path
              for s in discover_cached_subjects(["weaver_revised"], weaver_root / CACHE_NAME)["weaver_revised"]}
    adult = discover_cached_subjects(["SleepAccel", "SleepAccel-Clinical", "weaver"], adult_root / CACHE_NAME)
    sleepaccel = {s.subject_id: s.path for s in adult["SleepAccel"]}
    clinical = adult["SleepAccel-Clinical"]
    plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")

    chosen = arms(sorted(weaver))
    chosen.update(match_weaver([s.path for s in adult["weaver"]], weaver))
    earlier = {s.subject_id: s.path for s in adult["weaver"]}
    chosen[EARLIER] = sorted(earlier)
    if pediatric is not None:
        # The release trained on every earlier night, which are the matched children.
        seen = {participant_id(s) for s in chosen["matched"]}
        chosen[FINETUNE] = sorted(s for s in weaver if participant_id(s) not in seen)
    setup.mkdir(parents=True, exist_ok=True)
    (setup / "arms.json").write_text(json.dumps(chosen, indent=1) + "\n")

    training = json.loads((BIDOZE / "config/training-balanced.json").read_text())
    training.update(run_seed=0, seed_count=SEEDS)
    (setup / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    if pediatric is not None:
        finetune = {**training, "lr": training["lr"] * FINETUNE_LR_FACTOR, "init_weights": str(pediatric)}
        (setup / f"training-{FINETUNE}.json").write_text(json.dumps(finetune, indent=2) + "\n")

    for arm, ids in chosen.items():
        name = dataset_name(arm)
        cache = setup / "cache" / arm / CACHE_NAME
        sources = earlier if arm == EARLIER else weaver
        for sid in ids:
            link(sources[sid], cache / f"{name}_subject_{sid}_features.npz")
        for fold, sids in plan.folds.items():
            for sid in sids:
                link(sleepaccel[sid], cache / f"{fold}_subject_{sid}_features.npz")
        for subject in clinical:
            link(subject.path, cache / subject.path.name)
        # The finetune arm's Weaver recordings are its validation set, not training data.
        weaver_sets, validation_sets = ([], [name]) if arm == FINETUNE else ([name], [])
        for k, held_out in enumerate(plan.fold_names):
            cv = {
                "mode": "transfer",
                "train_sets": weaver_sets + [f for f in plan.fold_names if f != held_out],
                "test_sets": [held_out, "SleepAccel-Clinical"],
                "validation_sets": validation_sets,
                "fold_label": f"fold{k}",
            }
            path = setup / "cv" / arm / f"cv_fold{k}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cv, indent=2) + "\n")
        final = {
            "mode": "transfer",
            "train_sets": weaver_sets + list(plan.fold_names),
            "test_sets": ["SleepAccel-Clinical"],
            "validation_sets": validation_sets,
            "fold_label": "final",
        }
        (setup / "cv" / arm / "cv_final.json").write_text(json.dumps(final, indent=2) + "\n")
        children = len({participant_id(s) for s in ids})
        print(f"{arm}: {name}, {len(ids)} recordings, {children} children")


if __name__ == "__main__":
    main(*(Path(a).resolve() for a in sys.argv[1:5]))

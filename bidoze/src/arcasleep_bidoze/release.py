"""Retraining the BiDoze releases with the gap rule: the adult and pediatric recipes.

Each release has a model config (``config/model-gap-rule.json``), a training config and phases
``fold0`` to ``fold4`` and ``final``; ``scripts/harness.sh`` with ``BIDOZE_RELEASE`` runs them.

- **pediatric:** ``config/training-pediatric.json`` on the Weaver cache, with the folds of
  ``config/splits/weaver_5fold.json``; the released model trains on all of Weaver and is scored
  on the adult cohorts.
- **adult:** ``config/training-adult.json`` (3 seeds). Each phase trains on revised-Weaver
  recordings (``--weaver-set``) and 4 of the 5 SleepAccel folds of
  ``config/splits/sleepaccel_5fold.json``; ``final`` trains on all of SleepAccel. The bundle is
  3 seed bundles and ``composite.json`` (the adult stage divisor).

``prepare_adult`` writes the adult phases' feature-cache links and cv configs;
``build_release`` writes the release bundle from a finished ``final`` phase.

Usage: python -m arcasleep_bidoze.release prepare-adult <setup dir> <cache root> [--weaver-set matched|all]
       python -m arcasleep_bidoze.release bundle <adult|pediatric> <final phase dir> <output> <version>
                [--train-cache <setup dir>/cache --cache-root <cache root>]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from arcasleep.data.cache import discover_cached_subjects, participant_id
from arcasleep.data.splits import read_plan

CACHE_NAME = "12Hz_log_PSD_aligned"
RELEASES = ("adult", "pediatric")
#: The revised-Weaver recordings of the adult training set: ``matched`` is the 202 recordings
#: of the earlier ``weaver`` cache's nights, ``all`` every recording.
WEAVER_SETS = ("matched", "all")
ADULT_SEEDS = 3
FOLDS = 5
MODEL_CONFIG = "config/model-gap-rule.json"
TRAINING_CONFIGS = {"adult": "config/training-adult.json", "pediatric": "config/training-pediatric.json"}
SLEEPACCEL_PLAN = "config/splits/sleepaccel_5fold.json"


def weaver_dataset_name(weaver_set: str) -> str:
    return "weaver" + weaver_set.capitalize()


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


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def prepare_adult(
    setup: Path, cache_root: Path, weaver_set: str = "matched", config_dir: Path = Path("config")
) -> list[str]:
    """Write the adult phases under ``setup``; returns the revised-Weaver recordings trained on.

    ``cache_root`` holds ``12Hz_log_PSD_aligned/`` with ``weaver_revised``, ``SleepAccel`` and
    ``SleepAccel-Clinical``, and for ``matched`` the earlier ``weaver`` cache. Writes
    - ``cache/12Hz_log_PSD_aligned/``: symlinks to the recordings of every phase;
    - ``cv/cv_fold<k>.json`` and ``cv/cv_final.json``;
    - ``training.json``: ``config/training-adult.json``;
    - ``recordings.json``: the revised-Weaver recordings trained on.
    """
    if weaver_set not in WEAVER_SETS:
        raise ValueError(f"weaver set must be one of {WEAVER_SETS}, got {weaver_set!r}")
    cache = Path(cache_root) / CACHE_NAME
    weaver = {s.subject_id: s.path for s in discover_cached_subjects(["weaver_revised"], cache)["weaver_revised"]}
    adult = discover_cached_subjects(["SleepAccel", "SleepAccel-Clinical"], cache)
    sleepaccel = {s.subject_id: s.path for s in adult["SleepAccel"]}
    plan = read_plan(Path(config_dir) / "splits/sleepaccel_5fold.json")
    if weaver_set == "all":
        chosen = sorted(weaver)
    else:
        earlier = discover_cached_subjects(["weaver"], cache)["weaver"]
        chosen = match_weaver([s.path for s in earlier], weaver)["matched"]

    setup = Path(setup)
    links = setup / "cache" / CACHE_NAME
    name = weaver_dataset_name(weaver_set)
    for sid in chosen:
        link(weaver[sid], links / f"{name}_subject_{sid}_features.npz")
    for fold, sids in plan.folds.items():
        for sid in sids:
            link(sleepaccel[sid], links / f"{fold}_subject_{sid}_features.npz")
    for subject in adult["SleepAccel-Clinical"]:
        link(subject.path, links / subject.path.name)

    (setup / "cv").mkdir(parents=True, exist_ok=True)
    for k, held_out in enumerate(plan.fold_names):
        cv = {
            "mode": "transfer",
            "train_sets": [name] + [f for f in plan.fold_names if f != held_out],
            "test_sets": [held_out, "SleepAccel-Clinical"],
            "validation_sets": [],
            "fold_label": f"fold{k}",
        }
        (setup / "cv" / f"cv_fold{k}.json").write_text(json.dumps(cv, indent=2) + "\n")
    final = {
        "mode": "transfer",
        "train_sets": [name] + list(plan.fold_names),
        "test_sets": ["SleepAccel-Clinical"],
        "validation_sets": [],
        "fold_label": "final",
    }
    (setup / "cv" / "cv_final.json").write_text(json.dumps(final, indent=2) + "\n")
    shutil.copy(Path(config_dir) / "training-adult.json", setup / "training.json")
    (setup / "recordings.json").write_text(json.dumps(chosen, indent=1) + "\n")
    print(f"{name}: {len(chosen)} recordings, {len({participant_id(s) for s in chosen})} children; "
          f"{len(plan.fold_names)} folds")
    return chosen


def build_release(
    release: str,
    final: Path,
    output: Path,
    version: str,
    *,
    train_cache: Path | None = None,
    cache_root: Path | None = None,
) -> Path:
    """The release bundle of a finished ``final`` phase directory.

    - pediatric: one release bundle of the phase's checkpoint.
    - adult: a release bundle per seed checkpoint in ``model_outputs/`` and ``composite.json``;
      the divisor comes from the phase's training nights (``train_cache`` holds
      ``12Hz_log_PSD_aligned/`` with their links) and
      the nights of BIDSleep under ``cache_root``, the adult reference.

    The bundle's card is the generated one, marked pending review.
    """
    from arcasleep.artifacts.publish import build_release_bundle

    final, output = Path(final), Path(output)
    if release == "pediatric":
        return build_release_bundle(final, output, version=version, family="bidoze", overwrite=True).directory
    if release != "adult":
        raise ValueError(f"release must be one of {RELEASES}, got {release!r}")
    if train_cache is None or cache_root is None:
        raise ValueError("the adult bundle needs the training cache and the cache root")
    from arcasleep_bidoze.adult import adult_stage_divisor, build_adult_bundle, stage_counts

    cv = json.loads((final / "config/cv.json").read_text())
    training = json.loads((final / "config/training.json").read_text())
    train = discover_cached_subjects(cv["train_sets"], Path(train_cache) / CACHE_NAME)
    train_paths = [s.path for subjects in train.values() for s in subjects]
    bidsleep = discover_cached_subjects(["bidsleep"], Path(cache_root) / CACHE_NAME)["bidsleep"]

    def counts(paths: list[Path]) -> np.ndarray:
        return np.array([stage_counts(np.load(p)["y"]) for p in paths])

    divisor = adult_stage_divisor(
        counts(train_paths), counts([s.path for s in bidsleep]),
        training["balance_beta"], training["batch_size"],
    )
    staging = output.parent / f"{output.name}-seeds"
    seeds = [
        build_release_bundle(final, staging / checkpoint.stem, version=version, checkpoint=checkpoint,
                             family="bidoze", overwrite=True).directory
        for checkpoint in sorted((final / "model_outputs").glob("*.pisces"))
    ]
    return build_adult_bundle(
        output, seeds, divisor, version=version, overwrite=True,
        provenance={
            "train_sets": cv["train_sets"],
            "training_nights": len(train_paths),
            "adult_reference": "bidsleep",
            "adult_reference_nights": len(bidsleep),
            "seed_checkpoints": [p.name for p in seeds],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-adult", help="write the adult phases' cache links and cv configs")
    prepare.add_argument("setup", type=Path)
    prepare.add_argument("cache_root", type=Path)
    prepare.add_argument("--weaver-set", choices=WEAVER_SETS, default="matched")
    prepare.add_argument("--config-dir", type=Path, default=Path("config"))
    bundle = commands.add_parser("bundle", help="write the release bundle of a finished final phase")
    bundle.add_argument("release", choices=RELEASES)
    bundle.add_argument("final", type=Path)
    bundle.add_argument("output", type=Path)
    bundle.add_argument("version")
    bundle.add_argument("--train-cache", type=Path)
    bundle.add_argument("--cache-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare-adult":
        prepare_adult(args.setup, args.cache_root, args.weaver_set, args.config_dir)
    else:
        print(build_release(args.release, args.final, args.output, args.version,
                            train_cache=args.train_cache, cache_root=args.cache_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

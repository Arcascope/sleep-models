"""Retraining the BiDoze releases with the gap rule: the adult and pediatric recipes.

Each release has a model config (``MODEL_CONFIGS``), a training config and phases
``fold0`` to ``fold4`` and ``final``; ``scripts/harness.sh`` with ``BIDOZE_RELEASE`` runs them.

- **pediatric:** ``config/training-pediatric.json`` with the folds of a split plan over
  Weaver (``--weaver-set``); the released model trains on all of Weaver and is scored on the
  adult cohorts. ``matched`` trains on the earlier ``weaver`` cache with
  ``config/splits/weaver_5fold.json`` (202 recordings); ``all`` trains on every revised-Weaver
  recording with ``config/splits/weaver_revised_pediatric_5fold.json``.
- **adult:** ``config/training-adult.json`` (3 seeds) and the F2 network (``--model``). Each phase
  trains on revised-Weaver recordings (``--weaver-set``; ``matched`` is the 202 recordings of the
  earlier nights), 4 of the 5 SleepAccel folds of ``config/splits/sleepaccel_5fold.json`` and 4 of
  the 5 Wearanize+ folds of ``config/splits/wearanize_5fold.json``; ``final`` trains on all of
  SleepAccel and Wearanize+. The bundle is 3 seed bundles and ``composite.json`` (the adult stage
  divisor).

``prepare_adult`` writes the adult phases' feature-cache links and cv configs;
``build_release`` writes the release bundle from a finished ``final`` phase.

Usage: python -m arcasleep_bidoze.release run <adult|pediatric> --run-id <id> --cache-root <cache root>
                [--weaver-set all|matched] [--phases fold0 ... final] [--version v0.2.0] [--model g|f2]
       python -m arcasleep_bidoze.release pediatric-recipe [--weaver-set all|matched]
       python -m arcasleep_bidoze.release prepare-adult <setup dir> <cache root> [--weaver-set all|matched]
       python -m arcasleep_bidoze.release bundle <adult|pediatric> <final phase dir> <output> <version>
                [--train-cache <setup dir>/cache --cache-root <cache root>] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from arcasleep.data.cache import discover_cached_subjects, participant_id
from arcasleep.data.splits import read_plan

CACHE_NAME = "12Hz_log_PSD_aligned"
RELEASES = ("adult", "pediatric")
#: The Weaver recordings a release trains on. Adult: ``matched`` is the 202 revised-Weaver
#: recordings of the earlier ``weaver`` cache's nights, ``all`` every revised recording.
#: Pediatric: ``matched`` is the earlier ``weaver`` cache itself, ``all`` every revised recording.
WEAVER_SETS = ("matched", "all")
#: Per Weaver set, the pediatric release's split plan and cv template (under ``config/``).
PEDIATRIC_RECIPES = {
    "matched": ("config/splits/weaver_5fold.json", "config/cv-balanced.json"),
    "all": ("config/splits/weaver_revised_pediatric_5fold.json", "config/cv-weaver-revised.json"),
}
ADULT_SEEDS = 3
FOLDS = 5
#: The release architectures: ``g`` is BiDoze-G, ``f2`` the strand-fusion network.
MODEL_CONFIGS = {"g": "config/model-gap-rule.json", "f2": "config/model-f2.json"}
MODEL_CONFIG = MODEL_CONFIGS["g"]
#: The architecture each release trains unless one is named.
DEFAULT_MODELS = {"adult": "f2", "pediatric": "g"}
DEFAULT_WEAVER_SET = "all"
TRAINING_CONFIGS = {"adult": "config/training-adult.json", "pediatric": "config/training-pediatric.json"}
SLEEPACCEL_PLAN = "config/splits/sleepaccel_5fold.json"
WEARANIZE_PLAN = "config/splits/wearanize_5fold.json"


def pediatric_recipe(weaver_set: str) -> tuple[str, str]:
    """The split plan and cv template of the pediatric release on ``weaver_set``."""
    if weaver_set not in WEAVER_SETS:
        raise ValueError(f"weaver set must be one of {WEAVER_SETS}, got {weaver_set!r}")
    return PEDIATRIC_RECIPES[weaver_set]


def weaver_dataset_name(weaver_set: str) -> str:
    return "weaver" + weaver_set.capitalize()


def epoch_spectra(path: Path) -> np.ndarray:
    """Per-epoch mean of the frame spectra, (epochs, frequencies, channels)."""
    z = np.load(path)
    X, epochs = z["X"][0], z["y"].shape[1]
    frames = X.shape[0] // epochs
    return X[: epochs * frames].reshape(epochs, frames, *X.shape[1:]).mean(axis=1)


def holding_recordings(y: np.ndarray, labels: dict[str, np.ndarray]) -> list[str]:
    """The revised recordings whose PSG labels hold the earlier night with labels ``y``.

    A revised recording holds an earlier night when their labels agree epoch for epoch from the
    first epoch over the shorter of the two, and that covers ≥ 90% of the longer. Other devices
    of the same night hold it too (one PSG per night). Raises if the candidates are not one night.
    """
    candidates = [sid for sid, other in labels.items()
                  if min(len(y), len(other)) >= 0.9 * max(len(y), len(other))
                  and np.array_equal(y[: len(other)], other[: len(y)])]
    nights = {sid.split("_", 1)[0] + "_" + sid.rsplit("_", 1)[1] for sid in candidates}
    if len(nights) != 1:
        raise ValueError(f"{len(nights)} revised nights match the labels")
    return candidates


def match_weaver(earlier: list[Path], revised: dict[str, Path]) -> dict[str, list[str]]:
    """The revised recordings that hold the earlier ``weaver`` nights.

    Of the recordings holding a night (``holding_recordings``), the match is the one whose epoch
    spectra are closest. Returns ``matched`` (that recording per night) and ``matched-actigraph``
    (the ActiGraph recording of the same child and night, where there is one).
    """
    labels = {sid: np.load(path)["y"].ravel() for sid, path in revised.items()}
    matched, actigraph = [], []
    for path in earlier:
        try:
            candidates = holding_recordings(np.load(path)["y"].ravel(), labels)
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
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


def checkpoint_seed(checkpoint: Path) -> int:
    """The fit seed in a checkpoint's name, ``model_<fold>_seed_<seed>.pisces``."""
    match = re.search(r"_seed_(\d+)$", checkpoint.stem)
    if match is None:
        raise ValueError(f"{checkpoint.name} does not name a seed")
    return int(match.group(1))


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def prepare_adult(
    setup: Path, cache_root: Path, weaver_set: str = DEFAULT_WEAVER_SET, config_dir: Path = Path("config")
) -> list[str]:
    """Write the adult phases under ``setup``; returns the revised-Weaver recordings trained on.

    ``cache_root`` holds ``12Hz_log_PSD_aligned/`` with ``weaver_revised``, ``SleepAccel``,
    ``SleepAccel-Clinical`` and ``wearanize``, and for ``matched`` the earlier ``weaver`` cache.
    Fold k holds out SleepAccel fold k and Wearanize+ fold k. Writes
    - ``cache/12Hz_log_PSD_aligned/``: symlinks to the recordings of every phase;
    - ``cv/cv_fold<k>.json`` and ``cv/cv_final.json``;
    - ``training.json``: ``config/training-adult.json``;
    - ``recordings.json``: the revised-Weaver recordings trained on.
    """
    if weaver_set not in WEAVER_SETS:
        raise ValueError(f"weaver set must be one of {WEAVER_SETS}, got {weaver_set!r}")
    cache = Path(cache_root) / CACHE_NAME
    weaver = {s.subject_id: s.path for s in discover_cached_subjects(["weaver_revised"], cache)["weaver_revised"]}
    adult = discover_cached_subjects(["SleepAccel", "SleepAccel-Clinical", "wearanize"], cache)
    sleepaccel = {s.subject_id: s.path for s in adult["SleepAccel"]}
    wearanize = {s.subject_id: s.path for s in adult["wearanize"]}
    plan = read_plan(Path(config_dir) / "splits/sleepaccel_5fold.json")
    wearanize_plan = read_plan(Path(config_dir) / "splits/wearanize_5fold.json")
    if len(wearanize_plan.fold_names) != len(plan.fold_names):
        raise ValueError(f"the SleepAccel and Wearanize+ plans have {len(plan.fold_names)} and "
                         f"{len(wearanize_plan.fold_names)} folds")
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
    for fold, sids in wearanize_plan.folds.items():
        for sid in sids:
            link(wearanize[sid], links / f"{fold}_subject_{sid}_features.npz")
    for subject in adult["SleepAccel-Clinical"]:
        link(subject.path, links / subject.path.name)

    (setup / "cv").mkdir(parents=True, exist_ok=True)
    for k, (held_out, held_out_wearanize) in enumerate(zip(plan.fold_names, wearanize_plan.fold_names)):
        cv = {
            "mode": "transfer",
            "train_sets": [name] + [f for f in plan.fold_names if f != held_out]
            + [f for f in wearanize_plan.fold_names if f != held_out_wearanize],
            "test_sets": [held_out, held_out_wearanize, "SleepAccel-Clinical"],
            "validation_sets": [],
            "fold_label": f"fold{k}",
        }
        (setup / "cv" / f"cv_fold{k}.json").write_text(json.dumps(cv, indent=2) + "\n")
    final = {
        "mode": "transfer",
        "train_sets": [name] + list(plan.fold_names) + list(wearanize_plan.fold_names),
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
    seeds: Sequence[int] | None = None,
) -> Path:
    """The release bundle of a finished ``final`` phase directory.

    - pediatric: one release bundle of the phase's checkpoint.
    - adult: a release bundle per seed checkpoint in ``model_outputs/`` and ``composite.json``;
      the divisor comes from the phase's training nights (``train_cache`` holds
      ``12Hz_log_PSD_aligned/`` with their links) and
      the nights of BIDSleep under ``cache_root``, the adult reference. With ``seeds`` (fit
      seeds), only those seeds' checkpoints are bundled; the divisor is computed from the
      training labels alone, so it is the same for any subset of the phase's seeds.

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
    checkpoints = [
        p for p in sorted((final / "model_outputs").glob("*.pisces"))
        if seeds is None or checkpoint_seed(p) in seeds
    ]
    if not checkpoints:
        raise ValueError(f"no checkpoint in {final / 'model_outputs'} has a seed in {list(seeds)}")
    seed_bundles = [
        build_release_bundle(final, staging / checkpoint.stem, version=version, checkpoint=checkpoint,
                             family="bidoze", overwrite=True).directory
        for checkpoint in checkpoints
    ]
    return build_adult_bundle(
        output, seed_bundles, divisor, version=version, overwrite=True,
        provenance={
            "train_sets": cv["train_sets"],
            "training_nights": len(train_paths),
            "adult_reference": "bidsleep",
            "adult_reference_nights": len(bidsleep),
            "seed_checkpoints": [p.name for p in seed_bundles],
        },
    )


def run_release(
    release: str,
    *,
    run_id: str,
    cache_root: Path,
    weaver_set: str = DEFAULT_WEAVER_SET,
    phases: list[str] | None = None,
    version: str = "v0.2.0",
    training_config: str | None = None,
    results_dir: Path | None = None,
    bidoze_dir: Path = Path("."),
    model: str | None = None,
) -> int:
    """Retrain ``release`` by running ``scripts/harness.sh`` of ``bidoze_dir`` (the ``bidoze/``
    directory); its exit status is returned. With no ``phases``, every phase runs; with no
    ``model``, the release's own architecture (``DEFAULT_MODELS``) trains."""
    if release not in RELEASES:
        raise ValueError(f"release must be one of {RELEASES}, got {release!r}")
    if weaver_set not in WEAVER_SETS:
        raise ValueError(f"weaver set must be one of {WEAVER_SETS}, got {weaver_set!r}")
    model = model or DEFAULT_MODELS.get(release)
    if model not in MODEL_CONFIGS:
        raise ValueError(f"model must be one of {tuple(MODEL_CONFIGS)}, got {model!r}")
    env = {
        **os.environ,
        "BIDOZE_RELEASE": release,
        "BIDOZE_RELEASE_MODEL": model,
        "BIDOZE_WEAVER_SET": weaver_set,
        "BIDOZE_RELEASE_VERSION": version,
        "BIDOZE_RUN_ID": run_id,
        "FEATURE_CACHE_DIR": str(Path(cache_root).resolve()),
    }
    if training_config:
        env["BIDOZE_RELEASE_TRAINING"] = training_config
    if results_dir:
        env["BIDOZE_RESULTS_DIR"] = str(Path(results_dir).resolve())
    bidoze_dir = Path(bidoze_dir)
    return subprocess.run(
        ["bash", str(bidoze_dir / "scripts/harness.sh"), *(phases or [])], cwd=bidoze_dir, env=env
    ).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="retrain a release locally: its phases, then its bundle")
    run.add_argument("release", choices=RELEASES)
    run.add_argument("--run-id", required=True)
    run.add_argument("--cache-root", type=Path, required=True)
    run.add_argument("--weaver-set", choices=WEAVER_SETS, default=DEFAULT_WEAVER_SET)
    run.add_argument("--phases", nargs="*", default=None, help="default: fold0 to fold4 and final")
    run.add_argument("--version", default="v0.2.0")
    run.add_argument("--training-config", help="a training config in place of the release's own")
    run.add_argument("--results-dir", type=Path)
    run.add_argument("--model", choices=tuple(MODEL_CONFIGS),
                     help="g: BiDoze-G; f2: strand-fusion network (default: f2 for adult, g for pediatric)")
    run.add_argument("--bidoze-dir", type=Path, default=Path("."), help="the bidoze/ directory")
    recipe = commands.add_parser("pediatric-recipe", help="print the pediatric split plan and cv template")
    recipe.add_argument("--weaver-set", choices=WEAVER_SETS, default=DEFAULT_WEAVER_SET)
    prepare = commands.add_parser("prepare-adult", help="write the adult phases' cache links and cv configs")
    prepare.add_argument("setup", type=Path)
    prepare.add_argument("cache_root", type=Path)
    prepare.add_argument("--weaver-set", choices=WEAVER_SETS, default=DEFAULT_WEAVER_SET)
    prepare.add_argument("--config-dir", type=Path, default=Path("config"))
    bundle = commands.add_parser("bundle", help="write the release bundle of a finished final phase")
    bundle.add_argument("release", choices=RELEASES)
    bundle.add_argument("final", type=Path)
    bundle.add_argument("output", type=Path)
    bundle.add_argument("version")
    bundle.add_argument("--train-cache", type=Path)
    bundle.add_argument("--cache-root", type=Path)
    bundle.add_argument("--seeds", type=int, nargs="+", help="adult only: bundle only these fit seeds "
                        "(one seed gives a single-model bundle)")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_release(
            args.release, run_id=args.run_id, cache_root=args.cache_root, weaver_set=args.weaver_set,
            phases=args.phases, version=args.version, training_config=args.training_config,
            results_dir=args.results_dir, bidoze_dir=args.bidoze_dir, model=args.model,
        )
    if args.command == "pediatric-recipe":
        print(*pediatric_recipe(args.weaver_set))
    elif args.command == "prepare-adult":
        prepare_adult(args.setup, args.cache_root, args.weaver_set, args.config_dir)
    else:
        print(build_release(args.release, args.final, args.output, args.version,
                            train_cache=args.train_cache, cache_root=args.cache_root, seeds=args.seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

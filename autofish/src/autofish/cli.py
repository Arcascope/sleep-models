"""Command line entry points for training, processing and manifests."""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

from autofish.artifacts.provenance import (
    build_local_manifest,
    build_s3_manifest,
    code_digest,
    data_digest,
    run_fingerprint,
    write_manifest,
)
from autofish.data.constants import MIN_SCORED_MINUTES


def _manifest_local(args: argparse.Namespace) -> int:
    entries = build_local_manifest(Path(args.root))
    digest = write_manifest(entries, Path(args.output))
    print(f"wrote {len(entries)} entries to {args.output}")
    print(f"manifest_sha256={digest}")
    return 0


def _manifest_s3(args: argparse.Namespace) -> int:
    try:
        entries = build_s3_manifest(
            args.bucket, args.prefix, require_sha256=not args.allow_missing_sha256
        )
    except ModuleNotFoundError as exc:
        if exc.name == "boto3":
            raise SystemExit(
                "S3 manifests need the optional dependency: "
                "pip install 'arcascope-autofish[aws]'"
            ) from exc
        raise
    digest = write_manifest(entries, Path(args.output))
    print(f"wrote {len(entries)} entries to {args.output}")
    print(f"manifest_sha256={digest}")
    return 0


DEFAULT_FEATURE_CACHE_DIR = "feature_cache"


def _resolve_run_id(results: Path) -> str:
    """Identify this run by what determines it: its inputs and its code.

    ``results`` ends in the phase directory (``<run>/training``), so its parent
    is the run directory. That name is the fallback identifier -- it is a label
    someone chose, not a fingerprint, so it is used only when one of the two
    halves is missing, which is normal on a dev box with no manifest baked in.
    """
    data = data_digest()
    code = code_digest()
    if data and code:
        fingerprint = run_fingerprint(data_digest=data, code_digest=code)
        print(f"run id {fingerprint}  (data {data[:12]} + code {code})")
        return fingerprint
    missing = "manifest" if not data else "git commit"
    fallback = results.parent.name or results.name
    print(f"run id {fallback}  (not fingerprinted: no {missing})")
    return fallback


def _train(args: argparse.Namespace) -> int:
    from autofish.training.config import RunConfig
    from autofish.training.driver import RunPaths, run_training

    # -extn is the run's results directory: a bare
    # name resolves under the results root, an absolute path is used as given.
    results = Path(os.environ.get("AUTOFISH_RESULTS_DIR", "results")) / args.run_extn
    feature_cache = Path(os.environ.get("FEATURE_CACHE_DIR", DEFAULT_FEATURE_CACHE_DIR))

    try:
        config = RunConfig.from_paths(
            inputs=args.processing_config,
            model=args.model_config,
            cv=args.cv_config,
            training=args.training_config,
            metrics=args.metrics_config,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    paths = RunPaths(results=results, feature_cache=feature_cache)
    cache_dir = paths.cache_dir(config)
    if not cache_dir.is_dir():
        raise SystemExit(
            f"feature cache not found: {cache_dir}\n"
            f"cache_prefix={config.cache_prefix!r} (from inputs.json); "
            "set FEATURE_CACHE_DIR to the cache root."
        )

    run_training(config, paths, run_id=_resolve_run_id(results), split_seed=args.split_seed)
    return 0


def _process(args: argparse.Namespace) -> int:
    from autofish.data.cache import resolve_feature_cache_dir
    from autofish.data.prepare import process_dataset
    from autofish.training.config import load_inputs

    try:
        processing_config, cache_prefix = load_inputs(args.processing_config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    cache_root = Path(
        args.output_dir
        or os.environ.get("FEATURE_CACHE_DIR", DEFAULT_FEATURE_CACHE_DIR)
    )
    cache_dir = resolve_feature_cache_dir(cache_root, cache_prefix)

    if args.sib_only:
        from autofish.data.prepare import backfill_sib

        for dataset_dir in args.data_dir:
            backfill_sib(
                dataset_dir,
                cache_dir,
                name=args.name or Path(dataset_dir.rstrip("/")).name,
                overwrite=args.overwrite,
            )
        return 0

    for dataset_dir in args.data_dir:
        process_dataset(
            dataset_dir,
            processing_config,
            cache_dir,
            # Only a fallback selector: the loaded dataset takes its name from
            # data_set.json (or its adapter), so a directory named for the
            # symlink pointing at it still reports the cohort it holds.
            name=args.name or Path(dataset_dir.rstrip("/")).name,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            min_scored_minutes=args.min_scored_minutes,
            load_workers=args.load_workers,
            write_workers=args.write_workers,
        )
    return 0


def _split_targets(cv_path: str) -> tuple[str, list[str], int]:
    """Read cohort, external test sets and k out of the cv template.

    The cohort is the dataset named in both train_sets and test_sets -- the
    same overlap that used to trigger a runtime 50/50 participant split. Here
    it names what gets k-folded instead.
    """
    from autofish.training.config import AutofishCVConfig

    cv = AutofishCVConfig.from_json(cv_path)
    overlap = sorted(set(cv.train_sets) & set(cv.test_sets))
    if len(overlap) != 1:
        raise SystemExit(
            f"{cv_path}: expected exactly one dataset in both train_sets and "
            f"test_sets to split, found {overlap}"
        )
    external = sorted(set(cv.test_sets) - set(cv.train_sets))
    if not cv.n_splits:
        raise SystemExit(f"{cv_path}: set n_splits to the number of folds")
    return overlap[0], external, int(cv.n_splits)


def _cache_dir_for(processing_config_path: str) -> Path:
    from autofish.data.cache import resolve_feature_cache_dir
    from autofish.training.config import load_inputs

    _, cache_prefix = load_inputs(processing_config_path)
    root = Path(os.environ.get("FEATURE_CACHE_DIR", DEFAULT_FEATURE_CACHE_DIR))
    return resolve_feature_cache_dir(root, cache_prefix)


def _splits_plan(args: argparse.Namespace) -> int:
    from autofish.data.cache import discover_cached_subjects
    from autofish.data.splits import DATA_DIGEST_KEY, plan_splits, write_plan

    cohort, external, n_splits = _split_targets(args.cv_config)
    cache_dir = _cache_dir_for(args.processing_config)
    cached = discover_cached_subjects([cohort], cache_dir)
    ids = [subject.subject_id for subject in cached[cohort]]
    digest = data_digest()
    plan = plan_splits(
        ids,
        cohort=cohort,
        n_splits=n_splits,
        seed=args.seed,
        provenance={
            "cache_dir": str(cache_dir),
            "external_test_sets": external,
            "subject_count": len(ids),
            # The bytes this split was drawn from. Without it, a plan and a
            # cache that disagree look identical to one that agrees.
            DATA_DIGEST_KEY: digest,
        },
    )
    path = write_plan(plan, args.output)
    print(f"wrote {path}  (data {digest[:12] if digest else 'unknown'})")
    for name in plan.fold_names:
        print(f"  {name}: {len(plan.folds[name])} subjects")
    print("commit this file: it is the split, not a recipe for one")
    return 0


def _splits_apply(args: argparse.Namespace) -> int:
    import json

    from autofish.data.cache import discover_cached_subjects
    from autofish.data.splits import (
        cv_configs_for_plan,
        data_drift_warning,
        extra_deblip_sets,
        materialise,
        read_plan,
        write_cv_configs,
    )

    cohort, external, _ = _split_targets(args.cv_config)
    plan = read_plan(args.plan)
    if plan.cohort != cohort:
        raise SystemExit(f"plan cohort {plan.cohort!r} != cv template cohort {cohort!r}")

    # Reported before materialising: when the subject check below fails, this
    # line is usually the reason.
    drift = data_drift_warning(plan, data_digest())
    if drift:
        print(f"warning: {drift}")

    template = json.loads(Path(args.cv_config).read_text())
    passthrough = sorted({*external, *extra_deblip_sets(template, cohort)})
    cache_dir = _cache_dir_for(args.processing_config)
    cached = discover_cached_subjects([cohort, *passthrough], cache_dir)
    links = materialise(plan, cached, args.link_dir, passthrough=passthrough)
    print(f"linked {len(links)} subjects into {args.link_dir}")

    written = write_cv_configs(
        cv_configs_for_plan(plan, template, external_test_sets=external), args.cv_out
    )
    for path in written:
        print(f"  {path}")
    return 0


DEFAULT_RESULTS_S3_PREFIX = "s3://arcascope-sleep-models-data/training_outputs"


def _upload_results(args: argparse.Namespace) -> int:
    from autofish.artifacts.upload import upload_run

    run_dir = Path(args.run_dir)
    destination = args.destination.rstrip("/")
    if args.run_id:
        destination = f"{destination}/{args.run_id}"

    try:
        written = upload_run(
            run_dir, destination, overwrite=args.overwrite, dry_run=args.dry_run
        )
    except ModuleNotFoundError as exc:
        if exc.name == "boto3":
            raise SystemExit(
                "uploading results needs the optional dependency: "
                "pip install 'arcascope-autofish[aws]'"
            ) from exc
        raise
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"{verb} {len(written)} objects to {destination}")
    print(f"  manifest: {written[-1]}")
    return 0


#: The configs a finished training phase left behind, read back by `evaluate`.
EVALUATION_CONFIGS = ("inputs.json", "model.json", "cv.json", "training.json", "metrics.json")

def _expand_inputs(patterns: list[str]) -> list[Path]:
    """Resolve ``-i`` values to cache files, quoted or already shell-expanded.

    A value that still contains glob metacharacters is expanded here, so the
    flag behaves the same whether the shell expanded it or not; a plain path is
    taken as given, which is also how a shell-expanded list arrives.
    """
    files: dict[Path, None] = {}
    for pattern in patterns:
        if glob.has_magic(pattern):
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise SystemExit(f"-i matched no files: {pattern}")
            files.update(dict.fromkeys(Path(match) for match in matches))
        else:
            files[Path(pattern)] = None
    if not files:
        raise SystemExit("-i: no input files")
    return list(files)


def _training_phase_configs(training_dir: Path) -> dict[str, Path]:
    """The self-describing config snapshots a finished phase leaves behind."""
    config_dir = training_dir / "config"
    paths = {name: config_dir / name for name in EVALUATION_CONFIGS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            f"{training_dir} is not a training phase directory; missing:\n  "
            + "\n  ".join(missing)
        )
    return paths


def _checkpoint_for(training_dir: Path, override: str | None) -> Path:
    """The model to score: named outright, or the phase's one checkpoint."""
    if override:
        checkpoint = Path(override)
        if not checkpoint.is_file():
            raise SystemExit(f"no such checkpoint: {checkpoint}")
        return checkpoint
    found = sorted((training_dir / "model_outputs").glob("*.pisces"))
    if len(found) != 1:
        names = [path.name for path in found]
        raise SystemExit(
            f"expected exactly one .pisces in {training_dir / 'model_outputs'}, "
            f"found {len(found)}: {names}; name the one you want with --checkpoint"
        )
    return found[0]


def _load_phase_config(training_dir: Path):
    """The run config a finished phase left behind, or a SystemExit saying why not."""
    from autofish.training.config import RunConfig

    config_paths = _training_phase_configs(training_dir)
    try:
        return RunConfig.from_paths(
            inputs=config_paths["inputs.json"],
            model=config_paths["model.json"],
            cv=config_paths["cv.json"],
            training=config_paths["training.json"],
            metrics=config_paths["metrics.json"],
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _evaluate(args: argparse.Namespace) -> int:
    """Score a checkpointed model on a cohort, using its training run's configs."""
    from autofish.data.cache import load_bundles

    fold_args = (args.fold_config, args.fold_training)
    if any(fold_args) and not all(fold_args):
        raise SystemExit(
            "cross-validation mode needs both -fc/--fold-config and "
            "-ft/--fold-training; give both or neither"
        )
    if args.fold_training:
        if args.training_dir:
            raise SystemExit("-t/--training-dir is not used with -ft/--fold-training")
        if args.checkpoint:
            raise SystemExit(
                "--checkpoint names one model, but cross-validation scores one "
                "checkpoint per fold; drop --checkpoint and let each phase's "
                "model_outputs/ supply it"
            )
        return _evaluate_folds(args)

    if not args.training_dir:
        raise SystemExit("evaluate needs -t/--training-dir, or -ft/--fold-training")

    from autofish.evaluation.run import lift_identity, run_evaluation

    training_dir = Path(args.training_dir)
    config = _load_phase_config(training_dir)
    checkpoint = _checkpoint_for(training_dir, args.checkpoint)
    try:
        subjects = load_bundles(_expand_inputs(args.input))
        run_evaluation(
            config,
            subjects=subjects,
            checkpoint=checkpoint,
            output_dir=Path(args.output_dir),
            identity=lift_identity(training_dir, config),
            provenance={"training_dir": str(training_dir), "input": list(args.input)},
            overwrite=args.overwrite,
            render_hypnograms=args.hypnograms,
            ggir_sib_filter=getattr(args, "ggir_sib_filter", False),
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


FOLD_DIR = re.compile(r"^fold\d+$")


def _evaluate_folds(args: argparse.Namespace) -> int:
    """Score every ``fold{i}`` model on exactly the subjects it held out.

    ``-ft`` points at a training root holding the per-fold phase directories.
    For each one this reads the phase's own ``cv.json`` to find which split-plan
    fold it tested on, keeps only those subjects from ``-i``, and scores the
    fold's checkpoint into ``<output-dir>/fold{i}/`` -- the layout
    :mod:`autofish.reporting.summary` pools into its ``cv`` cohort, so a run of
    models that never trained on the participant they are scored on lands in one
    summary.
    """
    from autofish.data.cache import load_bundles
    from autofish.data.splits import read_plan
    from autofish.evaluation.run import lift_identity, run_evaluation

    training_root = Path(args.fold_training)
    if not training_root.is_dir():
        raise SystemExit(f"no such fold-training root: {training_root}")
    try:
        plan = read_plan(args.fold_config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    fold_dirs = sorted(
        (path for path in training_root.iterdir() if path.is_dir() and FOLD_DIR.match(path.name)),
        key=lambda path: path.name,
    )
    if not fold_dirs:
        raise SystemExit(f"no fold<i> phase directories under {training_root}")

    output_root = Path(args.output_dir)
    try:
        all_subjects = load_bundles(_expand_inputs(args.input))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for fold_dir in fold_dirs:
        config = _load_phase_config(fold_dir)
        plan_folds = sorted(set(config.cv.test_sets) & set(plan.folds))
        if not plan_folds:
            raise SystemExit(
                f"{fold_dir.name}: cv.json test_sets {config.cv.test_sets} name no "
                f"fold in {args.fold_config} ({sorted(plan.folds)})"
            )
        held_out = {sid for name in plan_folds for sid in plan.folds[name]}
        subjects = {ref: bundle for ref, bundle in all_subjects.items() if ref.subject_id in held_out}
        if not subjects:
            raise SystemExit(
                f"{fold_dir.name}: none of the -i subjects are in {plan_folds}"
            )
        missing = sorted(held_out - {ref.subject_id for ref in subjects})
        if missing:
            print(
                f"[warn] {fold_dir.name}: {len(missing)} held-out subject(s) have no "
                f"-i feature cache and are not scored: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )

        checkpoint = _checkpoint_for(fold_dir, args.checkpoint)
        output_dir = output_root / fold_dir.name
        print(f"\n==> {fold_dir.name}: {len(subjects)} subjects from {plan_folds}")
        try:
            run_evaluation(
                config,
                subjects=subjects,
                checkpoint=checkpoint,
                output_dir=output_dir,
                identity=lift_identity(fold_dir, config),
                provenance={
                    "training_dir": str(fold_dir),
                    "fold_config": str(args.fold_config),
                    "held_out": plan_folds,
                    "input": list(args.input),
                },
                overwrite=args.overwrite,
                render_hypnograms=args.hypnograms,
                ggir_sib_filter=getattr(args, "ggir_sib_filter", False),
            )
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    print(f"\nCross-validation results: {output_root}")
    return 0


def _write_card_preview(bundle_dir: Path, preview: str) -> Path | None:
    """Render a bundle's card to HTML; ``preview`` may be a path or ``-``.

    Returns the written path, or ``None`` for stdout. An empty target or an
    existing directory gets ``card.html`` inside it, so ``--preview`` with no
    value (or a bundle directory) works without naming a file. Imported here so
    the preview dependency is only demanded when a preview is actually
    requested.
    """
    from autofish.artifacts.publish import render_card_html

    html = render_card_html(bundle_dir)
    if preview == "-":
        print(html)
        return None
    path = Path(preview) if preview else bundle_dir / "card.html"
    if path.is_dir():
        path = path / "card.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    print(f"card preview: {path}")
    return path


def _card_preview(args: argparse.Namespace) -> int:
    """Render a built bundle's model card to a standalone HTML page."""
    try:
        _write_card_preview(Path(args.bundle_dir), args.output)
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def _predict(args: argparse.Namespace) -> int:
    """Per-epoch sleep-stage CSVs from raw accelerometer files."""
    from autofish.predict import predict_files

    origin = args.origin
    if origin not in (None, "unix"):
        origin = float(origin)
    try:
        inputs = _expand_inputs(args.inputs)
        written = predict_files(
            args.bundle_dir,
            inputs,
            args.output_dir,
            origin,
            return_gap_head=args.return_gap_head,
            ggir_sleep_filter=args.ggir_sleep_filter,
            ggir_error_fails=args.ggir_error_fails,
            hypnogram=args.hypnogram,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for path in written:
        print(path)
    return 0


def _hypnogram(args: argparse.Namespace) -> int:
    """Hypnogram PNGs from prediction CSVs that `autofish predict` wrote."""
    from autofish.predict import read_prediction_csv
    from autofish.reporting.prediction_plot import plot_prediction

    inputs = [path for path in _expand_inputs(args.inputs) if not path.stem.endswith("_summary")]
    if not inputs:
        raise SystemExit("no prediction CSVs among the inputs (summary files are skipped)")
    for path in inputs:
        output_dir = Path(args.output_dir) if args.output_dir else path.parent
        try:
            prediction = read_prediction_csv(path)
            written = plot_prediction(prediction, output_dir / f"{path.stem}.png", title=path.stem)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        print(written)
    return 0


def _publish(args: argparse.Namespace) -> int:
    """Build a release bundle from a training phase, optionally ship it."""
    try:
        from autofish.artifacts.publish import (
            build_release_bundle,
            publish_to_hub,
            verify_bundle,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "safetensors":
            raise SystemExit(
                "building a release bundle needs the optional dependency: "
                "pip install 'arcascope-autofish[publish]'"
            ) from exc
        raise

    try:
        bundle = build_release_bundle(
            Path(args.training_dir),
            Path(args.output_dir),
            version=args.version,
            checkpoint=args.checkpoint,
            evaluation=args.evaluation,
            metrics_summary=args.metrics_summary,
            card=args.card,
            info=args.info,
            repo_id=args.repo,
            overwrite=args.overwrite,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        ModuleNotFoundError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"built {bundle.directory}  ({bundle.version})")
    print(f"  checkpoint sha256: {bundle.checkpoint_sha256}")
    print(f"  input shape:       {bundle.input_shape}")
    print(f"  card review:       {bundle.card_review_status}")
    for name in bundle.files:
        print(f"  {name}")

    try:
        verify_bundle(bundle.directory)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("checksums verified")

    if args.preview is not None:
        try:
            _write_card_preview(bundle.directory, args.preview)
        except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        # A preview is a build-and-look step: never upload, even if --repo is set.
        return 0

    if not args.repo:
        return 0

    if not bundle.approved and not args.allow_unreviewed:
        raise SystemExit(
            "refusing to upload a card that is still PENDING review; supply "
            "--card card.md or pass --allow-unreviewed to override"
        )

    try:
        publish_to_hub(
            bundle.directory,
            args.repo,
            revision=args.revision,
            private=args.private,
            dry_run=args.dry_run,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            raise SystemExit(
                "uploading to the Hub needs the optional dependency: "
                "pip install 'arcascope-autofish[hub]'"
            ) from exc
        raise

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"{verb} {len(bundle.files)} files to https://huggingface.co/{args.repo}")
    return 0


def _summarize(args: argparse.Namespace) -> int:
    from autofish.reporting.summary import summarize

    try:
        summarize([Path(d) for d in args.directory], Path(args.output_dir))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0

def _pair_summaries(entries: list[list[str]]) -> tuple[list[Path], list[str] | None]:
    """Turn ``-s PATH [NAME]`` groups into parallel paths and names.

    Names are all-or-nothing: if any summary is named, every one must be, so
    the legend is never a mix of chosen and invented labels.
    """
    paths: list[Path] = []
    names: list[str | None] = []
    for entry in entries:
        if len(entry) > 2:
            raise SystemExit(
                f"-s takes a path and at most one name; got {len(entry)}: {entry}"
            )
        paths.append(Path(entry[0]))
        names.append(entry[1] if len(entry) == 2 else None)
    if all(name is None for name in names):
        return paths, None
    if any(name is None for name in names):
        raise SystemExit("-s: name every summary or none of them")
    return paths, names


def _compare(args: argparse.Namespace) -> int:
    """Draw a metric grid comparing one or more summaries."""
    try:
        from autofish.reporting.compare import compare
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "comparing summaries needs the optional dependency: "
                "pip install 'arcascope-autofish[report]'"
            ) from exc
        raise

    paths, labels = _pair_summaries(args.summary)
    try:
        compare(
            paths,
            output_dir=Path(args.output_dir),
            labels=labels,
            name=args.name,
            title=args.title,
            agg=args.agg,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autofish")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="run the Autofish v1 recipe from a feature cache")
    train.add_argument("-pc", "--processing-config", required=True)
    train.add_argument("-mc", "--model-config", required=True)
    train.add_argument("-cvc", "--cv-config", required=True)
    train.add_argument("-tc", "--training-config", required=True)
    train.add_argument("-mtc", "--metrics-config", required=True)
    train.add_argument("-extn", "--run-extn", required=True)
    train.add_argument(
        "--cache-only",
        action="store_true",
        help="discover subjects by scanning the feature cache",
    )
    train.add_argument(
        "--split-seed",
        type=int,
        help="participant split seed; defaults to training.json run_seed",
    )
    train.set_defaults(handler=_train)

    process = commands.add_parser(
        "process", help="build the feature cache one dataset directory at a time"
    )
    process.add_argument("-pc", "--processing-config", required=True)
    process.add_argument(
        "-o",
        "--output-dir",
        help=(
            "feature-cache root to write into; defaults to $FEATURE_CACHE_DIR "
            f"or {DEFAULT_FEATURE_CACHE_DIR!r}"
        ),
    )
    process.add_argument(
        "--min-scored-minutes",
        type=float,
        default=MIN_SCORED_MINUTES,
        help=(
            "skip recordings with less staged PSG than this "
            f"(default {MIN_SCORED_MINUTES:.0f}); 0 disables the filter"
        ),
    )
    process.add_argument(
        "-dd",
        "--data-dir",
        action="append",
        required=True,
        help="dataset directory; repeatable",
    )
    process.add_argument(
        "--name",
        help="dataset name, when it differs from the directory's basename",
    )
    process.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="recordings packed into one NUFFT call (default: 8)",
    )
    process.add_argument(
        "--load-workers",
        type=int,
        help="processes parsing and cleaning CSVs (default: a third of the cores, at most 12)",
    )
    process.add_argument(
        "--write-workers",
        type=int,
        help="threads compressing NPZs (default: a third of the cores, at most 12)",
    )
    process.add_argument(
        "--sib-only",
        action="store_true",
        help="add the GGIR anglez_5s array to existing cache files without "
             "re-running the NUFFT; with --overwrite, replace one already there",
    )
    process.add_argument(
        "--overwrite", action="store_true", help="rebuild subjects already cached"
    )
    process.set_defaults(handler=_process)

    evaluate = commands.add_parser(
        "evaluate",
        help="score a saved checkpoint on a cohort, reusing its training run's configs",
    )
    evaluate.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="feature-cache NPZ files or glob patterns to score",
    )
    evaluate.add_argument(
        "-t",
        "--training-dir",
        help="training phase directory holding config/ and model_outputs/",
    )
    evaluate.add_argument(
        "-fc",
        "--fold-config",
        help="split plan (e.g. config/splits/weaver_5fold.json); with -ft scores "
             "each fold's model on the subjects that fold held out",
    )
    evaluate.add_argument(
        "-ft",
        "--fold-training",
        help="training root holding fold<i>/ phase directories; each is scored on "
             "its held-out fold into <output-dir>/fold<i>/",
    )
    evaluate.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="where cv_results_<n>_stages.csv and evaluation.json are written",
    )
    evaluate.add_argument(
        "--checkpoint",
        help="a .pisces file, when model_outputs/ holds more than one",
    )
    evaluate.add_argument(
        "-hypno",
        "--hypnograms",
        action="store_true",
        help="also draw one per-subject hypnogram PNG into <output-dir>/hypnograms/",
    )
    evaluate.add_argument(
        "--ggir-sib-filter",
        action="store_true",
        help="demote predicted sleep that is not a GGIR (vanHees2015) sustained "
             "inactivity bout to wake before scoring; needs anglez_5s in the "
             "cache (built by `process`, or added with `process --sib-only`)",
    )
    evaluate.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a results CSV already in the output directory",
    )
    evaluate.set_defaults(handler=_evaluate)

    summarize = commands.add_parser(
        "summarize",
        help="pool finished phases into a per-cohort metrics_summary.csv",
    )
    summarize.add_argument(
        "directory",
        nargs="+",
        help="phase directories (or parents holding them) with cv_results_*_stages.csv",
    )
    summarize.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="where metrics_summary.csv is written",
    )
    summarize.set_defaults(handler=_summarize)

    publish = commands.add_parser(
        "publish",
        help="build a safetensors release bundle from a training phase, optionally "
             "upload it to the Hugging Face Hub",
    )
    publish.add_argument(
        "-t",
        "--training-dir",
        required=True,
        help="training phase directory holding config/ and model_outputs/",
    )
    publish.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="where the release bundle (model.safetensors, config.json, ...) is written",
    )
    publish.add_argument(
        "--version",
        required=True,
        help="semantic release version, e.g. v1.0.0",
    )
    publish.add_argument(
        "--checkpoint",
        help="a .pisces file, when model_outputs/ holds more than one",
    )
    publish.add_argument(
        "--evaluation",
        help="an evaluation.json to fold into provenance and the card; defaults "
             "to the sibling validation/evaluation.json when present",
    )
    publish.add_argument(
        "--metrics-summary",
        help="a metrics_summary.csv whose per-cohort means go on the card",
    )
    publish.add_argument(
        "--info",
        help="release-info TOML for this model variant: repo, audience, protocol, "
             "cohort descriptions and the authored card Markdown (see "
             "cards/autofish-pediatric.toml)",
    )
    publish.add_argument(
        "--card",
        help="authored Markdown for the card (intended use, limitations, how to "
             "apply, metrics, figures); placed after the title with the generated "
             "facts following. Relative images are bundled into assets/. Overrides "
             "the card named by --info; with neither, the card carries a PENDING "
             "review marker",
    )
    publish.add_argument(
        "--preview",
        nargs="?",
        const="",
        metavar="PATH",
        help="render the card to this standalone HTML file ('-' for stdout) and "
             "stop; nothing is uploaded, even with --repo. Omit PATH to write "
             "card.html into the bundle, or give a directory to write it there",
    )
    publish.add_argument(
        "--repo",
        help="Hub model repo to upload to, e.g. org/autofish-variant; omit to only build "
             "(also titles the card, overriding the repo in --info)",
    )
    publish.add_argument("--revision", help="Hub branch/revision to commit to")
    publish.add_argument("--private", action="store_true", help="create the Hub repo private")
    publish.add_argument(
        "--dry-run", action="store_true", help="verify and report without uploading"
    )
    publish.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="upload even though the card's review marker is still PENDING",
    )
    publish.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    publish.set_defaults(handler=_publish)

    predict = commands.add_parser(
        "predict",
        help="write per-epoch sleep stages for raw accelerometer files, using a release bundle",
    )
    predict.add_argument("bundle_dir", help="a downloaded release bundle")
    predict.add_argument(
        "inputs",
        nargs="+",
        help="files of t [s], x, y, z [g] columns, comma- or space-separated; globs allowed",
    )
    predict.add_argument(
        "-o", "--output-dir", required=True, help="one <input stem>.csv is written here per input"
    )
    predict.add_argument(
        "--origin",
        help="where epoch 0 starts, in seconds on the input clock, or 'unix'; "
             "defaults to the first sample",
    )
    predict.add_argument(
        "--return-gap-head",
        action="store_true",
        help="add a gap_head column with the network's gap probability; the stage "
             "columns still sum to 1",
    )
    predict.add_argument(
        "--ggir-sleep-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="demote quiet wakefulness to wake and flag sleep outside the main "
             "sleep window as naps, GGIR-style (default: on; use "
             "--no-ggir-sleep-filter for the raw Autofish stage)",
    )
    predict.add_argument(
        "--ggir-error-fails",
        action="store_true",
        help="abort if the GGIR sleep filter raises, instead of warning and "
             "writing the raw Autofish stages",
    )
    predict.add_argument(
        "--hypnogram",
        action="store_true",
        help="also save a <input stem>.png hypnogram per input (needs the [report] extra)",
    )
    predict.set_defaults(handler=_predict)

    hypnogram = commands.add_parser(
        "hypnogram",
        help="draw hypnogram PNGs from prediction CSVs written by `autofish predict`",
    )
    hypnogram.add_argument(
        "inputs", nargs="+", help="prediction CSVs; globs allowed; *_summary.csv files are skipped"
    )
    hypnogram.add_argument(
        "-o", "--output-dir", help="where <csv stem>.png is written (default: beside each CSV)"
    )
    hypnogram.set_defaults(handler=_hypnogram)

    preview = commands.add_parser(
        "card-preview",
        help="render a built release bundle's model card to standalone HTML",
    )
    preview.add_argument(
        "bundle_dir",
        help="a built bundle directory holding README.md",
    )
    preview.add_argument(
        "-o",
        "--output",
        default="card_preview.html",
        help="where to write the HTML ('-' for stdout; default: card_preview.html)",
    )
    preview.set_defaults(handler=_card_preview)

    compare = commands.add_parser(
        "compare",
        help="draw a metric grid comparing several metrics_summary.csv files",
    )
    compare.add_argument(
        "-s",
        "--summary",
        nargs="+",
        action="append",
        required=True,
        metavar="PATH [NAME]",
        help="a metrics_summary.csv and an optional legend name (at most one), "
             "e.g. -s path/summary.csv 'v1 run'; repeatable, in legend order",
    )
    compare.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="where the figure is written",
    )
    compare.add_argument(
        "-a",
        "--agg",
        choices=("mean", "median"),
        default="mean",
        help="center each bar on the mean (95%% CI whisker) or the median "
             "(IQR whisker); default: mean",
    )
    compare.add_argument(
        "--name",
        default="metrics_grid",
        help="output file stem (default: metrics_grid)",
    )
    compare.add_argument("--title", help="figure title; a default is derived")
    compare.set_defaults(handler=_compare)

    plan = commands.add_parser(
        "splits-plan", help="assign a cohort to k folds and write the split artifact"
    )
    plan.add_argument("-pc", "--processing-config", required=True)
    plan.add_argument("-cvc", "--cv-config", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--seed", type=int, default=0)
    plan.set_defaults(handler=_splits_plan)

    apply_ = commands.add_parser(
        "splits-apply", help="materialise a split artifact as linked fold datasets"
    )
    apply_.add_argument("-pc", "--processing-config", required=True)
    apply_.add_argument("-cvc", "--cv-config", required=True)
    apply_.add_argument("--plan", required=True)
    apply_.add_argument("--link-dir", required=True)
    apply_.add_argument("--cv-out", required=True)
    apply_.set_defaults(handler=_splits_apply)

    upload = commands.add_parser(
        "upload-results", help="ship a finished run directory to S3 with a manifest"
    )
    upload.add_argument("--run-dir", required=True, help="the run directory to upload")
    upload.add_argument(
        "--destination",
        default=DEFAULT_RESULTS_S3_PREFIX,
        help=f"s3:// prefix to upload under (default {DEFAULT_RESULTS_S3_PREFIX})",
    )
    upload.add_argument(
        "--run-id", help="appended to --destination; usually the run fingerprint"
    )
    upload.add_argument(
        "--overwrite",
        action="store_true",
        help="replace objects already at the destination",
    )
    upload.add_argument("--dry-run", action="store_true", help="list what would be sent")
    upload.set_defaults(handler=_upload_results)

    local = commands.add_parser("manifest-local", help="hash a local dataset into canonical JSONL")
    local.add_argument("--root", required=True)
    local.add_argument("--output", required=True)
    local.set_defaults(handler=_manifest_local)

    s3 = commands.add_parser("manifest-s3", help="inventory versioned S3 objects with SHA-256")
    s3.add_argument("--bucket", required=True)
    s3.add_argument("--prefix", required=True)
    s3.add_argument("--output", required=True)
    s3.add_argument("--allow-missing-sha256", action="store_true")
    s3.set_defaults(handler=_manifest_s3)

    return parser


#: Optional modules and the extra that installs them. Handlers import their
#: dependencies lazily, so a base install reaches the command and then fails
#: here with a message naming what to install rather than a bare traceback.
_OPTIONAL_DEP_EXTRAS = {
    "numpy": "inference",
    "jax": "inference",
    "flax": "inference",
    "safetensors": "inference",
    "pisces_lite": "inference",
    "optax": "train",
    "matplotlib": "report",
    "pandas": "report",
    "boto3": "aws",
    "huggingface_hub": "hub",
    "markdown_it": "preview",
}


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except ModuleNotFoundError as exc:
        extra = _OPTIONAL_DEP_EXTRAS.get(exc.name or "")
        if extra is None:
            raise
        raise SystemExit(
            f"`autofish {args.command}` needs an optional dependency ({exc.name}): "
            f"pip install 'arcascope-autofish[{extra}]'"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())

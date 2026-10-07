"""Score a saved checkpoint on a cohort that was not part of its fit.

A training phase already scores its own held-out subjects
(:mod:`arcasleep.training.driver`). This module is the other half of that: point
an already-trained ``.pisces`` checkpoint at a set of cached recordings and
write the same metrics CSV, so an external cohort is scored by the identical
code and lands in a file that reads as another cohort of the run it came from.

The training run stays the source of truth. Its ``config/`` snapshots say how
the model was built and how the protocol scores it, and its results CSV carries
the identity columns -- run id, fold, epoch, training datasets -- that the new
rows are stamped with. Nothing here re-derives a config from scratch, so the
command cannot be pointed at a checkpoint and a set of configs that disagree.

The scoring context comes from the model family, not from this module: a
family fitted at a fixed context scores every recording at that length.
Shorter recordings are padded, long ones truncated -- and since truncation is
the case that loses data, it is reported.
"""
from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from pisces_lite.cv import SubjectRef
from pisces_lite.metrics import MetricsLogger
from pisces_lite.model_io import ModelIOBundle

from arcasleep.artifacts.provenance import (
    code_digest,
    data_digest,
    run_fingerprint,
    sha256_file,
)
from arcasleep.data.batching import recording_input
from arcasleep.evaluation import staging_metrics  # noqa: F401  (registers the metrics)
from arcasleep.evaluation.predictions import PREDICTIONS_DIRNAME, write_scored_prediction
from arcasleep.evaluation.staging_metrics import (
    GAP_STAGE_INDEX,
    ThresholdSearch,
    search_sleep_threshold,
    threshold_predictions,
)
from arcasleep.training.config import RunConfig
from arcasleep.training.driver import RULE, evaluate_subject


@dataclass(frozen=True)
class RunIdentity:
    """What a scored row says about where it came from.

    These are the logger's own identity columns. They are lifted from the run
    that produced the checkpoint rather than invented here, which is what lets
    a new cohort's CSV be concatenated with the training run's.
    """

    run_id: str
    fold: str
    epoch: int
    training_dataset: str

    def key(self) -> tuple[str, str, int, str]:
        return (self.run_id, self.fold, self.epoch, self.training_dataset)


def results_csv_path(directory: Path | str, config: RunConfig) -> Path:
    """Where a phase's metrics land -- the same name the trainer writes."""
    return Path(directory) / f"cv_results_{config.model.num_classes}_stages.csv"


def _identity_tuples(csv_path: Path) -> list[RunIdentity]:
    """The distinct identities a run's CSV holds, in a stable order."""
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    distinct = {
        identity.key(): identity
        for identity in (
            RunIdentity(
                run_id=row["run_id"],
                fold=row["fold"],
                epoch=int(row["epoch"]),
                training_dataset=row["training_dataset"],
            )
            for row in rows
        )
    }
    return [distinct[key] for key in sorted(distinct)]


def _unfingerprinted_run_id(training_dir: Path) -> str:
    """Identify a run the way ``train`` does when it cannot fingerprint.

    Only reached when a phase has no results CSV to lift an identity from, which
    means the trainer never got as far as recording a subject.
    """
    data, code = data_digest(), code_digest()
    if data and code:
        return run_fingerprint(data_digest=data, code_digest=code)
    return training_dir.parent.name or training_dir.name


def lift_identity(training_dir: Path | str, config: RunConfig) -> RunIdentity:
    """Take the identity for the new rows from the run that produced the checkpoint.

    The training phase's CSV already carries exactly one identity -- which run
    it was, which fold (``final_seed_0``), how many epochs, and which datasets
    it trained on. Lifting it verbatim is what makes the new rows join the old
    ones on ``subject_id``.

    Without a CSV the identity is reconstructed from the configs instead, and
    says so: it is a plausible identity, not a lifted one, and rows carrying it
    will not join anything.
    """
    training_dir = Path(training_dir)
    csv_path = results_csv_path(training_dir, config)
    if csv_path.is_file():
        identities = _identity_tuples(csv_path)
        if len(identities) == 1:
            return identities[0]
        raise ValueError(
            f"{csv_path} holds {len(identities)} distinct identities, so there is "
            "no single one to stamp on the new rows; point --training-dir at a "
            "single phase directory (e.g. .../training/final). Found: "
            + "; ".join(str(identity) for identity in identities)
        )

    training = config.training
    identity = RunIdentity(
        run_id=_unfingerprinted_run_id(training_dir),
        fold=f"{config.cv.fold_label or 'validation'}_seed_{training.seeds[0]}",
        epoch=training.epochs,
        training_dataset="+".join(sorted(config.cv.train_sets)),
    )
    print(
        f"  no {csv_path.name} in {training_dir}; reconstructed the identity from "
        f"config/ instead ({identity})"
    )
    return identity


def cohort_input_shape(
    config: RunConfig, subjects: Mapping[SubjectRef, ModelIOBundle]
) -> tuple[int, ...]:
    """The fixed shape the model runs at: the family's context, plus feature dims.

    The context length is the one the family scores this cohort at. The
    trailing frequency and channel dims are a property of the feature surface
    rather than of any one recording, so they are read once from the cohort and
    required to agree across it: one model cannot score a cohort whose
    recordings disagree about what a frame is.
    """
    if not subjects:
        raise ValueError("cohort_input_shape: no subjects")
    frames_per_epoch = config.model.frames_per_epoch
    by_dims: Dict[tuple[int, ...], list[str]] = {}
    for ref, bundle in subjects.items():
        inputs = recording_input(bundle.X[0], int(bundle.y.shape[1]), frames_per_epoch)
        by_dims.setdefault(tuple(inputs.shape[1:]), []).append(
            f"{ref.dataset_name}/{ref.subject_id}"
        )
    if len(by_dims) != 1:
        detail = "; ".join(
            f"{dims} ({len(refs)} subjects, e.g. {refs[0]})" for dims, refs in by_dims.items()
        )
        raise ValueError(f"cohort mixes feature surfaces: {detail}")
    feature_dims = next(iter(by_dims))
    x_length, _ = config.family.scoring_lengths(
        config, [int(bundle.y.shape[1]) for bundle in subjects.values()]
    )
    return (x_length * frames_per_epoch, *feature_dims)


#: The class index the metrics treat as wake (``staging_metrics._wake_class``).
WAKE_STAGE_INDEX = 0


def apply_sib_filter(
    proba: np.ndarray, anglez_5s: np.ndarray, n_epochs: int
) -> tuple[np.ndarray, int]:
    """Demote predicted sleep outside the SPT that is not a GGIR SIB.

    The same rule as :func:`arcasleep.predict.predict`'s corrected ``stage``:
    outside the main sleep window an epoch whose argmax is a sleep class but
    which vanHees2015 does not call a sustained inactivity bout becomes wake.
    Inside the main sleep window SIB never flips sleep to wake, and a
    sub-recording of at most 9 h is treated as one long main sleep window, so
    for a <= 9 h recording nothing is demoted. It is a hard
    decision, so the epoch's row is replaced by a one-hot wake row and every
    metric -- thresholded or argmax -- sees wake. Sleep is never promoted, gap
    epochs are untouched, and naps outside the main sleep window are kept, since
    the scored night has no notion of them.

    ``anglez_5s`` is the bundle's ``(epochs, 6)`` 5 s z-angle series
    (:class:`arcasleep.data.bundle.CacheBundle`). It covers the whole
    recording, so the judgement is made on all of it even when only the first
    ``n_epochs`` are scored -- the number of leading rows of ``proba`` that are
    real epochs rather than padding. Returns the filtered copy and the number
    of epochs demoted.
    """
    from arcasleep.ggir import evaluate_epoch_grid

    ggir = evaluate_epoch_grid(anglez_5s)
    is_sib = ggir.is_sib[:n_epochs]
    is_spt = ggir.is_spt[:n_epochs]
    filtered = np.array(proba, copy=True)
    predicted = filtered[:n_epochs].argmax(axis=-1)
    sleep = (predicted != WAKE_STAGE_INDEX) & (predicted != GAP_STAGE_INDEX)
    demote = np.zeros(filtered.shape[0], dtype=bool)
    demote[:n_epochs] = sleep & ~is_sib & ~is_spt
    filtered[demote] = 0.0
    filtered[demote, WAKE_STAGE_INDEX] = 1.0
    return filtered, int(demote.sum())


def _banner(message: str) -> None:
    print(f"\n{RULE}\n{message}\n{RULE}\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


HYPNOGRAM_DIRNAME = "hypnograms"


def _hypnogram_renderer(config: RunConfig):
    """The per-subject hypnogram call, or ``None`` when matplotlib is absent.

    Imported here rather than at module scope so a local inference install --
    which has neither matplotlib nor the ``report`` extra -- can still score a
    cohort; only ``--hypnograms`` needs the plotter.
    """
    try:
        from arcasleep.reporting.hypnogram import render_subject_hypnogram
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "drawing hypnograms needs the optional dependency: "
                "pip install 'arcasleep[report]'"
            ) from exc
        raise

    processing = config.processing
    feature_kind = "embedding" if getattr(processing, "type", None) == "healthssl" else "spectrogram"
    channel_names = (
        ["HealthSSL dim"]
        if feature_kind == "embedding"
        else getattr(processing, "spectral_channels", None)
    )

    def render(output_path: Path, inputs, probs, labels, title_prefix) -> dict[str, float]:
        return render_subject_hypnogram(
            output_path=output_path,
            inputs=inputs,
            probs=probs,
            labels=labels,
            title_prefix=title_prefix,
            channel_names=channel_names,
            feature_kind=feature_kind,
        )

    return render


@dataclass
class ScoredSubject:
    """One subject's scored truth and probabilities, ready to record."""

    ref: SubjectRef
    epochs: int
    y_true: np.ndarray
    y_pred_proba: np.ndarray


@dataclass
class ScoredCohort:
    """Every subject one checkpoint scored, and what scoring them reported."""

    config: RunConfig
    checkpoint: Path
    input_shape: tuple[int, ...]
    subjects: list[ScoredSubject]
    truncated: list[str]
    hypnograms: Dict[str, str]
    ggir_sib_filter: bool
    sib_demoted: Dict[str, int]
    predictions: Dict[str, str]


def prepare_output(config: RunConfig, output_dir: Path | str, overwrite: bool) -> Path:
    """Refuse to reuse an existing results CSV unless ``overwrite``.

    The metrics logger skips identity rows it has already written, so a second
    run into the same file would quietly score nothing rather than duplicate
    anything. Run before any scoring, so a refusal costs nothing. Replacing a
    run also clears its per-subject predictions, so none are left from subjects
    the new run does not score.
    """
    csv_path = results_csv_path(output_dir, config)
    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"{csv_path} already exists; pass --overwrite to replace it, or "
            "point --output-dir somewhere else"
        )
    for stale in (Path(output_dir) / PREDICTIONS_DIRNAME).glob("*.csv"):
        stale.unlink()
    return csv_path


def score_cohort(
    config: RunConfig,
    *,
    subjects: Mapping[SubjectRef, ModelIOBundle],
    checkpoint: Path | str,
    output_dir: Path | str,
    render_hypnograms: bool = False,
    ggir_sib_filter: bool = False,
) -> ScoredCohort:
    """Score every subject with ``checkpoint``, without recording any metric.

    ``render_hypnograms`` draws one per-subject PNG into
    ``<output_dir>/``:data:`HYPNOGRAM_DIRNAME` from the arrays that were
    scored, using :mod:`arcasleep.reporting.hypnogram`.

    Each subject's per-epoch prediction is written to
    ``<output_dir>/``:data:`PREDICTIONS_DIRNAME` (:mod:`arcasleep.evaluation.predictions`),
    from the model's probabilities before any SIB filtering.

    ``ggir_sib_filter`` demotes predicted sleep outside the main sleep window
    that is not a sustained inactivity bout to wake (:func:`apply_sib_filter`)
    before anything is drawn or recorded, using the ``anglez_5s`` each cached
    bundle carries. Inside the window sleep is left alone, and a <= 9 h
    recording is all one main sleep window, so nothing in it is demoted. A
    subject whose cache lacks ``anglez_5s`` is an error, raised before any
    scoring -- never a silently unfiltered row.
    """
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if ggir_sib_filter:
        missing = [
            f"{ref.dataset_name}/{ref.subject_id}"
            for ref, bundle in subjects.items()
            if getattr(bundle, "anglez_5s", None) is None
        ]
        if missing:
            raise ValueError(
                f"--ggir-sib-filter needs anglez_5s in the feature cache, which "
                f"{len(missing)} subject(s) lack (e.g. {missing[0]}); add it with "
                "`arcasleep process --sib-only`"
            )
    renderer = _hypnogram_renderer(config) if render_hypnograms else None
    input_shape = cohort_input_shape(config, subjects)
    x_length, y_length = config.family.scoring_lengths(
        config, [int(bundle.y.shape[1]) for bundle in subjects.values()]
    )
    num_classes = config.model.num_classes
    datasets = sorted({ref.dataset_name for ref in subjects})
    hypnogram_dir = output_dir / HYPNOGRAM_DIRNAME
    predictions_dir = output_dir / PREDICTIONS_DIRNAME

    _banner(
        f"EVALUATE: {checkpoint}\n"
        f"subjects: {len(subjects)} from {datasets}\n"
        f"input_shape: {input_shape}\n"
        f"hypnograms: {'yes -> ' + str(hypnogram_dir) if renderer else 'no'}\n"
        f"ggir sib filter: {'yes' if ggir_sib_filter else 'no'}"
    )

    model = config.family.load_checkpoint(checkpoint, input_shape=input_shape)
    scored_subjects: list[ScoredSubject] = []
    truncated: list[str] = []
    hypnograms: Dict[str, str] = {}
    sib_demoted: Dict[str, int] = {}
    predictions: Dict[str, str] = {}
    for ref, bundle in subjects.items():
        name = f"{ref.dataset_name}/{ref.subject_id}"
        epochs = int(bundle.y.shape[1])
        if epochs > x_length:
            truncated.append(name)
            warnings.warn(
                f"{name}: {epochs} epochs exceeds the model's {x_length}-epoch "
                f"context; scoring the first {x_length} and dropping {epochs - x_length}",
                RuntimeWarning,
                stacklevel=2,
            )
        scored = evaluate_subject(
            model,
            bundle,
            target_x_length=x_length,
            target_y_length=y_length,
            num_classes=num_classes,
            return_context=renderer is not None,
        )
        if renderer is not None:
            y_true, y_pred_proba, inputs, labels = scored
        else:
            y_true, y_pred_proba = scored
        predictions[name] = str(
            write_scored_prediction(
                predictions_dir, ref, bundle, y_true, y_pred_proba, num_classes
            )
        )
        if ggir_sib_filter:
            try:
                y_pred_proba, sib_demoted[name] = apply_sib_filter(
                    y_pred_proba, bundle.anglez_5s[0], min(epochs, x_length)
                )
            except Exception as exc:
                raise ValueError(
                    f"{name}: GGIR SIB filter failed: {type(exc).__name__}: {exc}"
                ) from exc
        if renderer is not None:
            output_path = hypnogram_dir / f"{ref.dataset_name}_{ref.subject_id}.png"
            renderer(output_path, inputs, y_pred_proba, labels, name)
            hypnograms[name] = str(output_path)
        scored_subjects.append(ScoredSubject(ref, epochs, y_true, y_pred_proba))
        print(f"  scored {name} ({epochs} epochs)")

    return ScoredCohort(
        config=config,
        checkpoint=checkpoint,
        input_shape=input_shape,
        subjects=scored_subjects,
        truncated=truncated,
        hypnograms=hypnograms,
        ggir_sib_filter=ggir_sib_filter,
        sib_demoted=sib_demoted,
        predictions=predictions,
    )


def operating_point(cohorts: Sequence[ScoredCohort], target_sensitivity: float) -> ThresholdSearch:
    """The sleep threshold meeting ``target_sensitivity`` over every subject of ``cohorts``.

    Sensitivity is averaged over subjects, as ``sleep_sensitivity`` is; pooling
    several cohorts -- the folds of a cross-validation -- gives them one
    operating point.
    """
    recordings = [
        (subject.y_true, subject.y_pred_proba)
        for cohort in cohorts
        for subject in cohort.subjects
    ]
    return search_sleep_threshold(recordings, target_sensitivity=target_sensitivity)


def write_cohort(
    cohort: ScoredCohort,
    *,
    output_dir: Path | str,
    identity: RunIdentity,
    provenance: Mapping[str, object] | None = None,
    threshold: ThresholdSearch | None = None,
    target_sensitivity: float | None = None,
) -> Path:
    """Record a scored cohort's metrics and ``evaluation.json``.

    With ``threshold`` every hard-prediction metric scores the calls at that
    sleep threshold rather than the argmax.
    """
    output_dir = Path(output_dir)
    config = cohort.config
    csv_path = results_csv_path(output_dir, config)
    csv_path.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(config.metrics)
    gap_classes = (GAP_STAGE_INDEX,) if config.model.output_classes > config.model.num_classes else ()

    for subject in cohort.subjects:
        y_pred = None
        extra = None
        if threshold is not None:
            y_pred = threshold_predictions(
                subject.y_pred_proba, threshold.threshold, WAKE_STAGE_INDEX, gap_classes
            )
            extra = {"sleep_threshold": threshold.threshold}
        logger.record(
            csv_path=csv_path,
            y_true=subject.y_true,
            y_pred_proba=subject.y_pred_proba,
            y_pred=y_pred,
            run_id=identity.run_id,
            fold=identity.fold,
            epoch=identity.epoch,
            subject_id=subject.ref.subject_id,
            dataset=subject.ref.dataset_name,
            training_dataset=identity.training_dataset,
            extra=extra,
        )

    by_dataset: Dict[str, list[str]] = {}
    for subject in cohort.subjects:
        by_dataset.setdefault(subject.ref.dataset_name, []).append(subject.ref.subject_id)
    operating = None
    if threshold is not None:
        operating = {
            "target_sensitivity": target_sensitivity,
            "sleep_threshold": threshold.threshold,
            "mean_sensitivity": threshold.sensitivity,
            "attained": threshold.attained,
        }
    _write_json(
        output_dir / "evaluation.json",
        {
            "identity": asdict(identity),
            "input_shape": list(cohort.input_shape),
            "checkpoint": {
                "path": str(cohort.checkpoint),
                "sha256": sha256_file(cohort.checkpoint),
            },
            "subject_count": len(cohort.subjects),
            "subjects": by_dataset,
            "truncated": cohort.truncated,
            "hypnograms": cohort.hypnograms,
            "predictions": cohort.predictions,
            "ggir_sib_filter": cohort.ggir_sib_filter,
            "sib_demoted_epochs": cohort.sib_demoted,
            "operating_point": operating,
            **dict(provenance or {}),
        },
    )

    _summarise(csv_path, len(cohort.subjects))
    print(f"\nResults: {output_dir}")
    return output_dir


def run_evaluation(
    config: RunConfig,
    *,
    subjects: Mapping[SubjectRef, ModelIOBundle],
    checkpoint: Path | str,
    output_dir: Path | str,
    identity: RunIdentity,
    provenance: Mapping[str, object] | None = None,
    overwrite: bool = False,
    render_hypnograms: bool = False,
    ggir_sib_filter: bool = False,
    target_sensitivity: float | None = None,
) -> Path:
    """Score every subject and return the directory the results landed in.

    See :func:`score_cohort` for hypnograms and the SIB filter. With
    ``target_sensitivity`` the metrics are scored at the sleep threshold that
    meets it over these subjects (:func:`operating_point`); the filter, which
    demotes argmax sleep, cannot be combined with it.
    """
    if target_sensitivity is not None and ggir_sib_filter:
        raise ValueError("--target-sensitivity cannot be combined with --ggir-sib-filter")
    prepare_output(config, output_dir, overwrite)
    # Resolve the renderer before scoring: a missing matplotlib must fail
    # before any work is done.
    if render_hypnograms:
        _hypnogram_renderer(config)
    cohort = score_cohort(
        config,
        subjects=subjects,
        checkpoint=checkpoint,
        output_dir=output_dir,
        render_hypnograms=render_hypnograms,
        ggir_sib_filter=ggir_sib_filter,
    )
    threshold = None
    if target_sensitivity is not None:
        threshold = operating_point([cohort], target_sensitivity)
        print(f"  operating point: {threshold}")
    return write_cohort(
        cohort,
        output_dir=output_dir,
        identity=identity,
        provenance=provenance,
        threshold=threshold,
        target_sensitivity=target_sensitivity,
    )


def _summarise(csv_path: Path, subject_count: int) -> None:
    """Print the cohort's macro means -- the mean over subjects, as the report reads it.

    Read with the stdlib rather than pandas: this is the package's own summary,
    and pandas is not a runtime dependency of it.
    """
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_metric: Dict[str, list[float]] = {}
    for row in rows:
        try:
            value = float(row["value"])
        except ValueError:
            continue
        if np.isnan(value):
            continue
        by_metric.setdefault(row["metric"], []).append(value)

    print(f"\n{len(rows)} rows, {subject_count} subjects -> {csv_path}")
    for metric in sorted(by_metric):
        values = np.asarray(by_metric[metric])
        print(f"  {metric:<20} {values.mean():>10.4f}  (n={values.size})")

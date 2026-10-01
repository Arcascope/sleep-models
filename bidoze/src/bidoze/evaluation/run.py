"""Score a saved checkpoint on a cohort that was not part of its fit.

A training phase already scores its own held-out subjects
(:mod:`bidoze.training.driver`). This module is the other half of that: point
an already-trained ``.pisces`` checkpoint at a set of cached recordings and
write the same metrics CSV, so an external cohort is scored by the identical
code and lands in a file that reads as another cohort of the run it came from.

The training run stays the source of truth. Its ``config/`` snapshots say how
the model was built and how the protocol scores it, and its results CSV carries
the identity columns -- run id, fold, epoch, training datasets -- that the new
rows are stamped with. Nothing here re-derives a config from scratch, so the
command cannot be pointed at a checkpoint and a set of configs that disagree.

One thing is deliberately *not* taken from the data: the token context. The
model was fitted at a fixed ``x_length`` and is scored at that same length,
never resized per recording. Shorter recordings are padded, long ones truncated
-- and since truncation is the case that loses data, it is reported.
"""
from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from pisces_lite.cv import SubjectRef
from pisces_lite.metrics import MetricsLogger
from pisces_lite.model_io import ModelIOBundle

from bidoze.artifacts.provenance import (
    code_digest,
    data_digest,
    run_fingerprint,
    sha256_file,
)
from bidoze.data.batching import pool_frames_to_epochs
from bidoze.evaluation import staging_metrics  # noqa: F401  (registers the metrics)
from bidoze.evaluation.staging_metrics import GAP_STAGE_INDEX
from bidoze.model import BiDozeNetModel
from bidoze.training.config import RunConfig
from bidoze.training.driver import RULE, evaluate_subject


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
    """Identify a run the way ``bidoze train`` does when it cannot fingerprint.

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
    """The fixed shape the model runs at: the config's context, plus feature dims.

    The context length is the trained model's, read from ``training.json`` --
    evaluation never resizes it per recording. The trailing frequency and
    channel dims are a property of the feature surface rather than of any one
    recording, so they are read once from the cohort and required to agree
    across it: one model cannot score a cohort whose recordings disagree about
    what a frame is.
    """
    if not subjects:
        raise ValueError("cohort_input_shape: no subjects")
    by_dims: Dict[tuple[int, ...], list[str]] = {}
    for ref, bundle in subjects.items():
        pooled = pool_frames_to_epochs(bundle.X, bundle.y)
        by_dims.setdefault(tuple(pooled.shape[2:]), []).append(
            f"{ref.dataset_name}/{ref.subject_id}"
        )
    if len(by_dims) != 1:
        detail = "; ".join(
            f"{dims} ({len(refs)} subjects, e.g. {refs[0]})" for dims, refs in by_dims.items()
        )
        raise ValueError(f"cohort mixes feature surfaces: {detail}")
    feature_dims = next(iter(by_dims))
    return (config.training.x_length, *feature_dims)


#: The class index the metrics treat as wake (``staging_metrics._wake_class``).
WAKE_STAGE_INDEX = 0


def apply_sib_filter(
    proba: np.ndarray, anglez_5s: np.ndarray, n_epochs: int
) -> tuple[np.ndarray, int]:
    """Demote predicted sleep that is not a GGIR sustained inactivity bout.

    The same rule as :func:`bidoze.predict.predict`'s corrected ``stage``:
    an epoch whose argmax is a sleep class but which vanHees2015 does not call
    a sustained inactivity bout becomes wake. It is a hard decision, so the
    epoch's row is replaced by a one-hot wake row and every metric --
    thresholded or argmax -- sees wake. Sleep is never promoted, gap epochs are
    untouched, and naps outside the main sleep window are kept, since the
    scored night has no notion of them.

    ``anglez_5s`` is the bundle's ``(epochs, 6)`` 5 s z-angle series
    (:class:`bidoze.data.bundle.BiDozeBundle`). It covers the whole
    recording, so the judgement is made on all of it even when only the first
    ``n_epochs`` are scored -- the number of leading rows of ``proba`` that are
    real epochs rather than padding. Returns the filtered copy and the number
    of epochs demoted.
    """
    from bidoze.ggir import EPOCH_SECONDS, bins_from_epochs, evaluate_bins

    total = anglez_5s.shape[0]
    bin_start_s, anglez = bins_from_epochs(anglez_5s)
    is_sib = evaluate_bins(bin_start_s, anglez, EPOCH_SECONDS * np.arange(total), total).is_sib[
        :n_epochs
    ]
    filtered = np.array(proba, copy=True)
    predicted = filtered[:n_epochs].argmax(axis=-1)
    sleep = (predicted != WAKE_STAGE_INDEX) & (predicted != GAP_STAGE_INDEX)
    demote = np.zeros(filtered.shape[0], dtype=bool)
    demote[:n_epochs] = sleep & ~is_sib
    filtered[demote] = 0.0
    filtered[demote, WAKE_STAGE_INDEX] = 1.0
    return filtered, int(demote.sum())


def _banner(message: str) -> None:
    print(f"\n{RULE}\n{message}\n{RULE}\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


HYPNOGRAM_DIRNAME = "hypnograms"


def _hypnogram_renderer(config: RunConfig, stage_style: str = "apple"):
    """The per-subject hypnogram call, or ``None`` when matplotlib is absent.

    Imported here rather than at module scope so a local inference install --
    which has neither matplotlib nor the ``report`` extra -- can still score a
    cohort; only ``--hypnograms`` needs the plotter.
    """
    try:
        from bidoze.reporting.hypnogram import render_subject_hypnogram
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "drawing hypnograms needs the optional dependency: "
                "pip install 'arcasleep-bidoze[report]'"
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
            stage_style=stage_style,
            feature_kind=feature_kind,
        )

    return render


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
    stage_style: str = "apple",
    ggir_sib_filter: bool = False,
) -> Path:
    """Score every subject and return the directory the results landed in.

    Refuses to reuse an existing results CSV unless ``overwrite``: the metrics
    logger skips identity rows it has already written, so a second run into the
    same file would quietly score nothing rather than duplicate anything.

    ``render_hypnograms`` additionally draws one per-subject PNG into
    ``<output_dir>/``:data:`HYPNOGRAM_DIRNAME` from the arrays that were scored, using
    :mod:`bidoze.reporting.hypnogram` -- the same figure the research
    repository renders during training.

    ``ggir_sib_filter`` demotes predicted sleep that is not a sustained
    inactivity bout to wake (:func:`apply_sib_filter`) before anything is scored
    or drawn, using the ``anglez_5s`` each cached bundle carries. A subject
    whose cache lacks it is an error, raised before any scoring -- never a
    silently unfiltered row.
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
                "`bidoze process --sib-only`"
            )
    # Resolve the renderer before touching the CSV: a missing matplotlib must
    # fail before an existing results file is deleted.
    renderer = _hypnogram_renderer(config, stage_style) if render_hypnograms else None
    csv_path = results_csv_path(output_dir, config)
    if csv_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{csv_path} already exists; pass --overwrite to replace it, or "
                "point --output-dir somewhere else"
            )
        csv_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    input_shape = cohort_input_shape(config, subjects)
    training = config.training
    num_classes = config.model.num_classes
    datasets = sorted({ref.dataset_name for ref in subjects})
    hypnogram_dir = output_dir / HYPNOGRAM_DIRNAME

    _banner(
        f"EVALUATE: {identity.run_id}\n"
        f"checkpoint: {checkpoint}\n"
        f"subjects: {len(subjects)} from {datasets}\n"
        f"input_shape: {input_shape} (x_length from training.json)\n"
        f"identity: fold={identity.fold} epoch={identity.epoch}\n"
        f"          training_dataset={identity.training_dataset}\n"
        f"hypnograms: {'yes -> ' + str(hypnogram_dir) if renderer else 'no'}\n"
        f"ggir sib filter: {'yes' if ggir_sib_filter else 'no'}"
    )

    model = BiDozeNetModel.loadz(checkpoint, input_shape=input_shape)
    logger = MetricsLogger(config.metrics)

    truncated: list[SubjectRef] = []
    hypnograms: Dict[str, str] = {}
    sib_demoted: Dict[str, int] = {}
    for ref in subjects:
        bundle = subjects[ref]
        epochs = int(bundle.y.shape[1])
        if epochs > training.x_length:
            truncated.append(ref)
            warnings.warn(
                f"{ref.dataset_name}/{ref.subject_id}: {epochs} epochs exceeds the "
                f"model's {training.x_length}-epoch context; scoring the first "
                f"{training.x_length} and dropping {epochs - training.x_length}",
                RuntimeWarning,
                stacklevel=2,
            )
        scored = evaluate_subject(
            model,
            bundle,
            target_x_length=training.x_length,
            target_y_length=training.y_length,
            num_classes=num_classes,
            return_context=renderer is not None,
        )
        if renderer is not None:
            y_true, y_pred_proba, inputs, labels = scored
        else:
            y_true, y_pred_proba = scored
        if ggir_sib_filter:
            name = f"{ref.dataset_name}/{ref.subject_id}"
            try:
                y_pred_proba, sib_demoted[name] = apply_sib_filter(
                    y_pred_proba, bundle.anglez_5s[0], min(epochs, training.x_length)
                )
            except Exception as exc:
                raise ValueError(f"{name}: GGIR SIB filter failed: {type(exc).__name__}: {exc}") from exc
        if renderer is not None:
            output_path = hypnogram_dir / f"{ref.dataset_name}_{ref.subject_id}.png"
            renderer(
                output_path,
                inputs,
                y_pred_proba,
                labels,
                f"{ref.dataset_name}/{ref.subject_id}",
            )
            hypnograms[f"{ref.dataset_name}/{ref.subject_id}"] = str(output_path)
        logger.record(
            csv_path=csv_path,
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            run_id=identity.run_id,
            fold=identity.fold,
            epoch=identity.epoch,
            subject_id=ref.subject_id,
            dataset=ref.dataset_name,
            training_dataset=identity.training_dataset,
        )
        print(f"  scored {ref.dataset_name}/{ref.subject_id} ({epochs} epochs)")

    by_dataset: Dict[str, list[str]] = {}
    for ref in subjects:
        by_dataset.setdefault(ref.dataset_name, []).append(ref.subject_id)
    _write_json(
        output_dir / "evaluation.json",
        {
            "identity": asdict(identity),
            "input_shape": list(input_shape),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
            "subject_count": len(subjects),
            "subjects": by_dataset,
            "truncated": [f"{ref.dataset_name}/{ref.subject_id}" for ref in truncated],
            "hypnograms": hypnograms,
            "ggir_sib_filter": ggir_sib_filter,
            "sib_demoted_epochs": sib_demoted,
            **dict(provenance or {}),
        },
    )

    _summarise(csv_path, subjects)
    print(f"\nResults: {output_dir}")
    return output_dir


def _summarise(csv_path: Path, subjects: Mapping[SubjectRef, ModelIOBundle]) -> None:
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

    print(f"\n{len(rows)} rows, {len(subjects)} subjects -> {csv_path}")
    for metric in sorted(by_metric):
        values = np.asarray(by_metric[metric])
        print(f"  {metric:<20} {values.mean():>10.4f}  (n={values.size})")

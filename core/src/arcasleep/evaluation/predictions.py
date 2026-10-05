"""Per-epoch predictions for scored subjects, in the ``predict`` CSV format.

Training and ``evaluate`` both score every test subject; this keeps what they
computed, one CSV per subject, so a later question about the scored nights is
answered from files rather than by loading the model again. The columns are
those of :meth:`arcasleep.predict.Prediction.to_csv` with the gap head, plus
``psg``: the PSG stage per epoch, empty where the epoch is not scored.

The probabilities are the model's own, before any ``--ggir-sib-filter``
demotion; ``stage`` carries the GGIR sleep filter and ``stage_raw`` the argmax,
as in ``predict``. The metrics CSV scores the argmax over the stages and the
gap head together, so an epoch with data whose gap head wins counts as a gap
there while ``stage_raw`` names its best stage; the ``gap_head`` column
reproduces either. The cache keeps no absolute clock, so ``start_s`` counts
from the start of the recording. The GGIR columns come from the cache's
``anglez_5s``; a cache built without it leaves them empty, and ``stage`` is
then the argmax.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
from pisces_lite.cv import SubjectRef
from pisces_lite.model_io import ModelIOBundle

from arcasleep.data.batching import pool_frames_to_epochs, split_gap_row
from arcasleep.predict import (
    EPOCH_SECONDS,
    STAGE_NAMES,
    Prediction,
    _apply_ggir_filter,
    _label_epochs,
    _summarize,
)

#: Subdirectory of a results or evaluation directory that holds the CSVs.
PREDICTIONS_DIRNAME = "predictions"


def scored_prediction(
    bundle: ModelIOBundle,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_classes: int,
    subject_id: str | None = None,
) -> Prediction:
    """One scored subject as a :class:`Prediction`, its PSG included.

    ``y_true`` and ``y_pred_proba`` are what
    :func:`arcasleep.training.driver.evaluate_subject` returned, padded to the
    scoring length; only the recording's own epochs are kept, and a recording
    longer than the scoring context keeps the epochs that were scored.
    """
    proba = np.asarray(y_pred_proba, dtype=np.float64)
    n_epochs = min(int(bundle.y.shape[1]), proba.shape[0])
    stage_names = STAGE_NAMES[:num_classes]

    stage_mass = proba[:n_epochs, :num_classes]
    probabilities = stage_mass / stage_mass.sum(axis=-1, keepdims=True)
    gap_head = proba[:n_epochs, num_classes] if proba.shape[1] > num_classes else None

    X = np.asarray(bundle.X[0])
    frames_per_epoch = X.shape[0] // int(bundle.y.shape[1])
    pooled = pool_frames_to_epochs(
        X[None, : n_epochs * frames_per_epoch], np.zeros((1, n_epochs))
    )[0]
    _, missing = split_gap_row(pooled)
    is_gap = missing >= 1.0

    stage = is_nap = is_sib = is_spt = None
    nap_minutes: float | None = None
    anglez = getattr(bundle, "anglez_5s", None)
    if anglez is not None:
        from arcasleep.ggir import evaluate_epoch_grid

        try:
            ggir = evaluate_epoch_grid(anglez[0])
            ggir = dataclasses.replace(
                ggir, is_sib=ggir.is_sib[:n_epochs], is_spt=ggir.is_spt[:n_epochs]
            )
            stage, is_nap = _apply_ggir_filter(probabilities, stage_names, is_gap, ggir)
            is_sib, is_spt = ggir.is_sib, ggir.is_spt
            nap_minutes = float(np.count_nonzero(is_nap)) * EPOCH_SECONDS / 60.0
        except Exception as exc:
            warnings.warn(
                f"{subject_id}: GGIR sleep filter failed ({type(exc).__name__}: {exc}); "
                "writing the raw model stages",
                RuntimeWarning,
                stacklevel=2,
            )
            stage = is_nap = is_sib = is_spt = nap_minutes = None

    truth = np.asarray(y_true)[:n_epochs]
    names = np.asarray(stage_names, dtype=object)
    psg = np.where(truth >= 0, names[np.clip(truth, 0, num_classes - 1)], "").astype(object)

    stage_raw = _label_epochs(probabilities, stage_names, is_gap)
    return Prediction(
        epoch_start_s=EPOCH_SECONDS * np.arange(n_epochs),
        probabilities=probabilities,
        stage_names=stage_names,
        is_gap=is_gap,
        gap_head=gap_head,
        stage=stage,
        stage_raw=stage_raw,
        is_nap=is_nap,
        summary=_summarize(
            stage if stage is not None else stage_raw, is_gap, nap_minutes, subject_id
        ),
        is_sib=is_sib,
        is_spt=is_spt,
        psg=psg,
    )


def prediction_path(directory: Path | str, ref: SubjectRef) -> Path:
    """``<directory>/<dataset>_<subject>.csv``."""
    return Path(directory) / f"{ref.dataset_name}_{ref.subject_id}.csv"


def write_scored_prediction(
    directory: Path | str,
    ref: SubjectRef,
    bundle: ModelIOBundle,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_classes: int,
) -> Path:
    """Build :func:`scored_prediction` and write it to :func:`prediction_path`."""
    prediction = scored_prediction(
        bundle, y_true, y_pred_proba, num_classes, subject_id=ref.subject_id
    )
    return prediction.to_csv(prediction_path(directory, ref))

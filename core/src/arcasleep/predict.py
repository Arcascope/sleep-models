"""Sleep stages from a raw accelerometer recording, using a released bundle.

This is the inference entry point a model card points at: it runs the bundle's
own ``preprocessing.json`` over timestamped triaxial samples, builds the
model input from the spectrogram frames exactly as training did, and scores
it with the bundle's weights. No PSG, labels or feature cache are
involved.

    from arcasleep.predict import predict

    result = predict("arcasleep-bidoze-pediatric", accel)   # accel: (N, 4) t [s], x, y, z [g]
    result.to_csv("night.csv")
"""
from __future__ import annotations

import csv
import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arcasleep.divisor import adult_probabilities, divisor_vector, hybrid_calls

#: Stage names in readout order; the network emits ``num_classes`` of them.
STAGE_NAMES = ("wake", "light", "deep", "rem")
#: What :attr:`Prediction.labels` and the CSV ``stage`` column hold for an
#: epoch with no accelerometer data.
GAP_LABEL_NAME = "gap"
#: The CSV column holding the network's gap probability, when requested.
GAP_HEAD_COLUMN = "gap_head"
EPOCH_SECONDS = 30.0


@dataclass(frozen=True)
class Summary:
    """One recording's headline durations, in minutes.

    ``tst_minutes`` counts the corrected sleep stages (everything but wake and
    gaps), naps included; with the filter off, the raw stages. ``nap_minutes``
    is ``None`` when the GGIR filter is off.
    """

    subject_id: str | None
    tst_minutes: float
    missing_data_minutes: float
    nap_minutes: float | None

    def to_csv(self, path: Path | str) -> Path:
        """One row: subject, total sleep time, missing data, nap time."""
        return write_summaries([self], path)


#: The file :func:`predict_files` collects every recording's summary row into.
SUMMARY_FILENAME = "sleep_summary.csv"


def write_summaries(summaries: Sequence[Summary], path: Path | str) -> Path:
    """One header, then one row per summary: subject, total sleep time, missing data, nap time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "tst_minutes", "missing_data_minutes", "nap_minutes"])
        for summary in summaries:
            writer.writerow(
                [
                    summary.subject_id or "",
                    f"{summary.tst_minutes:.1f}",
                    f"{summary.missing_data_minutes:.1f}",
                    "" if summary.nap_minutes is None else f"{summary.nap_minutes:.1f}",
                ]
            )
    return path


#: The file :func:`predict_files` writes the cohort-level TST statistics to.
MACRO_SUMMARY_FILENAME = "sleep_macro_summary.csv"
#: Percentiles reported by :func:`write_macro_summary`: p0, p10, ..., p100.
MACRO_PERCENTILES = tuple(10 * i for i in range(11))


def write_macro_summary(summaries: Sequence[Summary], path: Path | str) -> Path:
    """One row of ``tst_minutes`` statistics pooled over recordings.

    Columns: ``n``, ``mean``, ``stderr`` (sample standard deviation over
    ``sqrt(n)``, blank for a single recording), then ``p0`` to ``p100`` in
    steps of 10 (linear interpolation).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tst = np.array([summary.tst_minutes for summary in summaries], dtype=float)
    stderr = "" if tst.size < 2 else f"{tst.std(ddof=1) / math.sqrt(tst.size):.1f}"
    quantiles = np.percentile(tst, MACRO_PERCENTILES)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "mean", "stderr", *(f"p{q}" for q in MACRO_PERCENTILES)])
        writer.writerow([tst.size, f"{tst.mean():.1f}", stderr, *(f"{v:.1f}" for v in quantiles)])
    return path


@dataclass(frozen=True)
class Prediction:
    """One recording's per-epoch output.

    ``epoch_start_s[k]`` is on the input's clock: epoch ``k`` covers
    ``[epoch_start_s[k], epoch_start_s[k] + 30)``. ``is_gap`` marks epochs with
    no accelerometer data; their probabilities are the model's guess from
    context and should be left out of durations.

    ``probabilities`` always sum to 1 over the stages. ``gap_head`` is the
    network's own gap probability, present only when asked for; it is not
    renormalized with the stages, so a row's stages plus ``gap_head`` lie in
    ``[1, 2)``. ``is_gap``, not ``gap_head``, is what says an epoch has no data.

    ``stage`` is the corrected label (the model with the GGIR sleep filter
    applied); ``None`` means no correction was made, and it falls back to the
    raw argmax. ``stage_raw`` is always that raw argmax. ``is_nap`` flags
    corrected sleep outside every estimated main-sleep window; that sleep is
    kept, not removed. ``is_sib`` and ``is_spt`` are GGIR's per-epoch
    sustained-inactivity and sleep-period judgements, ``None`` without the
    filter.

    ``psg`` is the PSG stage name per epoch, ``""`` where the epoch is not
    scored. Only a recording scored against its PSG has one
    (:mod:`arcasleep.evaluation.predictions`); ``predict`` never sets it.
    """

    epoch_start_s: np.ndarray
    probabilities: np.ndarray
    stage_names: tuple[str, ...]
    is_gap: np.ndarray
    gap_head: np.ndarray | None = None
    stage: np.ndarray | None = None
    stage_raw: np.ndarray | None = None
    is_nap: np.ndarray | None = None
    summary: Summary | None = None
    is_sib: np.ndarray | None = None
    is_spt: np.ndarray | None = None
    psg: np.ndarray | None = None

    @property
    def stages(self) -> np.ndarray:
        """Index of the most probable stage per epoch, gaps included."""
        return self.probabilities.argmax(axis=-1)

    @property
    def raw_labels(self) -> np.ndarray:
        """Raw model stage name per epoch, ``"gap"`` where there was no data."""
        if self.stage_raw is not None:
            return np.asarray(self.stage_raw, dtype=object)
        names = np.asarray(self.stage_names, dtype=object)[self.stages]
        names[self.is_gap] = GAP_LABEL_NAME
        return names

    @property
    def labels(self) -> np.ndarray:
        """Corrected stage name per epoch, ``"gap"`` where there was no data."""
        if self.stage is not None:
            return np.asarray(self.stage, dtype=object)
        return self.raw_labels

    @property
    def nap_flags(self) -> np.ndarray:
        """Per-epoch nap flag; all zero when the filter made no judgement."""
        if self.is_nap is None:
            return np.zeros(self.epoch_start_s.shape[0], dtype=np.int8)
        return np.asarray(self.is_nap, dtype=np.int8)

    def to_csv(self, path: Path | str) -> Path:
        """One row per epoch: probabilities, corrected and raw stage, GGIR flags, gap.

        A :data:`GAP_HEAD_COLUMN` column follows the stage probabilities when requested.
        ``is_sib`` and ``is_spt`` are empty when the filter did not run. A
        ``psg`` column is last, and only when :attr:`psg` is set.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "epoch",
            "start_s",
            *self.stage_names,
            *([GAP_HEAD_COLUMN] if self.gap_head is not None else []),
            "stage",
            "stage_raw",
            "is_nap",
            "is_sib",
            "is_spt",
            "is_gap",
            *(["psg"] if self.psg is not None else []),
        ]
        n = self.epoch_start_s.shape[0]
        sib = _flag_text(self.is_sib, n)
        spt = _flag_text(self.is_spt, n)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            heads = self.probabilities
            if self.gap_head is not None:
                heads = np.column_stack([heads, self.gap_head])
            for k, (start, probs, label, raw, nap, gap) in enumerate(
                zip(
                    self.epoch_start_s,
                    heads,
                    self.labels,
                    self.raw_labels,
                    self.nap_flags,
                    self.is_gap,
                    strict=True,
                )
            ):
                writer.writerow(
                    [
                        k,
                        f"{start:.3f}",
                        *(f"{p:.4f}" for p in probs),
                        label,
                        raw,
                        int(nap),
                        sib[k],
                        spt[k],
                        int(gap),
                        *([self.psg[k]] if self.psg is not None else []),
                    ]
                )
        return path

    def plot(self, path: Path | str, title: str | None = None) -> Path:
        """Save a hypnogram PNG; see :func:`arcasleep.reporting.prediction_plot.plot_prediction`."""
        from arcasleep.reporting.prediction_plot import plot_prediction

        return plot_prediction(self, path, title=title)


def _flag_text(flags: np.ndarray | None, n: int) -> list[str]:
    if flags is None:
        return [""] * n
    return [str(int(bool(flag))) for flag in flags]


def read_prediction_csv(path: Path | str) -> Prediction:
    """Read a CSV written by :meth:`Prediction.to_csv` back into a ``Prediction``.

    Stage probability columns are whichever of the stage names the file holds;
    the summary is not stored in the file and comes back as ``None``.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no epochs")
    columns = rows[0].keys()
    required = {"start_s", "stage", "is_gap"}
    if not required <= set(columns):
        raise ValueError(
            f"{path} is not a prediction CSV; missing {sorted(required - set(columns))}"
        )
    stage_names = tuple(name for name in STAGE_NAMES if name in columns)

    def flags(name: str) -> np.ndarray | None:
        if name not in columns or any(row[name] == "" for row in rows):
            return None
        return np.array([row[name] == "1" for row in rows])

    stage = np.array([row["stage"] for row in rows], dtype=object)
    stage_raw = (
        np.array([row["stage_raw"] for row in rows], dtype=object)
        if "stage_raw" in columns
        else None
    )
    is_nap = flags("is_nap")
    gap_head = (
        np.array([float(row[GAP_HEAD_COLUMN]) for row in rows])
        if GAP_HEAD_COLUMN in columns
        else None
    )
    return Prediction(
        epoch_start_s=np.array([float(row["start_s"]) for row in rows]),
        probabilities=np.array([[float(row[name]) for name in stage_names] for row in rows]),
        stage_names=stage_names,
        is_gap=np.array([row["is_gap"] == "1" for row in rows]),
        gap_head=gap_head,
        stage=stage,
        stage_raw=stage_raw,
        is_nap=None if is_nap is None else is_nap.astype(np.int8),
        is_sib=flags("is_sib"),
        is_spt=flags("is_spt"),
        psg=np.array([row["psg"] for row in rows], dtype=object) if "psg" in columns else None,
    )


class Predictor:
    """A loaded bundle, reusable across recordings.

    A bundle with ``members`` in its ``config.json`` runs its members as one batched
    call and scores the mean of their outputs. With a ``stage_divisor`` there, the
    mean decides wake or sleep, and on its sleep epochs the stage is the argmax of the
    light, deep and REM probabilities divided by the divisor (:mod:`arcasleep.divisor`).
    """

    #: ``(wake, light, deep, REM)`` stage divisor of the bundle, ``None`` for the model's own argmax.
    divisor: np.ndarray | None = None

    def __init__(self, bundle_dir: Path | str):
        from arcasleep.artifacts.publish import bundle_family, load_bundle_model
        from arcasleep.models import get_family
        from arcasleep.processing import frames_per_label, processing_config

        bundle_dir = Path(bundle_dir)
        self.bundle_dir = bundle_dir
        self.model = load_bundle_model(bundle_dir)
        self.processing = processing_config(
            json.loads((bundle_dir / "preprocessing.json").read_text()), "preprocessing.json"
        )
        self.frames_per_epoch = frames_per_label(self.processing)
        self.context_epochs, self.epoch_multiple = get_family(
            bundle_family(bundle_dir)
        ).prediction_window(self.model)
        self.stage_names = STAGE_NAMES[: self.model.config.num_classes]
        self.min_sampling_hz = float(self.processing.fs)
        stage_divisor = json.loads((bundle_dir / "config.json").read_text()).get("stage_divisor")
        self.divisor = None if stage_divisor is None else divisor_vector(stage_divisor)

    def features(
        self, accel: np.ndarray, origin: float | str | None = None
    ) -> tuple[np.ndarray, float]:
        """``(frames, bins, channels)`` spectrogram over whole epochs, and the grid origin.

        The origin is in seconds on the input clock; frames run
        ``frames_per_epoch`` to an epoch from it.
        """
        from arcasleep.processing import clean_accelerometer_rows, median_sampling_rate

        accel = np.asarray(accel, dtype=np.float64)
        if accel.ndim != 2 or accel.shape[1] != 4:
            raise ValueError(f"accel must be (N, 4): t [s], x, y, z [g]; got {accel.shape}")
        accel = clean_accelerometer_rows(accel)
        fs = median_sampling_rate(accel[:, 0])
        if fs < self.min_sampling_hz:
            raise ValueError(
                f"recording is sampled at {fs:.2f} Hz; this model needs at least "
                f"{self.min_sampling_hz:g} Hz"
            )
        origin_s = _resolve_origin(origin, accel[0, 0])
        frames = self.processing.apply(accel, normalize=False, padding=False, origin=origin_s)
        n_epochs = frames.shape[0] // self.frames_per_epoch
        if n_epochs < 1:
            raise ValueError("recording is shorter than one 30-second epoch")
        return frames[: n_epochs * self.frames_per_epoch], origin_s

    def predict(
        self,
        accel: np.ndarray,
        origin: float | str | None = None,
        return_gap_head: bool = False,
        *,
        ggir_sleep_filter: bool = True,
        ggir_error_fails: bool = False,
        subject_id: str | None = None,
    ) -> Prediction:
        """Score one recording; see :func:`predict`."""
        n_stages = len(self.stage_names)
        if return_gap_head and self.model.config.output_classes <= n_stages:
            raise ValueError("return_gap_head=True, but this model has no gap head")
        from arcasleep.data.batching import (
            pad_recording_input,
            pool_frames_to_epochs,
            recording_input,
            split_gap_row,
        )
        from arcasleep.processing import clean_accelerometer_rows
        from arcasleep.ggir import evaluate as ggir_evaluate

        frames, origin_s = self.features(accel, origin)
        n_epochs = frames.shape[0] // self.frames_per_epoch
        pooled = pool_frames_to_epochs(frames[None], np.zeros((1, n_epochs)))[0]
        model_fpe = self.model.config.frames_per_epoch
        inputs = recording_input(frames, n_epochs, model_fpe)
        # A recording longer than a fixed context window is scored as
        # consecutive windows; the last one, or a whole recording scored in
        # one pass, is padded with absent epochs, as in training.
        L = self.context_epochs or -(-n_epochs // self.epoch_multiple) * self.epoch_multiple
        chunks = [
            pad_recording_input(inputs[s * model_fpe : (s + L) * model_fpe], L, model_fpe)
            for s in range(0, n_epochs, L)
        ]
        logits = self.model.infer(np.stack(chunks))
        logits = logits.reshape(-1, logits.shape[-1])[:n_epochs]
        probabilities, gap_head = _stage_and_gap_probabilities(logits, n_stages)
        _, missing = split_gap_row(pooled)
        epoch_start_s = origin_s + EPOCH_SECONDS * np.arange(n_epochs)
        is_gap = missing >= 1.0
        if self.divisor is None:
            stage_raw = _label_epochs(probabilities, self.stage_names, is_gap)
        else:
            names = np.asarray(self.stage_names, dtype=object)
            stage_raw = names[hybrid_calls(probabilities, probabilities, self.divisor)]
            stage_raw[is_gap] = GAP_LABEL_NAME
            probabilities = adult_probabilities(probabilities, self.divisor)
        return finish_prediction(
            accel,
            origin_s,
            epoch_start_s,
            probabilities,
            self.stage_names,
            is_gap,
            stage_raw,
            gap_head=gap_head if return_gap_head else None,
            ggir_sleep_filter=ggir_sleep_filter,
            ggir_error_fails=ggir_error_fails,
            subject_id=subject_id,
        )


def finish_prediction(
    accel: np.ndarray,
    origin_s: float,
    epoch_start_s: np.ndarray,
    probabilities: np.ndarray,
    stage_names: tuple[str, ...],
    is_gap: np.ndarray,
    stage_raw: np.ndarray,
    *,
    gap_head: np.ndarray | None = None,
    ggir_sleep_filter: bool = True,
    ggir_error_fails: bool = False,
    subject_id: str | None = None,
) -> Prediction:
    """A :class:`Prediction` from per-epoch model output and its raw stage labels.

    Applies the GGIR sleep filter (see :func:`predict`) to ``stage_raw`` and
    summarizes the result.
    """
    from arcasleep.data.prepare import clean_accelerometer_rows
    from arcasleep.ggir import evaluate as ggir_evaluate

    stage = None
    is_nap = None
    ggir = None
    nap_minutes: float | None = None
    if ggir_sleep_filter:
        try:
            # The same rows the model scored: cleaned, and none before the origin.
            clean = clean_accelerometer_rows(np.asarray(accel, dtype=np.float64))
            clean = clean[clean[:, 0] >= origin_s]
            ggir = ggir_evaluate(clean, epoch_start_s, epoch_start_s.shape[0])
            stage, is_nap = _filter_labels(stage_raw, is_gap, ggir)
            nap_minutes = float(np.count_nonzero(is_nap)) * EPOCH_SECONDS / 60.0
        except Exception as exc:
            if ggir_error_fails:
                raise
            warnings.warn(
                f"GGIR sleep filter failed ({type(exc).__name__}: {exc}); "
                "returning the raw model stages",
                RuntimeWarning,
                stacklevel=3,
            )
            stage = is_nap = ggir = nap_minutes = None

    summary = _summarize(stage if stage is not None else stage_raw, is_gap, nap_minutes, subject_id)
    return Prediction(
        epoch_start_s=epoch_start_s,
        probabilities=probabilities,
        stage_names=stage_names,
        is_gap=is_gap,
        gap_head=gap_head,
        stage=stage,
        stage_raw=stage_raw,
        is_nap=is_nap,
        summary=summary,
        is_sib=None if ggir is None else ggir.is_sib,
        is_spt=None if ggir is None else ggir.is_spt,
    )


def load_predictor(bundle_dir: Path | str) -> Predictor:
    """A :class:`Predictor` for a bundle."""
    return Predictor(bundle_dir)


def predict(
    bundle_dir: Path | str,
    accel: np.ndarray,
    origin: float | str | None = None,
    return_gap_head: bool = False,
    *,
    ggir_sleep_filter: bool = True,
    ggir_error_fails: bool = False,
    subject_id: str | None = None,
) -> Prediction:
    """Per-epoch sleep stages for one recording.

    ``accel`` is ``(N, 4)``: timestamp in seconds, then x, y, z in g, with
    missing stretches simply absent (never zero-filled). ``origin`` is where
    epoch 0 starts, on the timestamps' clock: ``None`` for the first sample, a
    number (a PSG or lights-off start, say) to line epochs up with it, or
    ``"unix"`` for whole 30-second boundaries since the Unix epoch. Samples
    before the origin are not used.

    Stage probabilities are always renormalized over the stages alone.
    ``return_gap_head`` adds the network's gap probability as a separate
    ``gap_head`` output; it needs a model trained with a gap head.

    With ``ggir_sleep_filter`` (the default) the corrected ``stage`` column has
    non-SIB quiet wakefulness outside the main sleep window demoted to wake,
    and sleep outside that window flagged as naps but kept; inside the window
    sleep is never flipped to wake, and a recording <= 9 h is all one main
    sleep window, so the SIB judgement changes nothing for it. Pass ``False``
    for the raw model argmax. If the filter itself raises, the raw prediction
    is returned with the filter columns empty and a warning;
    ``ggir_error_fails=True`` re-raises instead.

    Loads the bundle on every call; use :class:`Predictor` for many recordings.
    """
    return load_predictor(bundle_dir).predict(
        accel,
        origin,
        return_gap_head=return_gap_head,
        ggir_sleep_filter=ggir_sleep_filter,
        ggir_error_fails=ggir_error_fails,
        subject_id=subject_id,
    )


#: Seconds per tick of each timestamp unit :func:`read_accelerometer` accepts.
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
#: Unix timestamps at or above these are taken as microseconds or milliseconds;
#: 1e11 s is the year 5138, while 1e11 ms is 1973 and 1e14 us is 1973.
_US_THRESHOLD = 1e14
_MS_THRESHOLD = 1e11


def read_accelerometer(path: Path | str, time_unit: str | None = None) -> np.ndarray:
    """Read ``t, x, y, z`` columns from a comma- or whitespace-separated file.

    A single header row is skipped. Extra columns after the fourth are ignored.
    The returned timestamps are in seconds. ``time_unit`` is the file's unit,
    ``"s"``, ``"ms"`` or ``"us"``; with ``None`` it is detected:

    - a first header column containing ``ms`` means milliseconds;
    - otherwise a first timestamp ≥ 1e14 means microseconds, and ≥ 1e11
      milliseconds, as Unix times in seconds are far below both;
    - otherwise seconds.
    """
    if time_unit is not None and time_unit not in TIME_UNITS:
        raise ValueError(f"time_unit must be one of {sorted(TIME_UNITS)}; got {time_unit!r}")
    path = Path(path)
    with path.open() as handle:
        first = handle.readline()
    delimiter = "," if "," in first else None
    header = not _is_numeric_row(first, delimiter)
    accel = np.loadtxt(
        path, delimiter=delimiter, skiprows=int(header), usecols=range(4), ndmin=2
    )
    if time_unit is None:
        time_unit = _detect_time_unit(first.split(delimiter)[0] if header else None, accel)
    accel[:, 0] *= TIME_UNITS[time_unit]
    return accel


def _detect_time_unit(time_column: str | None, accel: np.ndarray) -> str:
    """The timestamp unit from the time column's header, else the timestamps' magnitude."""
    if time_column is not None and "ms" in time_column.strip().lower():
        return "ms"
    first_t = abs(float(accel[0, 0])) if accel.shape[0] else 0.0
    if first_t >= _US_THRESHOLD:
        return "us"
    if first_t >= _MS_THRESHOLD:
        return "ms"
    return "s"


def _is_numeric_row(line: str, delimiter: str | None) -> bool:
    try:
        [float(value) for value in line.split(delimiter)[:4]]
    except ValueError:
        return False
    return True


def _label_epochs(
    probabilities: np.ndarray, stage_names: tuple[str, ...], is_gap: np.ndarray
) -> np.ndarray:
    """Argmax stage name per epoch, overwriting gaps with ``"gap"``."""
    names = np.asarray(stage_names, dtype=object)[probabilities.argmax(axis=-1)]
    names = np.array(names, dtype=object)
    names[is_gap] = GAP_LABEL_NAME
    return names


def _apply_ggir_filter(
    probabilities: np.ndarray,
    stage_names: tuple[str, ...],
    is_gap: np.ndarray,
    ggir,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrected labels and nap flags from the GGIR SIB and SPT judgement.

    Outside the main sleep window predicted sleep that is not a sustained
    inactivity bout is demoted to wake. Inside the window SIB never flips
    sleep to wake: the SPT window is a sleep period, so sleep the model found
    there stands even when it is not a SIB. A sub-recording of at most
    9 h has no estimated SPT, so all of it is treated as the main
    sleep window and no sleep inside it is demoted -- for a <= 9 h recording
    the SIB judgement changes nothing. Remaining sleep outside every main sleep
    window is flagged as a nap and kept. Sleep is never promoted, and gap
    epochs are neither sleep nor naps.
    """
    return _filter_labels(_label_epochs(probabilities, stage_names, is_gap), is_gap, ggir)


def _filter_labels(
    stage_raw: np.ndarray, is_gap: np.ndarray, ggir
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_apply_ggir_filter` on stage labels already decided."""
    stage = np.array(stage_raw, dtype=object)
    sleep = ~np.isin(stage, ["wake", GAP_LABEL_NAME])
    demote = sleep & ~ggir.is_sib & ~ggir.is_spt
    stage[demote] = "wake"
    sleep = sleep & ~demote
    is_nap = np.zeros(stage.shape[0], dtype=np.int8)
    is_nap[sleep & ~ggir.is_spt] = 1
    stage[is_gap] = GAP_LABEL_NAME
    return stage, is_nap


def _summarize(
    stage: np.ndarray,
    is_gap: np.ndarray,
    nap_minutes: float | None,
    subject_id: str | None,
) -> Summary:
    """Headline durations from the (corrected) labels and the gap flags."""
    sleep = (stage != "wake") & (stage != GAP_LABEL_NAME)
    return Summary(
        subject_id=subject_id,
        tst_minutes=float(np.count_nonzero(sleep)) * EPOCH_SECONDS / 60.0,
        missing_data_minutes=float(np.count_nonzero(is_gap)) * EPOCH_SECONDS / 60.0,
        nap_minutes=nap_minutes,
    )


def _resolve_origin(origin: float | str | None, first_sample_s: float) -> float:
    if origin is None:
        return float(first_sample_s)
    if origin == "unix":
        # Whole epochs since the Unix epoch, so the grid is also whole steps.
        return math.floor(first_sample_s / EPOCH_SECONDS) * EPOCH_SECONDS
    if isinstance(origin, str):
        raise ValueError(f"origin must be None, 'unix' or seconds; got {origin!r}")
    return float(origin)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _stage_and_gap_probabilities(
    logits: np.ndarray, n_stages: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(stage probabilities, gap-head probability or None)`` from raw logits.

    The stages are a softmax over the stage logits alone, which is the full
    softmax's ``p[:n] / p[:n].sum()``. The gap head, when the network has one,
    is its probability under the full softmax.
    """
    stages = _softmax(logits[:, :n_stages])
    if logits.shape[-1] <= n_stages:
        return stages, None
    return stages, _softmax(logits)[:, n_stages]


def predict_files(
    bundle_dir: Path | str,
    inputs: Sequence[Path | str],
    output_dir: Path | str,
    origin: float | str | None = None,
    return_gap_head: bool = False,
    *,
    ggir_sleep_filter: bool = True,
    ggir_error_fails: bool = False,
    hypnogram: bool = False,
    time_unit: str | None = None,
) -> list[Path]:
    """Write ``<output_dir>/<input stem>.csv`` per input, one CSV containing per-subject 
    summary metrics, and one CSV that has macro statistics for all iputs.

    The summary has a row per input, in input order; the subject id
    recorded in it is the input file's stem. The macro summary pools
    their ``tst_minutes`` into a mean, standard error and deciles. With ``hypnogram``, a
    ``<stem>.png`` hypnogram is saved beside each prediction. ``time_unit`` is
    the inputs' timestamp unit, detected per file when ``None``; see
    :func:`read_accelerometer`.
    """
    predictor = load_predictor(bundle_dir)
    output_dir = Path(output_dir)
    written = []
    summaries = []
    for path in map(Path, inputs):
        result = predictor.predict(
            read_accelerometer(path, time_unit),
            origin,
            return_gap_head=return_gap_head,
            ggir_sleep_filter=ggir_sleep_filter,
            ggir_error_fails=ggir_error_fails,
            subject_id=path.stem,
        )
        written.append(result.to_csv(output_dir / f"{path.stem}.csv"))
        if result.summary is not None:
            summaries.append(result.summary)
        if hypnogram:
            written.append(result.plot(output_dir / f"{path.stem}.png", title=path.stem))
    if summaries:
        written.append(write_summaries(summaries, output_dir / SUMMARY_FILENAME))
        written.append(write_macro_summary(summaries, output_dir / MACRO_SUMMARY_FILENAME))
    return written

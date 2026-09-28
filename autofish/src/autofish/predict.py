"""Sleep stages from a raw accelerometer recording, using a released bundle.

This is the inference entry point a model card points at: it runs the bundle's
own ``preprocessing.json`` over timestamped triaxial samples, pools the
sub-epoch spectrogram frames to 30-second epochs exactly as training did, and
scores them with the bundle's weights. No PSG, labels or feature cache are
involved.

    from autofish.predict import predict

    result = predict("autofish-pediatric", accel)   # accel: (N, 4) t [s], x, y, z [g]
    result.to_csv("night.csv")
"""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Stage names in readout order; the network emits ``num_classes`` of them.
STAGE_NAMES = ("wake", "light", "deep", "rem")
#: What :attr:`Prediction.labels` and the CSV ``stage`` column hold for an
#: epoch with no accelerometer data.
GAP_LABEL_NAME = "gap"
EPOCH_SECONDS = 30.0


@dataclass(frozen=True)
class Prediction:
    """One recording's per-epoch output.

    ``epoch_start_s[k]`` is on the input's clock: epoch ``k`` covers
    ``[epoch_start_s[k], epoch_start_s[k] + 30)``. ``is_gap`` marks epochs with
    no accelerometer data; their probabilities are the model's guess from
    context and should be left out of durations.
    """

    epoch_start_s: np.ndarray
    probabilities: np.ndarray
    stage_names: tuple[str, ...]
    is_gap: np.ndarray

    @property
    def stages(self) -> np.ndarray:
        """Index of the most probable stage per epoch, gaps included."""
        return self.probabilities.argmax(axis=-1)

    @property
    def labels(self) -> np.ndarray:
        """Stage name per epoch, ``"gap"`` where there was no data."""
        names = np.asarray(self.stage_names, dtype=object)[self.stages]
        names[self.is_gap] = GAP_LABEL_NAME
        return names

    def to_csv(self, path: Path | str) -> Path:
        """One row per epoch: start time, stage probabilities, stage, is_gap."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "start_s", *self.stage_names, "stage", "is_gap"])
            for k, (start, probs, label, gap) in enumerate(
                zip(self.epoch_start_s, self.probabilities, self.labels, self.is_gap, strict=True)
            ):
                writer.writerow(
                    [k, f"{start:.3f}", *(f"{p:.4f}" for p in probs), label, int(gap)]
                )
        return path


class Predictor:
    """A loaded bundle, reusable across recordings."""

    def __init__(self, bundle_dir: Path | str):
        from pisces_lite.proc import ProcessingConfig

        from autofish.artifacts.publish import load_bundle_model
        from autofish.data.prepare import frames_per_label

        bundle_dir = Path(bundle_dir)
        self.bundle_dir = bundle_dir
        self.model = load_bundle_model(bundle_dir)
        self.processing = ProcessingConfig.from_dict(
            json.loads((bundle_dir / "preprocessing.json").read_text())
        )
        self.frames_per_epoch = frames_per_label(self.processing)
        provenance = json.loads((bundle_dir / "provenance.json").read_text())
        self.context_epochs = int(provenance["input_shape"][0])
        self.stage_names = STAGE_NAMES[: self.model.config.num_classes]
        self.min_sampling_hz = float(self.processing.fs)

    def features(
        self, accel: np.ndarray, origin: float | str | None = None
    ) -> tuple[np.ndarray, float]:
        """``(epochs, bins + 1, channels)`` model input and the grid origin in seconds.

        The last row along the bin axis is the fraction of the epoch with no
        accelerometer data.
        """
        from autofish.data.batching import pool_frames_to_epochs
        from autofish.data.prepare import clean_accelerometer_rows, median_sampling_rate

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
        frames = frames[: n_epochs * self.frames_per_epoch]
        pooled = pool_frames_to_epochs(frames[None], np.zeros((1, n_epochs)))[0]
        return pooled, origin_s

    def predict(self, accel: np.ndarray, origin: float | str | None = None) -> Prediction:
        """Score one recording; see :func:`predict`."""
        from autofish.data.batching import pad_pooled, split_gap_row

        pooled, origin_s = self.features(accel, origin)
        n_epochs = pooled.shape[0]
        # A recording longer than the context window is scored as consecutive
        # windows; the last one is padded with absent epochs, as in training.
        L = self.context_epochs
        chunks = [pad_pooled(pooled[s : s + L], L) for s in range(0, n_epochs, L)]
        logits = self.model.infer(np.stack(chunks))
        logits = logits.reshape(-1, logits.shape[-1])[:n_epochs, : len(self.stage_names)]
        probabilities = _softmax(logits)
        _, missing = split_gap_row(pooled)
        return Prediction(
            epoch_start_s=origin_s + EPOCH_SECONDS * np.arange(n_epochs),
            probabilities=probabilities,
            stage_names=self.stage_names,
            is_gap=missing >= 1.0,
        )


def predict(
    bundle_dir: Path | str, accel: np.ndarray, origin: float | str | None = None
) -> Prediction:
    """Per-epoch sleep stages for one recording.

    ``accel`` is ``(N, 4)``: timestamp in seconds, then x, y, z in g, with
    missing stretches simply absent (never zero-filled). ``origin`` is where
    epoch 0 starts, on the timestamps' clock: ``None`` for the first sample, a
    number (a PSG or lights-off start, say) to line epochs up with it, or
    ``"unix"`` for whole 30-second boundaries since the Unix epoch. Samples
    before the origin are not used.

    Loads the bundle on every call; use :class:`Predictor` for many recordings.
    """
    return Predictor(bundle_dir).predict(accel, origin)


def read_accelerometer(path: Path | str) -> np.ndarray:
    """Read ``t, x, y, z`` columns from a comma- or whitespace-separated file.

    A single header row is skipped. Extra columns after the fourth are ignored.
    """
    path = Path(path)
    with path.open() as handle:
        first = handle.readline()
    delimiter = "," if "," in first else None
    header = not _is_numeric_row(first, delimiter)
    return np.loadtxt(path, delimiter=delimiter, skiprows=int(header), usecols=range(4), ndmin=2)


def _is_numeric_row(line: str, delimiter: str | None) -> bool:
    try:
        [float(value) for value in line.split(delimiter)[:4]]
    except ValueError:
        return False
    return True


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


def predict_files(
    bundle_dir: Path | str,
    inputs: Sequence[Path | str],
    output_dir: Path | str,
    origin: float | str | None = None,
) -> list[Path]:
    """Write ``<output_dir>/<input stem>.csv`` for each input file."""
    predictor = Predictor(bundle_dir)
    written = []
    for path in map(Path, inputs):
        result = predictor.predict(read_accelerometer(path), origin)
        written.append(result.to_csv(Path(output_dir) / f"{path.stem}.csv"))
    return written

"""From raw accelerometer rows to spectrogram frames.

The feature cache (:mod:`arcasleep.data.prepare`) and prediction
(:mod:`arcasleep.predict`) both read processing settings and clean raw rows
through this module, so a recording scored by a bundle goes through the same
steps as the recordings the bundle was trained on.
"""
from __future__ import annotations

import warnings

import numpy as np
from pisces_lite.datasets.constants import PSG_DT
from pisces_lite.proc import ProcessingConfig

#: The one key in ``inputs.json`` and ``preprocessing.json`` that is not a
#: ``ProcessingConfig`` field: the feature-cache directory name.
CACHE_PREFIX_KEY = "cache_prefix"

#: Largest acceleration, in g on any axis, kept by :func:`clean_accelerometer_rows`;
#: the full-scale range of common wrist accelerometers.
MAX_ABS_G = 16.0

#: Largest share of finite rows :func:`clean_accelerometer_rows` drops for range before it
#: refuses the recording; a larger share usually means the input is not in g.
MAX_OUT_OF_RANGE_SHARE = 0.001


class FeatureExtractionError(RuntimeError):
    """A subject could not be turned into features, with the reason attached."""


def processing_config(data: dict, what: str) -> ProcessingConfig:
    """``data`` as a ``ProcessingConfig``; refuses settings the installed pisces-lite does not know.

    ``ProcessingConfig.from_dict`` drops keys it does not declare, so an older
    pisces-lite would otherwise compute different spectra from the ones ``data``
    describes. ``cache_prefix`` is the one other key allowed.
    """
    kind = str(data.get("type", "")).lower()
    if kind != "nufft":
        raise ValueError(f"{what}: only type='nufft' is supported, got {kind!r}")
    unknown = set(data) - set(ProcessingConfig.__dataclass_fields__) - {CACHE_PREFIX_KEY}
    if unknown:
        raise ValueError(
            f"{what} sets {sorted(unknown)}, which the installed pisces-lite does not "
            "support; upgrade pisces-lite to the version in arcasleep's requirements"
        )
    return ProcessingConfig.from_dict(data)


def clean_accelerometer_rows(accel: np.ndarray) -> np.ndarray:
    """Drop invalid rows and enforce strictly increasing timestamps.

    A row is invalid when any value is non-finite or any axis is beyond
    ±``MAX_ABS_G`` g; such samples are sensor or export errors, not motion.
    Rows dropped for range are reported with a ``RuntimeWarning``. More than
    ``MAX_OUT_OF_RANGE_SHARE`` of the finite rows out of range raises
    :class:`FeatureExtractionError`.
    """
    acc = np.asarray(accel, dtype=np.float64)
    acc = acc[np.all(np.isfinite(acc), axis=1)]
    in_range = np.all(np.abs(acc[:, 1:]) <= MAX_ABS_G, axis=1)
    dropped = int(acc.shape[0] - np.count_nonzero(in_range))
    if dropped:
        share = dropped / acc.shape[0]
        if share > MAX_OUT_OF_RANGE_SHARE:
            raise FeatureExtractionError(
                f"{dropped} of {acc.shape[0]} accelerometer rows ({share:.2%}) have an axis "
                f"beyond ±{MAX_ABS_G:g} g; acceleration must be in g (divide m/s² by 9.81)"
            )
        warnings.warn(
            f"dropped {dropped} accelerometer rows with an axis beyond ±{MAX_ABS_G:g} g",
            RuntimeWarning,
            stacklevel=2,
        )
    acc = acc[in_range]
    if acc.shape[0] < 2:
        raise FeatureExtractionError("need at least two valid accelerometer rows")
    acc = acc[np.argsort(acc[:, 0], kind="mergesort")]
    acc = acc[np.concatenate([[True], np.diff(acc[:, 0]) > 0])]
    if acc.shape[0] < 2:
        raise FeatureExtractionError(
            "need at least two accelerometer rows with distinct timestamps"
        )
    return acc


def median_sampling_rate(timestamps: np.ndarray) -> float:
    """Median rate over positive steps, as senpy measures it for its window grid."""
    steps = np.diff(np.asarray(timestamps, dtype=np.float64))
    steps = np.sort(steps[steps > 0.0])
    return 1.0 / float(steps[steps.size // 2]) if steps.size else 0.0


def frames_per_label(processing_config: ProcessingConfig) -> int:
    """Feature frames per PSG epoch: ``PSG_DT / window_step_seconds``.

    15 for the committed configs' 2 s hop. The hop must divide the PSG epoch
    evenly, or X and y would drift apart over the night.
    """
    step = float(processing_config.window_step_seconds)
    ratio = float(PSG_DT) / step
    if abs(ratio - round(ratio)) > 1e-9 or ratio < 1.0:
        raise ValueError(
            f"window_step_seconds={step} must evenly divide the {PSG_DT}s PSG epoch"
        )
    return int(round(ratio))

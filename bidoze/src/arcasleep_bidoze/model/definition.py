from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, List, Sequence, Tuple, Type

from arcasleep_bidoze.model.constants import (
    DEFAULT_FEEDBACK_FOCAL_ALPHA,
    GAP_LABEL,
    SPECTROGRAM_PADDING_VALUE,
    STAGE_1_MAP,
    STAGE_2_MAP,
    STAGE_3_MAP,
)
# Keep XLA/absl compiler noise out of normal training runs unless the caller overrides it.
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import flax.linen as nn
from flax import traverse_util
from flax.core import FrozenDict, freeze, unfreeze
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion
from arcasleep.training.balanced import (
    CLASS_SAMPLE_MAX_ATTEMPTS,
    LR_DROP_FACTOR,
    SEGMENT_PLACEMENTS,
    crop_rows,
    placement_offsets,
    roll_rows,
    Patience,
    Plateau,
    balanced_cross_entropy,
    place_rows,
    sample_balanced_segments,
    selection_confusion,
)



class CVModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

@dataclass
class ModelConfig:  # type: ignore[no-redef]
    num_classes: int

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


LAYER_NORM_EPS = 1e-3
GLOROT_UNIFORM = nn.initializers.glorot_uniform()
ZEROS = nn.initializers.zeros_init()
TRAINABLE_SCOPES = {"all", "gate_head"}
# Fraction of validation records scored each epoch, resampled uniformly without
# replacement. A test-sized validation set costs about as much per full pass as
# the training step does; 0.2 keeps the monitor's cost near noise. The draw uses
# its own RNG (seed + this offset) so it cannot perturb the fitted weights.
VAL_SUBSAMPLE_FRACTION = 0.2
VAL_SUBSAMPLE_SEED_OFFSET = 90210

GATE_HEAD_TRAINABLE_ROOTS = {
    "gated_fusion",
    "concat_fusion",
    "head_hidden",
    "head_logits",
}


@dataclass
class BiDozeNetConfig(ModelConfig):
    # With this `True` the net learns to predict sections that have
    # no accelerometer samples to a 5th class.
    score_gap: bool = True
    # Feature frames per 30 s PSG epoch (15 for the committed configs' 2 s
    # hop); the model pools them down to one token per epoch before the
    # conv/transformer stack. With 1 the pool is the identity.
    frames_per_epoch: int = 1
    d_model: int = 256
    n_heads: int = 8
    n_transformer_layers: int = 4
    use_bicausal: bool = True
    d_ff: int = 512
    conv_kernels: Tuple[int, ...] = (3, 7, 31, 63)
    dropout: float = 0.2
    focal_alpha: List[float] | None = None
    focal_gamma: float = 2.0
    attn_dropout: float = 0.0
    temporal_slide: bool = True
    stage_epoch_fractions: Tuple[float, float, float] = (0.10, 0.10, 0.80)
    stage_focal_alphas: Tuple[Tuple[float, ...] | None, ...] = (
        None,
        (1.0, 0.5, 1.5, 1.0),
        (0.3, 0.8, 2.0, 2.0, 1.0),
    )
    stage_focal_gammas: Tuple[float, float, float] = (2.0, 2.0, 2.0)

    @property
    def output_classes(self) -> int:
        return self.num_classes + int(self.score_gap)

    @staticmethod
    def _normalize_stage_focal_alphas(
        stage_focal_alphas: Iterable[Iterable[float] | None],
    ) -> Tuple[Tuple[float, ...] | None, ...]:
        normalized: list[Tuple[float, ...] | None] = []
        for stage_alpha in stage_focal_alphas:
            if stage_alpha is None:
                normalized.append(None)
            else:
                normalized.append(tuple(float(value) for value in stage_alpha))
        return tuple(normalized)

    @staticmethod
    def _normalize_stage_focal_gammas(stage_focal_gammas: Iterable[float]) -> Tuple[float, ...]:
        return tuple(float(value) for value in stage_focal_gammas)

    @classmethod
    def from_dict(cls, data: dict) -> "BiDozeNetConfig":
        fields = {f for f in cls.__dataclass_fields__}
        config_data = {k: v for k, v in data.items() if k in fields}

        stage_focal_alphas_supplied = "stage_focal_alphas" in config_data
        stage_focal_gammas_supplied = "stage_focal_gammas" in config_data

        if stage_focal_alphas_supplied:
            config_data["stage_focal_alphas"] = cls._normalize_stage_focal_alphas(config_data["stage_focal_alphas"])
        if stage_focal_gammas_supplied:
            config_data["stage_focal_gammas"] = cls._normalize_stage_focal_gammas(config_data["stage_focal_gammas"])

        config = cls(**config_data)

        if config.num_classes not in (2, 3, 4):
            raise ValueError(
                "BiDozeNetConfig.num_classes is the sleep-stage interface size; "
                "expected 2, 3, or 4"
            )

        if config.frames_per_epoch < 1:
            raise ValueError("BiDozeNetConfig.frames_per_epoch must be at least 1")

        if config.focal_alpha is not None:
            config.focal_alpha = [float(value) for value in config.focal_alpha]

        if not stage_focal_alphas_supplied and config.focal_alpha is not None:
            stage_focal_alphas = list(config.stage_focal_alphas)
            stage_focal_alphas[-1] = tuple(config.focal_alpha)
            config.stage_focal_alphas = tuple(stage_focal_alphas)

        if not stage_focal_gammas_supplied:
            stage_focal_gammas = list(config.stage_focal_gammas)
            stage_focal_gammas[-1] = float(config.focal_gamma)
            config.stage_focal_gammas = tuple(stage_focal_gammas)

        if len(config.stage_epoch_fractions) != 3 or len(config.stage_focal_alphas) != 3:
            raise ValueError("BiDoze uses exactly three curriculum stages")

        return config

    def make_model(self) -> "BiDozeNetModel":
        return BiDozeNetModel(config=self)


def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.gelu(x, approximate=False)


def _resolve_seed(seed: int | None, default: int = 0) -> int:
    return default if seed is None else int(seed)


def _alibi_slopes(n_heads: int) -> np.ndarray:
    def _get(n: int) -> list[float]:
        if math.log2(n).is_integer():
            return [2.0 ** (-(2.0 ** -(math.log2(n) - 3)) * i) for i in range(1, n + 1)]
        closest = 2 ** math.floor(math.log2(n))
        base = _get(closest)
        extra = _get(2 * closest)
        return base + extra[0 : 2 * (n - closest) : 2]

    return np.array(_get(n_heads), dtype=np.float32)


def _causal_mask(length: int) -> jnp.ndarray:
    return jnp.triu(jnp.ones((length, length), dtype=bool), k=1)


def _reverse_causal_mask(length: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((length, length), dtype=bool), k=-1)


def _normalize_valid_mask(x: jnp.ndarray, valid_mask: jnp.ndarray | None) -> jnp.ndarray:
    if valid_mask is None:
        return jnp.ones(x.shape[:2], dtype=x.dtype)
    mask = jnp.asarray(valid_mask, dtype=x.dtype)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = jnp.squeeze(mask, axis=-1)
    return mask


def _pool_frames_to_epochs(x: jnp.ndarray, frames_per_epoch: int) -> jnp.ndarray:
    """Mean-pool raw sub-epoch feature frames down to one token per 30 s PSG epoch.

    Sub-epoch caches (e.g. the 12 Hz log_psd config, window_step_seconds=2) carry
    ``frames_per_epoch`` rows per label instead of 1. The host pools those
    already (``arcasleep.data.batching.pool_frames_to_epochs``, which drops
    missing frames and appends the gap row), so this in-model pool only runs
    for a config that sets ``frames_per_epoch > 1`` and feeds raw frames; it is
    a plain mean and does not treat SPECTROGRAM_PADDING_VALUE specially.
    """
    n_frames = x.shape[1]
    if n_frames % frames_per_epoch != 0:
        raise ValueError(
            f"frames_per_epoch={frames_per_epoch} does not evenly divide the T "
            f"axis (length {n_frames})"
        )
    n_epochs = n_frames // frames_per_epoch
    pooled = x.reshape(x.shape[0], n_epochs, frames_per_epoch, *x.shape[2:])
    return pooled.mean(axis=2)


def _flatten_timestep_features(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim < 3:
        raise ValueError(f"Expected input shape (B, T, ...), got {x.shape}")
    if x.ndim == 3:
        return x
    return x.reshape((*x.shape[:2], int(np.prod(x.shape[2:]))))


def _numpy_feature_activity_mask(input_x: np.ndarray) -> np.ndarray:
    feature_axes = tuple(range(2, input_x.ndim))
    return (np.abs(input_x).sum(axis=feature_axes) > 0).astype(np.float32)


def _class_lookup(label: int, stage_map: dict[int, list[int]]) -> int:
    for group_idx, members in stage_map.items():
        if label in members:
            return group_idx
    return label


def normalize_gap_labels(labels: np.ndarray) -> np.ndarray:
    """Canonicalize all gap/padding-style labels to GAP_LABEL."""
    return np.where(labels <= GAP_LABEL, GAP_LABEL, labels)


def normalize_gap_labels_jnp(labels: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(labels <= GAP_LABEL, GAP_LABEL, labels)


def remap_classes(labels: np.ndarray, stage_map: dict[int, list[int]]) -> np.ndarray:
    remapped = normalize_gap_labels(np.array(labels, copy=True))
    valid = (remapped >= 0) | (remapped == GAP_LABEL)
    vectorized = np.vectorize(lambda value: _class_lookup(int(value), stage_map))
    remapped[valid] = vectorized(remapped[valid])
    return remapped


def gap_group_index(stage_map: dict[int, list[int]]) -> int | None:
    """The stage map's gap group, or ``None`` when the map has no such class.

    A map without it (``score_gap=false``) has no output to put a gap
    epoch in, so those epochs are dropped from the loss rather than folded into a
    sleep stage -- the alternative, aliasing them onto the last logit, would train
    the model to call every gap epoch REM.
    """
    return next(
        (group_idx for group_idx, members in stage_map.items() if GAP_LABEL in members),
        None,
    )


def stage_map_without_gap(stage_map: dict[int, list[int]]) -> dict[int, list[int]]:
    """The same map minus its gap group (which is always the last index)."""
    return {
        group_idx: members
        for group_idx, members in stage_map.items()
        if GAP_LABEL not in members
    }


def grouped_logits(logits: jnp.ndarray, stage_map: dict[int, list[int]]) -> jnp.ndarray:
    grouped = []
    for group_idx in range(len(stage_map)):
        member_idx = jnp.asarray(
            [
                logits.shape[-1] - 1 if member == GAP_LABEL else member
                for member in stage_map[group_idx]
            ],
            dtype=jnp.int32,
        )
        grouped.append(jax.scipy.special.logsumexp(logits[..., member_idx], axis=-1))
    return jnp.stack(grouped, axis=-1)


def _focal_alpha_for_group_count(
    focal_alpha: tuple[float, ...] | None,
    group_count: int,
    dtype,
) -> jnp.ndarray:
    if focal_alpha is None:
        return jnp.ones((group_count,), dtype=dtype)
    values = tuple(float(value) for value in focal_alpha)
    if len(values) < group_count:
        values = values + (1.0,) * (group_count - len(values))
    return jnp.asarray(values[:group_count], dtype=dtype)


def hierarchical_masked_focal_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    stage_map: dict[int, list[int]],
    focal_alpha: tuple[float, ...] | None,
    focal_gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    grouped = grouped_logits(logits, stage_map)
    flat_logits = grouped.reshape((-1, grouped.shape[-1]))
    flat_labels = normalize_gap_labels_jnp(labels.reshape((-1,)))

    gap_group = gap_group_index(stage_map)
    is_gap = flat_labels == GAP_LABEL
    if gap_group is None:
        valid = flat_labels >= 0
        safe_labels = jnp.where(valid, flat_labels, 0)
    else:
        valid = (flat_labels >= 0) | is_gap
        safe_labels = jnp.where(
            is_gap,
            gap_group,
            jnp.where(valid, flat_labels, 0),
        )
    log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
    probs = jnp.exp(log_probs)

    gathered_log_probs = log_probs[jnp.arange(flat_logits.shape[0]), safe_labels]
    gathered_probs = probs[jnp.arange(flat_logits.shape[0]), safe_labels]
    gathered_probs = jnp.clip(gathered_probs, 1e-7, 1.0 - 1e-7)

    alpha = _focal_alpha_for_group_count(
        focal_alpha,
        grouped.shape[-1],
        flat_logits.dtype,
    )
    alpha_t = alpha[safe_labels]

    ce_loss = -gathered_log_probs
    focal_weight = jnp.power(1.0 - gathered_probs, focal_gamma)
    per_token = alpha_t * focal_weight * ce_loss * valid.astype(flat_logits.dtype)
    valid_count = valid.astype(flat_logits.dtype).sum()
    loss_sum = per_token.sum()
    mean_loss = jnp.where(valid_count > 0, loss_sum / valid_count, 0.0)
    return mean_loss, loss_sum, valid_count


CURRICULUM_STAGE_MAPS = (STAGE_1_MAP, STAGE_2_MAP, STAGE_3_MAP)


def _grouped_probs_and_labels(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    stage_map: dict[int, list[int]],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Stage-grouped probabilities plus grouped labels and a validity mask.

    Same label handling as :func:`hierarchical_masked_focal_loss` — gap is a
    real class at its stage's group index, only ``-1`` is dropped.
    """
    grouped = grouped_logits(logits, stage_map)
    probs = jax.nn.softmax(grouped, axis=-1)

    normalized = normalize_gap_labels_jnp(labels)
    gap_group = gap_group_index(stage_map)
    is_gap = normalized == GAP_LABEL
    if gap_group is None:
        valid = normalized >= 0
        group_labels = jnp.where(valid, normalized, 0)
    else:
        valid = (normalized >= 0) | is_gap
        group_labels = jnp.where(
            is_gap,
            gap_group,
            jnp.where(valid, normalized, 0),
        )
    return probs, group_labels, valid


def weighted_curriculum_objective(
    logits: jnp.ndarray,
    stage_labels: jnp.ndarray,
    loss_weights: jnp.ndarray,
    stage_focal_alphas: tuple[tuple[float, ...] | None, ...],
    stage_focal_gammas: tuple[float, ...],
    stage_maps: tuple[dict[int, list[int]], ...] = CURRICULUM_STAGE_MAPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Blend the three fixed-shape hierarchical curriculum objectives.

    The weights are dynamic JAX inputs (normally one-hot).  This keeps stage
    maps and loss shapes inside one executable, so moving a curriculum boundary
    changes data rather than triggering a new JIT compilation.  ``stage_labels``
    contains host-remapped labels in stage-map order with shape ``[B, T, 3]``.
    """
    if stage_labels.shape[-1] != len(stage_maps):
        raise ValueError(
            "stage_labels must contain one label plane per curriculum stage; "
            f"got shape {stage_labels.shape}"
        )
    terms = [
        hierarchical_masked_focal_loss(
            logits,
            stage_labels[..., stage_index],
            stage_map,
            stage_focal_alphas[stage_index],
            stage_focal_gammas[stage_index],
        )
        for stage_index, stage_map in enumerate(stage_maps)
    ]
    weights = jnp.asarray(loss_weights, dtype=logits.dtype)
    return (
        jnp.vdot(weights, jnp.stack([term[0] for term in terms])),
        jnp.vdot(weights, jnp.stack([term[1] for term in terms])),
        jnp.vdot(weights, jnp.stack([term[2] for term in terms])),
    )


def feedback_adjustment_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    focal_alpha: tuple[float, float] | None = DEFAULT_FEEDBACK_FOCAL_ALPHA,
    focal_gamma: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Binary sleep/wake loss for sparse feedback labels.

    Feedback PSG labels are intentionally sparse: -1 is unspecified and -3 is
    GAP. Labels 0 and 1 train wake vs sleep, grouping all sleep
    stages through STAGE_1_MAP.
    """
    normalized_labels = normalize_gap_labels_jnp(labels)
    feedback_labels = jnp.where(
        (normalized_labels == 0)
        | (normalized_labels == 1)
        | (normalized_labels == GAP_LABEL),
        normalized_labels,
        -1,
    )
    return hierarchical_masked_focal_loss(
        logits,
        feedback_labels,
        STAGE_1_MAP,
        focal_alpha,
        focal_gamma,
    )


#: Epochs next to either view's context edges left out of the shift-consistency term;
#: the two views see different neighbours there.
CONSISTENCY_EDGE_EPOCHS = 60


def consistency_keep(shifts: np.ndarray, width: int) -> np.ndarray:
    """``(B, width)`` 1 where an epoch of the second view counts in the shift-consistency term.

    The second view is the first rolled forward by ``shifts``; an epoch counts when it is
    more than ``CONSISTENCY_EDGE_EPOCHS`` from the context edges of both views.
    """
    m = CONSISTENCY_EDGE_EPOCHS
    t = np.arange(width)[None, :]
    near_b = (t < m) | (t >= width - m)
    in_a = (t - np.asarray(shifts)[:, None]) % width
    near_a = (in_a < m) | (in_a >= width - m)
    return (~(near_a | near_b)).astype(np.float32)


def _roll_time_by_offsets(values: jnp.ndarray, offsets: jnp.ndarray) -> jnp.ndarray:
    time_steps = values.shape[1]
    base = jnp.arange(time_steps)[None, :]
    gather_idx = jnp.mod(base - offsets[:, None], time_steps)
    if values.ndim == 3:
        gather_idx = jnp.broadcast_to(gather_idx[:, :, None], values.shape)
    return jnp.take_along_axis(values, gather_idx, axis=1)


def masked_logit_pairing_loss(
    logits_a: jnp.ndarray,
    logits_b: jnp.ndarray,
    valid_mask: jnp.ndarray,
    stage_map: dict[int, list[int]],
) -> jnp.ndarray:
    grouped_a = grouped_logits(logits_a, stage_map)
    grouped_b = grouped_logits(logits_b, stage_map)
    mask = valid_mask.astype(grouped_a.dtype)
    per_token = jnp.mean(jnp.square(grouped_a - grouped_b), axis=-1) * mask
    valid_count = jnp.sum(mask)
    return jnp.where(valid_count > 0, jnp.sum(per_token) / valid_count, 0.0)


def masked_raw_logit_pairing_loss(
    student_logits: jnp.ndarray,
    target_logits: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    target = jax.lax.stop_gradient(target_logits)
    mask = valid_mask.astype(student_logits.dtype)
    per_token = jnp.mean(jnp.square(student_logits - target), axis=-1) * mask
    valid_count = jnp.sum(mask)
    return jnp.where(valid_count > 0, jnp.sum(per_token) / valid_count, 0.0)


def masked_class_accuracies(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> dict[str, float]:
    pred = logits.argmax(axis=-1)
    metrics: dict[str, float] = {}
    for class_idx in range(num_classes):
        mask = labels == class_idx
        metrics[f"class_{class_idx}_accuracy"] = float((pred[mask] == labels[mask]).mean()) if np.any(mask) else float("nan")
    return metrics


def _format_class_accuracy_metrics(metrics: dict[str, float], prefix: str = "val") -> str:
    parts: list[str] = []
    for key in sorted(metrics):
        value = metrics[key]
        class_suffix = key.removeprefix("class_").removesuffix("_accuracy")
        if math.isnan(value):
            parts.append(f"{prefix}_c{class_suffix}=nan")
        else:
            parts.append(f"{prefix}_c{class_suffix}={value:.3f}")
    return " ".join(parts)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _deblip_sleep_probabilities_np(
    predictions_proba: np.ndarray,
    *,
    epoch_seconds: float,
    min_wake_block_minutes: float,
    max_sleep_minutes: float,
) -> np.ndarray:
    probs = np.asarray(predictions_proba)
    if probs.ndim != 2:
        raise ValueError(f"expected 2D probabilities, got {probs.shape}")
    if probs.shape[-1] not in (4, 5):
        raise ValueError(f"expected 4- or 5-class probabilities, got {probs.shape}")
    if epoch_seconds <= 0:
        raise ValueError(f"epoch_seconds must be positive, got {epoch_seconds}")
    if min_wake_block_minutes <= 0:
        raise ValueError(f"min_wake_block_minutes must be positive, got {min_wake_block_minutes}")
    if max_sleep_minutes <= 0:
        raise ValueError(f"max_sleep_minutes must be positive, got {max_sleep_minutes}")

    n_epochs = int(probs.shape[0])
    if n_epochs == 0:
        return probs.copy()

    max_sleep_seconds = float(max_sleep_minutes) * 60.0
    min_wake_block_seconds = float(min_wake_block_minutes) * 60.0
    out = probs.copy()

    while True:
        stages = np.argmax(out, axis=-1)
        wake = stages == 0
        sleep = (stages > 0) & (stages < 4)
        changed = False

        i = 0
        while i < n_epochs:
            if not sleep[i]:
                i += 1
                continue

            sleep_start = i
            while i < n_epochs and sleep[i]:
                i += 1
            sleep_end = i
            sleep_seconds = (sleep_end - sleep_start) * float(epoch_seconds)
            if sleep_seconds >= max_sleep_seconds:
                continue

            left_start = sleep_start
            while left_start > 0 and wake[left_start - 1]:
                left_start -= 1
            right_end = sleep_end
            while right_end < n_epochs and wake[right_end]:
                right_end += 1

            has_left_wake = left_start < sleep_start
            has_right_wake = right_end > sleep_end
            merged_wake_seconds = (right_end - left_start) * float(epoch_seconds)
            if has_left_wake and has_right_wake and merged_wake_seconds >= min_wake_block_seconds:
                out[sleep_start:sleep_end, :] = 0.0
                out[sleep_start:sleep_end, 0] = 1.0
                changed = True

        if not changed:
            break

    return out


def soft_deblipped_logits_from_logits(
    logits: np.ndarray,
    *,
    epoch_seconds: float,
    min_wake_block_minutes: float,
    max_sleep_minutes: float,
    epsilon: float,
) -> np.ndarray:
    if not 0.0 < epsilon < 0.5:
        raise ValueError(f"epsilon must be between 0 and 0.5, got {epsilon}")
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"expected logits shape (B, T, C), got {logits.shape}")
    probs = _softmax_np(logits)
    deblipped = np.stack(
        [
            _deblip_sleep_probabilities_np(
                row,
                epoch_seconds=epoch_seconds,
                min_wake_block_minutes=min_wake_block_minutes,
                max_sleep_minutes=max_sleep_minutes,
            )
            for row in probs
        ],
        axis=0,
    )
    soft = np.clip(deblipped, epsilon, 1.0 - epsilon)
    soft = soft / np.sum(soft, axis=-1, keepdims=True)
    return np.log(soft).astype(np.float32)


def _save_training_history_plot(history: dict[str, list[float]], output_path: Path) -> None:
    # Plotting belongs to the optional training dependency set. Keeping this
    # import local lets a local inference installation avoid matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    if len(epochs) == 0:
        return

    class_metric_keys = sorted(
        key for key in history
        if key.startswith("val_class_") and key.endswith("_accuracy")
    )
    nrows = 2 if class_metric_keys else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), squeeze=False)

    loss_ax = axes[0, 0]
    loss_ax.plot(epochs, history["loss"], label="train_loss", linewidth=2.0)
    if "val_loss" in history and len(history["val_loss"]) == len(epochs):
        loss_ax.plot(epochs, history["val_loss"], label="val_loss", linewidth=2.0)
    loss_ax.set_title("Training Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    if class_metric_keys:
        acc_ax = axes[1, 0]
        for key in class_metric_keys:
            values = np.asarray(history[key], dtype=float)
            metric_epochs = np.arange(1, len(values) + 1)
            class_suffix = key.removeprefix("val_class_").removesuffix("_accuracy")
            acc_ax.plot(metric_epochs, values, label=f"class {class_suffix}", linewidth=2.0)
        acc_ax.set_title("Validation Per-Class Masked Accuracy")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.set_ylim(0.0, 1.0)
        acc_ax.grid(True, alpha=0.3)
        acc_ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _make_sliding_epoch_copies(
    train_x: np.ndarray,
    train_mask: np.ndarray,
    train_y: np.ndarray,
    sliding_copies: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one unshifted copy plus per-epoch random rotations along time."""
    if sliding_copies < 1:
        raise ValueError(f"Expected sliding_copies >= 1, got {sliding_copies}")
    if sliding_copies == 1:
        return train_x, train_mask, train_y

    num_examples, width = train_y.shape[:2]
    if train_x.shape[:2] != train_y.shape[:2] or train_mask.shape != train_y.shape[:2]:
        raise ValueError(
            "Expected train_x, train_mask, and train_y to agree on batch/time axes; "
            f"got X={train_x.shape}, mask={train_mask.shape}, y={train_y.shape}"
        )

    out_x = np.empty((num_examples * sliding_copies, *train_x.shape[1:]), dtype=train_x.dtype)
    out_mask = np.empty((num_examples * sliding_copies, width), dtype=train_mask.dtype)
    out_y = np.empty(
        (num_examples * sliding_copies, width, *train_y.shape[2:]),
        dtype=train_y.dtype,
    )
    random_offsets = rng.integers(
        0,
        width,
        size=(num_examples, sliding_copies - 1),
        dtype=np.int32,
    )

    out_idx = 0
    for row_idx in range(num_examples):
        offsets = (0, *random_offsets[row_idx])
        for offset in offsets:
            offset = int(offset)
            if offset == 0:
                out_x[out_idx] = train_x[row_idx]
                out_mask[out_idx] = train_mask[row_idx]
                out_y[out_idx] = train_y[row_idx]
            else:
                out_x[out_idx] = np.roll(train_x[row_idx], shift=offset, axis=0)
                out_mask[out_idx] = np.roll(train_mask[row_idx], shift=offset, axis=0)
                out_y[out_idx] = np.roll(train_y[row_idx], shift=offset, axis=0)
            out_idx += 1

    return out_x, out_mask, out_y


def _expand_dataset_labels_for_sliding_copies(
    dataset_labels: np.ndarray,
    sliding_copies: int,
) -> np.ndarray:
    labels = np.asarray(dataset_labels)
    if labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional dataset labels, got shape {labels.shape}")
    if sliding_copies < 1:
        raise ValueError(f"Expected sliding_copies >= 1, got {sliding_copies}")
    if sliding_copies == 1:
        return labels
    return np.repeat(labels, sliding_copies)


def _balanced_dataset_quotas(
    dataset_labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    del rng
    labels = np.asarray(dataset_labels).astype(str)
    datasets = sorted(set(labels.tolist()))
    if batch_size < 1:
        raise ValueError(f"Expected batch_size >= 1, got {batch_size}")
    base = math.ceil(batch_size / len(datasets))
    return {name: base for name in datasets}


def _make_balanced_dataset_batches(
    dataset_labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Build full optimizer batches with balanced per-dataset composition."""
    labels = np.asarray(dataset_labels).astype(str)
    if labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional dataset labels, got shape {labels.shape}")
    if labels.size == 0:
        raise ValueError("balanced_by_dataset requires at least one training row")

    quotas = _balanced_dataset_quotas(labels, batch_size, rng)
    groups = {
        name: np.flatnonzero(labels == name).astype(np.int32)
        for name in sorted(quotas)
    }
    counts = {name: int(indices.size) for name, indices in groups.items()}
    if any(count <= 0 for count in counts.values()):
        raise ValueError(f"balanced_by_dataset received empty dataset groups: {counts}")

    largest_count = max(counts.values())
    num_batches = max(
        int(math.ceil(count / quotas[name]))
        for name, count in counts.items()
        if count == largest_count
    )
    large_datasets = {name for name, count in counts.items() if count == largest_count}
    large_permutations = {
        name: rng.permutation(groups[name]).astype(np.int32)
        for name in large_datasets
    }

    batches: list[np.ndarray] = []
    for batch_idx in range(num_batches):
        pieces: list[np.ndarray] = []
        for name in sorted(quotas):
            quota = quotas[name]
            if name in large_datasets:
                permuted = large_permutations[name]
                start = batch_idx * quota
                stop = min(start + quota, permuted.size)
                chosen = permuted[start:stop]
                if chosen.size < quota:
                    fill = rng.choice(groups[name], size=quota - chosen.size, replace=True)
                    chosen = np.concatenate([chosen, fill.astype(np.int32)])
            else:
                chosen = rng.choice(groups[name], size=quota, replace=True).astype(np.int32)
            pieces.append(chosen)
        batch = np.concatenate(pieces).astype(np.int32)
        rng.shuffle(batch)
        batches.append(batch)

    return np.stack(batches, axis=0), quotas, counts


def _make_subsample_to_smallest_batches(
    dataset_labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """One epoch of batches drawn equally from every dataset, sized by the smallest.

    The smallest training set dictates the epoch: it is used in full, and every
    larger set contributes an equal-sized sample drawn without replacement and
    redrawn each epoch. So a 30-subject walch against a 246-subject bidsleep
    trains on 30 + 30 rows per epoch rather than 276, and the bidsleep 30 are a
    different 30 every epoch — over enough epochs the large set is still seen
    broadly, but no epoch is dominated by it.

    Contrast :func:`_make_balanced_dataset_batches`, which balances *within* each
    batch and sizes the epoch to cover the *largest* set once, oversampling the
    small ones with replacement. That equalizes per-batch composition at full
    cost; this equalizes per-epoch composition at the cost of the smallest set.

    Returns the batches plus the per-dataset sampled and total counts. The final
    batch is topped up with rows drawn from the same epoch sample, so every batch
    is exactly ``batch_size`` wide as ``planned_batches`` requires; the top-up
    rows are duplicates within that epoch.
    """
    labels = np.asarray(dataset_labels).astype(str)
    if labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional dataset labels, got shape {labels.shape}")
    if labels.size == 0:
        raise ValueError("subsample_to_smallest requires at least one training row")
    if batch_size < 1:
        raise ValueError(f"Expected batch_size >= 1, got {batch_size}")

    groups = {
        name: np.flatnonzero(labels == name).astype(np.int32)
        for name in sorted(set(labels.tolist()))
    }
    counts = {name: int(indices.size) for name, indices in groups.items()}
    target = min(counts.values())

    sampled: list[np.ndarray] = []
    for name in sorted(groups):
        indices = groups[name]
        chosen = (
            rng.permutation(indices)
            if indices.size == target
            else rng.choice(indices, size=target, replace=False)
        )
        sampled.append(chosen.astype(np.int32))
    epoch_indices = np.concatenate(sampled).astype(np.int32)
    rng.shuffle(epoch_indices)

    num_batches = max(int(math.ceil(epoch_indices.size / batch_size)), 1)
    shortfall = num_batches * batch_size - epoch_indices.size
    if shortfall:
        fill = rng.choice(epoch_indices, size=shortfall, replace=True).astype(np.int32)
        epoch_indices = np.concatenate([epoch_indices, fill])
    batches = epoch_indices.reshape(num_batches, batch_size)

    sampled_counts = {name: target for name in sorted(groups)}
    return batches, sampled_counts, counts


def _average_param_trees(param_trees: list[FrozenDict[str, Any]]) -> FrozenDict[str, Any]:
    if not param_trees:
        raise ValueError("Expected at least one parameter tree to average.")
    total = param_trees[0]
    for params in param_trees[1:]:
        total = jax.tree_util.tree_map(lambda a, b: a + b, total, params)
    inv_count = 1.0 / float(len(param_trees))
    return jax.tree_util.tree_map(lambda value: value * inv_count, total)


def bypass_learned_normalization(
    x: jnp.ndarray,
    valid_mask: jnp.ndarray | None = None,
    return_outputs: bool = False,
) -> jnp.ndarray | dict[str, jnp.ndarray]:
    valid_mask = _normalize_valid_mask(x, valid_mask)
    x_raw = x * valid_mask[:, :, None]
    if return_outputs:
        return {
            "valid_mask": valid_mask,
            "x_norm": x_raw,
            "x_combined": x_raw,
        }
    return x_raw


class CausalConv1DModule(nn.Module):
    filters: int
    kernel_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.kernel_size > 1:
            x = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        return nn.Conv(
            self.filters,
            kernel_size=(self.kernel_size,),
            padding="VALID",
            kernel_init=GLOROT_UNIFORM,
            bias_init=ZEROS,
            name="conv1d",
        )(x)


class ConvBlockModule(nn.Module):
    filters: int
    kernel_size: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        h = CausalConv1DModule(self.filters, self.kernel_size, name="causal_conv1d")(x)
        h = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="layer_normalization")(h)
        h = _gelu(h)
        return nn.Dropout(rate=self.dropout, name="dropout")(h, deterministic=not train)


class ConvBankModule(nn.Module):
    d_model: int
    kernel_sizes: Tuple[int, ...]
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        n_branches = len(self.kernel_sizes)
        d_per_branch = self.d_model // n_branches
        branch_outputs = []
        for kernel_size in self.kernel_sizes:
            branch_outputs.append(
                ConvBlockModule(
                    filters=d_per_branch,
                    kernel_size=kernel_size,
                    dropout=self.dropout,
                    name=f"conv_k{kernel_size}",
                )(x, train=train)
            )
        h = jnp.concatenate(branch_outputs, axis=-1)
        h = nn.Dense(self.d_model, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="proj")(h)
        return nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="norm")(h)


class ALiBiAttentionModule(nn.Module):
    d_model: int
    n_heads: int
    dropout: float
    attn_dropout: float

    def setup(self) -> None:
        self.d_head = self.d_model // self.n_heads
        self.scale = self.d_head ** -0.5
        self.slopes = jnp.asarray(_alibi_slopes(self.n_heads))

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal_mask: jnp.ndarray | None = None, train: bool = False) -> jnp.ndarray:
        batch_size, time_steps, _ = x.shape
        qkv = nn.Dense(3 * self.d_model, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="qkv")(x)
        qkv = qkv.reshape((batch_size, time_steps, 3, self.n_heads, self.d_head))
        q = jnp.transpose(qkv[:, :, 0, :, :], (0, 2, 1, 3))
        k = jnp.transpose(qkv[:, :, 1, :, :], (0, 2, 1, 3))
        v = jnp.transpose(qkv[:, :, 2, :, :], (0, 2, 1, 3))

        attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * self.scale
        positions = jnp.arange(time_steps, dtype=jnp.float32)
        dist = jnp.abs(positions[None, :] - positions[:, None])
        bias = -self.slopes[:, None, None] * dist[None, :, :]
        attn = attn + bias[None, :, :, :]
        if causal_mask is not None:
            attn = jnp.where(causal_mask[None, None, :, :], -1e9, attn)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_dropout, name="attn_drop")(attn, deterministic=not train)
        attn = nn.Dropout(rate=self.dropout, name="drop")(attn, deterministic=not train)

        out = jnp.matmul(attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, time_steps, self.d_model))
        return nn.Dense(self.d_model, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="out_proj")(out)


class TransformerBlockModule(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    attn_dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal_mask: jnp.ndarray | None = None, train: bool = False) -> jnp.ndarray:
        h = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="norm1")(x)
        h = ALiBiAttentionModule(self.d_model, self.n_heads, self.dropout, self.attn_dropout, name="attn")(h, causal_mask=causal_mask, train=train)
        x = x + h

        h = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="norm2")(x)
        h = nn.Dense(self.d_ff, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="ff_dense_0")(h)
        h = _gelu(h)
        h = nn.Dropout(rate=self.dropout, name="ff_drop_0")(h, deterministic=not train)
        h = nn.Dense(self.d_model, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="ff_dense_1")(h)
        h = nn.Dropout(rate=self.dropout, name="ff_drop_1")(h, deterministic=not train)
        return x + h


class BidirectionalCausalTransformerModule(nn.Module):
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    attn_dropout: float
    use_bicausal: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        heads_per_direction = self.n_heads // 2
        d_stream = self.d_model // 2
        time_steps = x.shape[1]
        if self.use_bicausal:
            fwd_mask = _causal_mask(time_steps)
            rev_mask = _reverse_causal_mask(time_steps)
        else:
            fwd_mask = None
            rev_mask = None

        h_fwd = nn.Dense(d_stream, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="in_proj_fwd")(x)
        h_rev = nn.Dense(d_stream, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="in_proj_rev")(x)

        for layer_idx in range(self.n_layers):
            h_fwd = TransformerBlockModule(d_stream, heads_per_direction, self.d_ff // 2, self.dropout, self.attn_dropout, name=f"fwd_{layer_idx}")(h_fwd, causal_mask=fwd_mask, train=train)
        for layer_idx in range(self.n_layers):
            h_rev = TransformerBlockModule(d_stream, heads_per_direction, self.d_ff // 2, self.dropout, self.attn_dropout, name=f"rev_{layer_idx}")(h_rev, causal_mask=rev_mask, train=train)

        h = jnp.concatenate([h_fwd, h_rev], axis=-1)
        h = nn.Dense(self.d_model, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="out_proj")(h)
        return nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="out_norm")(h)


class GatedFusionModule(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, h_local: jnp.ndarray, h_global: jnp.ndarray) -> jnp.ndarray:
        gate_local = nn.Dense(self.d_model, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="gate_local")(h_local)
        gate_global = nn.Dense(self.d_model, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="gate_global")(h_global)
        gate = nn.sigmoid(gate_local + gate_global)
        fused = gate * h_local + (1.0 - gate) * h_global
        return nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="norm")(fused)


class BiDozeNetModule(nn.Module):
    cfg: BiDozeNetConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        valid_mask: jnp.ndarray | None = None,
        train: bool = False,
        return_outputs: bool = False,
    ) -> jnp.ndarray | dict[str, jnp.ndarray]:
        if self.cfg.frames_per_epoch > 1:
            x = _pool_frames_to_epochs(x, self.cfg.frames_per_epoch)
            if valid_mask is not None:
                valid_mask = _pool_frames_to_epochs(valid_mask, self.cfg.frames_per_epoch)
        x = _flatten_timestep_features(x)
        # Zeroes invalid rows and passes the spectrogram through otherwise;
        # normalization is learned downstream by LayerNorm.
        learned_norm_outputs = bypass_learned_normalization(
            x,
            valid_mask=valid_mask,
            return_outputs=return_outputs,
        )
        x = learned_norm_outputs["x_combined"] if return_outputs else learned_norm_outputs

        h_local = ConvBankModule(
            self.cfg.d_model,
            self.cfg.conv_kernels,
            self.cfg.dropout,
            name="conv_bank",
        )(x, train=train)
        h_global = BidirectionalCausalTransformerModule(
            self.cfg.d_model,
            self.cfg.n_heads,
            self.cfg.n_transformer_layers,
            self.cfg.d_ff,
            self.cfg.dropout,
            self.cfg.attn_dropout,
            use_bicausal=self.cfg.use_bicausal,
            name="bi_causal_transformer",
        )(h_local, train=train)
        h_fused = GatedFusionModule(self.cfg.d_model, name="gated_fusion")(h_local, h_global)

        # One MLP readout head.
        h = nn.Dense(
            self.cfg.d_model // 2,
            kernel_init=GLOROT_UNIFORM,
            bias_init=ZEROS,
            name="head_hidden",
        )(h_fused)
        h = _gelu(h)
        h = nn.Dropout(rate=self.cfg.dropout, name="head_drop")(h, deterministic=not train)
        logits = nn.Dense(
            self.cfg.output_classes,
            kernel_init=GLOROT_UNIFORM,
            bias_init=ZEROS,
            name="head_logits",
        )(h)
        if return_outputs:
            assert isinstance(learned_norm_outputs, dict)
            return {
                "learned_normalization": learned_norm_outputs["x_norm"],
                "h_local": h_local,
                "h_global": h_global,
                "h_fused": h_fused,
                "probabilities": jax.nn.softmax(logits, axis=-1),
            }
        return logits


class TrainState(train_state.TrainState):
    pass


def _zero_grads_except_roots(
    grads: FrozenDict[str, Any],
    trainable_roots: set[str],
) -> FrozenDict[str, Any]:
    masked = {}
    for root_name, subtree in grads.items():
        if root_name in trainable_roots:
            masked[root_name] = subtree
        else:
            masked[root_name] = jax.tree_util.tree_map(jnp.zeros_like, subtree)
    return freeze(masked) if isinstance(grads, FrozenDict) else masked


def _mask_grads_for_trainable_scope(
    grads: FrozenDict[str, Any],
    trainable_scope: str,
) -> FrozenDict[str, Any]:
    if trainable_scope == "all":
        return grads
    if trainable_scope == "gate_head":
        return _zero_grads_except_roots(grads, GATE_HEAD_TRAINABLE_ROOTS)
    raise ValueError(f"Unknown trainable scope: {trainable_scope}")


def _trainable_scope_mask(
    params: FrozenDict[str, Any],
    trainable_scope: str,
) -> FrozenDict[str, Any]:
    if trainable_scope == "all":
        return jax.tree_util.tree_map(lambda _: True, params)
    if trainable_scope == "gate_head":
        mask = {}
        for root_name, subtree in params.items():
            trainable = root_name in GATE_HEAD_TRAINABLE_ROOTS
            mask[root_name] = jax.tree_util.tree_map(lambda _: trainable, subtree)
        return freeze(mask) if isinstance(params, FrozenDict) else mask
    raise ValueError(f"Unknown trainable scope: {trainable_scope}")


def _anchor_l2_penalty(
    params: FrozenDict[str, Any],
    anchor_params: FrozenDict[str, Any],
    trainable_scope: str,
) -> jnp.ndarray:
    # Sum (not mean) over trainable elements, so anchor_l2_weight's effective
    # gradient scale does not shrink as the trainable parameter count grows.
    mask = _trainable_scope_mask(anchor_params, trainable_scope)
    sq_sum = jnp.asarray(0.0, dtype=jnp.float32)
    for param_leaf, anchor_leaf, mask_leaf in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(anchor_params),
        jax.tree_util.tree_leaves(mask),
    ):
        if bool(mask_leaf):
            diff = param_leaf - anchor_leaf
            sq_sum = sq_sum + jnp.sum(jnp.square(diff))
    return sq_sum


def _adamw_for_trainable_scope(
    learning_rate: optax.ScalarOrSchedule,
    weight_decay: float,
    trainable_scope: str,
) -> optax.GradientTransformation:
    adamw = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    if trainable_scope == "all":
        return adamw
    if trainable_scope == "gate_head":
        return optax.masked(
            adamw,
            lambda params: _trainable_scope_mask(params, trainable_scope),
        )
    raise ValueError(f"Unknown trainable scope: {trainable_scope}")


@dataclass(frozen=True)
class RuntimeLearningRate:
    """Learning-rate knobs supplied at run time instead of baked into the graph.

    Flax makes ``TrainState.tx`` a static field, so a freshly built
    ``optax.chain`` — even an identically configured one — changes the treedef
    and forces every jitted train step to recompile.  Building one optimizer for
    the whole fit is only possible if the schedule's shape is data rather than
    Python closure state, so all three shapes collapse into a single formula:

        lr(count) = base_lr
                    * (1 + cos(pi * pos) * cosine_on) / (1 + cosine_on)
                    * decay_rate ** ((count / steps_per_epoch) * exponential_on)

    ``cosine_on=0`` makes the middle factor exactly ``1.0`` and
    ``exponential_on=0`` makes the last one exactly ``1.0``, so a disabled
    factor drops out of the product exactly.
    """

    base_lr: float
    cycle_steps: float = 1.0
    cosine_on: float = 0.0
    decay_rate: float = 1.0
    steps_per_epoch: float = 1.0
    exponential_on: float = 0.0

    @classmethod
    def constant(cls, learning_rate: float) -> "RuntimeLearningRate":
        return cls(base_lr=float(learning_rate))

    @classmethod
    def cosine(cls, learning_rate: float, cycle_steps: int) -> "RuntimeLearningRate":
        return cls(
            base_lr=float(learning_rate),
            cycle_steps=float(max(int(cycle_steps), 1)),
            cosine_on=1.0,
        )

    @classmethod
    def exponential(
        cls,
        learning_rate: float,
        decay_rate: float,
        steps_per_epoch: int,
    ) -> "RuntimeLearningRate":
        return cls(
            base_lr=float(learning_rate),
            decay_rate=float(decay_rate),
            steps_per_epoch=float(max(int(steps_per_epoch), 1)),
            exponential_on=1.0,
        )

    def as_hyperparams(self) -> dict[str, jnp.ndarray]:
        return {
            "base_lr": jnp.float32(self.base_lr),
            "cycle_steps": jnp.float32(self.cycle_steps),
            "cosine_on": jnp.float32(self.cosine_on),
            "decay_rate": jnp.float32(self.decay_rate),
            "steps_per_epoch": jnp.float32(self.steps_per_epoch),
            "exponential_on": jnp.float32(self.exponential_on),
        }

    def describe(self) -> str:
        if self.exponential_on:
            return f"exponential (base_lr={self.base_lr:.3g}, decay_rate={self.decay_rate:.4g}/epoch)"
        if self.cosine_on:
            return f"cosine (base_lr={self.base_lr:.3g}, cycle {int(self.cycle_steps)} steps)"
        return f"constant (base_lr={self.base_lr:.3g})"


# One optimizer object per (weight_decay, trainable_scope) for the whole process.
# optax transformations are pure, and the trainable-scope mask is evaluated
# against whatever params are passed to ``init``, so sharing is safe across
# stages, adjustment phases, and fit seeds, so crossing any of those boundaries
# reuses the compiled train step instead of retracing it.
_RUNTIME_LR_OPTIMIZERS: dict[tuple[float, str], optax.GradientTransformation] = {}


def _runtime_lr_optimizer(
    weight_decay: float,
    trainable_scope: str,
) -> optax.GradientTransformation:
    key = (float(weight_decay), str(trainable_scope))
    if key not in _RUNTIME_LR_OPTIMIZERS:

        def inner(
            base_lr: jax.Array,
            cycle_steps: jax.Array,
            cosine_on: jax.Array,
            decay_rate: jax.Array,
            steps_per_epoch: jax.Array,
            exponential_on: jax.Array,
        ) -> optax.GradientTransformation:
            def schedule(count: jax.Array) -> jax.Array:
                count_f = count.astype(jnp.float32)
                cycle_pos = jnp.mod(count_f, cycle_steps) / cycle_steps
                cosine = (1.0 + jnp.cos(jnp.pi * cycle_pos) * cosine_on) / (1.0 + cosine_on)
                decay = jnp.power(decay_rate, (count_f / steps_per_epoch) * exponential_on)
                return base_lr * cosine * decay

            return optax.chain(
                optax.clip_by_global_norm(1.0),
                _adamw_for_trainable_scope(schedule, weight_decay, trainable_scope),
            )

        _RUNTIME_LR_OPTIMIZERS[key] = optax.inject_hyperparams(inner)(
            **RuntimeLearningRate(base_lr=0.0).as_hyperparams()
        )
    return _RUNTIME_LR_OPTIMIZERS[key]


def _make_train_state(
    apply_fn: Callable[..., Any],
    params: FrozenDict[str, Any],
    learning_rate: RuntimeLearningRate,
    weight_decay: float,
    trainable_scope: str = "all",
) -> TrainState:
    """Fresh optimizer state (moments and step counter reset) on a shared ``tx``."""
    state = TrainState.create(
        apply_fn=apply_fn,
        params=params,
        tx=_runtime_lr_optimizer(weight_decay, trainable_scope),
    )
    hyperparams = dict(state.opt_state.hyperparams)
    hyperparams.update(learning_rate.as_hyperparams())
    return state.replace(opt_state=state.opt_state._replace(hyperparams=hyperparams))


def _make_train_step(
    module: nn.Module,
    stage_map: dict[int, list[int]],
    focal_alpha: tuple[float, ...] | None,
    focal_gamma: float,
    trainable_scope: str = "all",
    anchor_params: FrozenDict[str, Any] | None = None,
    anchor_l2_weight: float = 0.0,
) -> Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array]]:
    if trainable_scope not in TRAINABLE_SCOPES:
        raise ValueError(f"Unknown trainable scope: {trainable_scope}")
    anchor_weight = float(anchor_l2_weight)
    if anchor_weight < 0.0:
        raise ValueError("anchor_l2_weight must be non-negative")

    @jax.jit
    def train_step(
        local_state: TrainState,
        batch_x: jax.Array,
        batch_mask: jax.Array,
        batch_y: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        def loss_fn(params: FrozenDict[str, Any]) -> jnp.ndarray:
            logits = module.apply(
                {"params": params},
                batch_x,
                batch_mask,
                train=True,
                rngs={"dropout": rng_key, "augment": rng_key},
            )
            loss, _, _ = hierarchical_masked_focal_loss(
                logits, batch_y, stage_map, focal_alpha, focal_gamma
            )
            if anchor_params is not None and anchor_weight > 0.0:
                loss = loss + anchor_weight * _anchor_l2_penalty(
                    params,
                    anchor_params,
                    trainable_scope,
                )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(local_state.params)
        grads = _mask_grads_for_trainable_scope(grads, trainable_scope)
        local_state = local_state.apply_gradients(grads=grads)
        return local_state, loss

    return train_step


def _make_weighted_curriculum_train_step(
    module: nn.Module,
    stage_focal_alphas: tuple[tuple[float, ...] | None, ...],
    stage_focal_gammas: tuple[float, ...],
    trainable_scope: str = "all",
    anchor_params: FrozenDict[str, Any] | None = None,
    anchor_l2_weight: float = 0.0,
    stage_maps: tuple[dict[int, list[int]], ...] = CURRICULUM_STAGE_MAPS,
) -> Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array]]:
    """One compiled main-step for all curriculum stages.

    Unlike :func:`_make_train_step`, the active stage is selected by the
    runtime ``loss_weights`` vector, not static Python closure state.
    """
    if trainable_scope not in TRAINABLE_SCOPES:
        raise ValueError(f"Unknown trainable scope: {trainable_scope}")
    anchor_weight = float(anchor_l2_weight)
    if anchor_weight < 0.0:
        raise ValueError("anchor_l2_weight must be non-negative")

    @jax.jit
    def train_step(
        local_state: TrainState,
        batch_x: jax.Array,
        batch_mask: jax.Array,
        batch_stage_labels: jax.Array,
        loss_weights: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        def loss_fn(params: FrozenDict[str, Any]) -> jnp.ndarray:
            logits = module.apply(
                {"params": params},
                batch_x,
                batch_mask,
                train=True,
                rngs={"dropout": rng_key, "augment": rng_key},
            )
            loss, _, _ = weighted_curriculum_objective(
                logits,
                batch_stage_labels,
                loss_weights,
                stage_focal_alphas,
                stage_focal_gammas,
                stage_maps,
            )
            if anchor_params is not None and anchor_weight > 0.0:
                loss = loss + anchor_weight * _anchor_l2_penalty(
                    params,
                    anchor_params,
                    trainable_scope,
                )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(local_state.params)
        grads = _mask_grads_for_trainable_scope(grads, trainable_scope)
        local_state = local_state.apply_gradients(grads=grads)
        return local_state, loss

    return train_step


def _make_adjustment_train_step(
    module: nn.Module,
    focal_alpha: tuple[float, float] | None,
    focal_gamma: float,
    trainable_scope: str = "all",
) -> Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array]]:
    if trainable_scope not in TRAINABLE_SCOPES:
        raise ValueError(f"Unknown adjustment trainable scope: {trainable_scope}")

    @jax.jit
    def train_step(
        local_state: TrainState,
        batch_x: jax.Array,
        batch_mask: jax.Array,
        batch_y: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        def loss_fn(params: FrozenDict[str, Any]) -> jnp.ndarray:
            logits = module.apply(
                {"params": params},
                batch_x,
                batch_mask,
                train=True,
                rngs={"dropout": rng_key, "augment": rng_key},
            )
            loss, _, _ = feedback_adjustment_loss(logits, batch_y, focal_alpha, focal_gamma)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(local_state.params)
        grads = _mask_grads_for_trainable_scope(grads, trainable_scope)
        local_state = local_state.apply_gradients(grads=grads)
        return local_state, loss

    return train_step


def _make_adjustment_target_train_step(
    module: nn.Module,
    focal_alpha: tuple[float, float] | None,
    focal_gamma: float,
    target_logit_pairing_weight: float,
    trainable_scope: str = "all",
) -> Callable[
    [TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[TrainState, jax.Array, jax.Array, jax.Array],
]:
    if trainable_scope not in TRAINABLE_SCOPES:
        raise ValueError(f"Unknown adjustment trainable scope: {trainable_scope}")
    target_weight = float(target_logit_pairing_weight)

    @jax.jit
    def train_step(
        local_state: TrainState,
        feedback_x: jax.Array,
        feedback_mask: jax.Array,
        feedback_y: jax.Array,
        target_x: jax.Array,
        target_mask: jax.Array,
        target_logits: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[TrainState, jax.Array, jax.Array, jax.Array]:
        def loss_fn(params: FrozenDict[str, Any]) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
            feedback_key, target_key = jax.random.split(rng_key)
            feedback_logits = module.apply(
                {"params": params},
                feedback_x,
                feedback_mask,
                train=True,
                rngs={"dropout": feedback_key, "augment": feedback_key},
            )
            feedback_loss, _, _ = feedback_adjustment_loss(
                feedback_logits,
                feedback_y,
                focal_alpha,
                focal_gamma,
            )
            student_logits = module.apply(
                {"params": params},
                target_x,
                target_mask,
                train=True,
                rngs={"dropout": target_key, "augment": target_key},
            )
            target_loss = masked_raw_logit_pairing_loss(
                student_logits,
                target_logits,
                target_mask,
            )
            total_loss = feedback_loss + target_weight * target_loss
            return total_loss, (feedback_loss, target_loss)

        (loss, (feedback_loss, target_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            local_state.params
        )
        grads = _mask_grads_for_trainable_scope(grads, trainable_scope)
        local_state = local_state.apply_gradients(grads=grads)
        return local_state, loss, feedback_loss, target_loss

    return train_step


def _make_run_full_batches(
    train_step: Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array]],
) -> Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array, jax.Array]]:
    @jax.jit
    def run_full_batches(
        local_state: TrainState,
        train_x: jax.Array,
        train_mask: jax.Array,
        train_y: jax.Array,
        batch_indices: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[TrainState, jax.Array, jnp.ndarray]:
        def scan_step(
            carry: tuple[TrainState, jax.Array],
            batch_idx: jnp.ndarray,
        ) -> tuple[tuple[TrainState, jax.Array], jnp.ndarray]:
            scan_state, scan_key = carry
            next_key, step_key = jax.random.split(scan_key)
            scan_state, loss = train_step(
                scan_state,
                train_x[batch_idx],
                train_mask[batch_idx],
                train_y[batch_idx],
                step_key,
            )
            return (scan_state, next_key), loss

        (local_state, rng_key), losses = jax.lax.scan(scan_step, (local_state, rng_key), batch_indices)
        return local_state, rng_key, losses

    return run_full_batches


def _make_run_weighted_curriculum_full_batches(
    train_step: Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array]],
) -> Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array, jax.Array]]:
    @jax.jit
    def run_full_batches(
        local_state: TrainState,
        train_x: jax.Array,
        train_mask: jax.Array,
        train_stage_labels: jax.Array,
        loss_weights: jax.Array,
        batch_indices: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[TrainState, jax.Array, jnp.ndarray]:
        def scan_step(
            carry: tuple[TrainState, jax.Array],
            batch_idx: jnp.ndarray,
        ) -> tuple[tuple[TrainState, jax.Array], jnp.ndarray]:
            scan_state, scan_key = carry
            next_key, step_key = jax.random.split(scan_key)
            scan_state, loss = train_step(
                scan_state,
                train_x[batch_idx],
                train_mask[batch_idx],
                train_stage_labels[batch_idx],
                loss_weights,
                step_key,
            )
            return (scan_state, next_key), loss

        (local_state, rng_key), losses = jax.lax.scan(scan_step, (local_state, rng_key), batch_indices)
        return local_state, rng_key, losses

    return run_full_batches


def _make_eval_step(
    module: nn.Module,
    stage_map: dict[int, list[int]],
    focal_alpha: tuple[float, ...] | None,
    focal_gamma: float,
) -> Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    @jax.jit
    def eval_step(
        params: FrozenDict[str, Any],
        batch_x: jax.Array,
        batch_mask: jax.Array,
        batch_y: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits = module.apply({"params": params}, batch_x, batch_mask, train=False)
        _, loss_sum, valid_count = hierarchical_masked_focal_loss(
            logits, batch_y, stage_map, focal_alpha, focal_gamma
        )
        return loss_sum, valid_count, logits

    return eval_step


def _make_weighted_curriculum_eval_step(
    module: nn.Module,
    stage_focal_alphas: tuple[tuple[float, ...] | None, ...],
    stage_focal_gammas: tuple[float, ...],
    stage_maps: tuple[dict[int, list[int]], ...] = CURRICULUM_STAGE_MAPS,
) -> Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    @jax.jit
    def eval_step(
        params: FrozenDict[str, Any],
        batch_x: jax.Array,
        batch_mask: jax.Array,
        batch_stage_labels: jax.Array,
        loss_weights: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits = module.apply({"params": params}, batch_x, batch_mask, train=False)
        _, loss_sum, valid_count = weighted_curriculum_objective(
            logits,
            batch_stage_labels,
            loss_weights,
            stage_focal_alphas,
            stage_focal_gammas,
            stage_maps,
        )
        return loss_sum, valid_count, logits

    return eval_step


def _make_run_weighted_curriculum_eval_batches(
    eval_step: Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
) -> Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    @jax.jit
    def run_eval_batches(
        params: FrozenDict[str, Any],
        eval_x: jax.Array,
        eval_mask: jax.Array,
        eval_stage_labels: jax.Array,
        loss_weights: jax.Array,
        batch_indices: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def scan_step(
            carry: None,
            batch_idx: jnp.ndarray,
        ) -> tuple[None, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            loss_sum, valid_count, logits = eval_step(
                params,
                eval_x[batch_idx],
                eval_mask[batch_idx],
                eval_stage_labels[batch_idx],
                loss_weights,
            )
            return None, (loss_sum, valid_count, logits)

        _, outputs = jax.lax.scan(scan_step, None, batch_indices)
        return outputs

    return run_eval_batches


def _make_run_eval_batches(
    eval_step: Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
) -> Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    @jax.jit
    def run_eval_batches(
        params: FrozenDict[str, Any],
        eval_x: jax.Array,
        eval_mask: jax.Array,
        eval_y: jax.Array,
        batch_indices: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def scan_step(
            carry: None,
            batch_idx: jnp.ndarray,
        ) -> tuple[None, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            loss_sum, valid_count, logits = eval_step(
                params,
                eval_x[batch_idx],
                eval_mask[batch_idx],
                eval_y[batch_idx],
            )
            return None, (loss_sum, valid_count, logits)

        _, outputs = jax.lax.scan(scan_step, None, batch_indices)
        return outputs

    return run_eval_batches


class BiDozeNetModel(CVModel):
    config_class: ClassVar[Type[ModelConfig]] = BiDozeNetConfig
    backend: ClassVar[str] = "jax"

    def __init__(self, config: BiDozeNetConfig):
        super().__init__(config)
        self.config = config
        self.module = BiDozeNetModule(config)
        self.params: FrozenDict[str, Any] | None = None
        self.model: dict[str, Any] | None = None
        self.input_shape: tuple[int, ...] | None = None
        self._build_seed = 0
        # Zipper feedback can run between curriculum stages.  Keep the compiled
        # adjustment functions alive across those calls; the optimizer state is
        # still deliberately re-created for each adjustment phase.
        self._adjustment_step_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}

    @property
    def is_trained(self) -> bool:
        return self.params is not None

    def returns_logits(self) -> bool:
        return True

    def _build_model(self, input_shape: Tuple[int, ...]) -> FrozenDict[str, Any]:
        self.input_shape = tuple(input_shape)
        init_key = jax.random.PRNGKey(self._build_seed)
        params_key, dropout_key, augment_key = jax.random.split(init_key, 3)
        dummy_x = jnp.zeros((1, *input_shape), dtype=jnp.float32)
        dummy_mask = jnp.ones((1, input_shape[0]), dtype=jnp.float32)
        variables = self.module.init(
            {"params": params_key, "dropout": dropout_key, "augment": augment_key},
            dummy_x,
            dummy_mask,
            train=False,
        )
        self.params = variables["params"]
        self.model = {"params": self.params}
        return self.params

    @staticmethod
    def _default_valid_mask(input_x: np.ndarray) -> np.ndarray:
        return np.ones(input_x.shape[:2], dtype=np.float32)

    @staticmethod
    def _coerce_valid_mask(input_x: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
        if valid_mask is None:
            return BiDozeNetModel._default_valid_mask(input_x)

        mask = np.asarray(valid_mask, dtype=np.float32)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError(f"Expected valid_mask shape (B, T) or (B, T, 1), got {mask.shape}")

        batch_size, target_t = input_x.shape[:2]
        if mask.shape[0] != batch_size:
            raise ValueError(f"Expected valid_mask batch dimension {batch_size}, got {mask.shape[0]}")
        if mask.shape[1] < target_t:
            pad = np.zeros((batch_size, target_t - mask.shape[1]), dtype=np.float32)
            mask = np.concatenate([mask, pad], axis=1)
        elif mask.shape[1] > target_t:
            mask = mask[:, :target_t]
        return mask

    def _ensure_built(self, input_shape: tuple[int, ...], seed: int | None = None) -> None:
        if self.params is None:
            if seed is not None:
                self._build_seed = int(seed)
            self._build_model(input_shape)

    def _resolve_summary_input_shape(self, input_shape: tuple[int, ...] | None) -> tuple[int, int]:
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError(
                    "Provide input_shape=(time_steps, ...) before calling summary() on an unbuilt model."
                )
            input_shape = self.input_shape

        if len(input_shape) < 2:
            raise ValueError(f"Expected input_shape=(time_steps, ...), got {input_shape}")

        time_steps = int(input_shape[0])
        n_freq = int(np.prod(input_shape[1:]))
        if time_steps <= 0 or n_freq <= 0:
            raise ValueError(f"Expected positive input dimensions, got {input_shape}")
        return time_steps, n_freq

    @staticmethod
    def _format_graph_shape(*dims: int) -> str:
        return "[" + " x ".join(str(int(dim)) for dim in dims) + "]"

    def _graphical_summary(self, time_steps: int, n_freq: int, batch_size: int, train: bool) -> str:
        cfg = self.config
        input_shape = self._format_graph_shape(batch_size, time_steps, n_freq)
        mask_shape = self._format_graph_shape(batch_size, time_steps)
        normalized_shape = self._format_graph_shape(batch_size, time_steps, n_freq)
        model_shape = self._format_graph_shape(batch_size, time_steps, cfg.d_model)
        hidden_shape = self._format_graph_shape(batch_size, time_steps, cfg.d_model // 2)
        logits_shape = self._format_graph_shape(batch_size, time_steps, cfg.output_classes)
        conv_name = "ConvBankModule"
        fusion_name = "GatedFusionModule"

        lines = [
            "BiDozeNetModel graph",
            f"input_x {input_shape}",
            "  |",
            f"  +--> valid_mask {mask_shape}",
        ]

        lines.extend(
            [
                "  |",
                "  +--> invalid-row zeroing",
                "  |     uses: x, valid_mask",
                f"  |     out: x_combined {normalized_shape}",
            ]
        )

        lines.extend(
            [
                "  |",
                f"  +--> {conv_name}",
                f"  |     kernels={cfg.conv_kernels} d_model={cfg.d_model} dropout={cfg.dropout}",
                f"  |     out: h_local {model_shape}",
                "  |     |",
                "  |     +--> BidirectionalCausalTransformerModule",
                (
                    "  |     |     "
                    f"layers={cfg.n_transformer_layers} heads={cfg.n_heads} d_ff={cfg.d_ff} "
                    f"attn_dropout={cfg.attn_dropout}"
                ),
                f"  |     |     out: h_global {model_shape}",
                "  |     |",
                f"  |     +--> {fusion_name}(h_local, h_global)",
                f"  |           out: h_fused {model_shape}",
            ]
        )

        lines.extend(
            [
                "  |           |",
                "  |           +--> head_hidden",
                f"  |           |     out: h_hidden {hidden_shape}",
                "  |           +--> gelu",
                f"  |           |     out: h_hidden {hidden_shape}",
                f"  |           +--> head_drop(train={train}, rate={cfg.dropout})",
                f"  |           |     out: h_hidden {hidden_shape}",
                "  |           +--> head_logits",
                f"  |                 out: logits {logits_shape}",
            ]
        )
        return "\n".join(lines)

    def summary(
        self,
        input_shape: tuple[int, ...] | None = None,
        batch_size: int = 1,
        train: bool = False,
        depth: int | None = None,
        show_repeated: bool = False,
        graphical: bool = True,
    ) -> str:
        time_steps, n_freq = self._resolve_summary_input_shape(input_shape)
        if batch_size <= 0:
            raise ValueError(f"Expected batch_size >= 1, got {batch_size}")
        if graphical:
            return self._graphical_summary(time_steps, n_freq, batch_size, train)

        init_key = jax.random.PRNGKey(self._build_seed)
        params_key, dropout_key, augment_key = jax.random.split(init_key, 3)
        summary_shape = input_shape if input_shape is not None else (self.input_shape or (time_steps, n_freq))
        dummy_x = jnp.zeros((batch_size, *summary_shape), dtype=jnp.float32)
        dummy_mask = jnp.ones((batch_size, time_steps), dtype=jnp.float32)
        return self.module.tabulate(
            {"params": params_key, "dropout": dropout_key, "augment": augment_key},
            dummy_x,
            dummy_mask,
            train=train,
            depth=depth,
            show_repeated=show_repeated,
            console_kwargs={"force_terminal": False, "width": 120},
        )

    def infer(self, input_x: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        input_x = np.asarray(input_x, dtype=np.float32)
        if input_x.ndim < 3:
            raise ValueError(f"Expected input shape (B, T, ...), got {input_x.shape}")
        self._ensure_built(tuple(input_x.shape[1:]))
        assert self.params is not None
        if valid_mask is None:
            if self.config.score_gap:
                valid_mask = np.ones(input_x.shape[:2], dtype=np.float32)
            else:
                valid_mask = _numpy_feature_activity_mask(input_x)
        else:
            valid_mask = self._coerce_valid_mask(input_x, valid_mask)
        logits = self.module.apply({"params": self.params}, jnp.asarray(input_x), jnp.asarray(valid_mask), train=False)
        return np.asarray(logits, dtype=np.float32)

    def soft_deblipped_target_logits(
        self,
        input_x: np.ndarray,
        *,
        epoch_seconds: float,
        min_wake_block_minutes: float,
        max_sleep_minutes: float,
        epsilon: float,
        batch_size: int | None = None,
    ) -> np.ndarray:
        input_x = np.asarray(input_x, dtype=np.float32)
        if input_x.ndim < 3:
            raise ValueError(f"Expected input shape (B, T, ...), got {input_x.shape}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"Expected batch_size >= 1, got {batch_size}")

        if batch_size is None:
            logits = self.infer(input_x)
        else:
            chunks = [
                self.infer(input_x[start : start + batch_size])
                for start in range(0, input_x.shape[0], batch_size)
            ]
            logits = np.concatenate(chunks, axis=0) if chunks else np.empty(
                (0, input_x.shape[1], self.config.output_classes),
                dtype=np.float32,
            )
        return soft_deblipped_logits_from_logits(
            logits,
            epoch_seconds=epoch_seconds,
            min_wake_block_minutes=min_wake_block_minutes,
            max_sleep_minutes=max_sleep_minutes,
            epsilon=epsilon,
        )

    def intermediates(self, input_x: np.ndarray, valid_mask: np.ndarray | None = None) -> list[dict[str, np.ndarray]]:
        input_x = np.asarray(input_x, dtype=np.float32)
        if input_x.ndim < 3:
            raise ValueError(f"Expected input shape (B, T, ...), got {input_x.shape}")
        self._ensure_built(tuple(input_x.shape[1:]))
        assert self.params is not None
        if valid_mask is None:
            if self.config.score_gap:
                valid_mask = np.ones(input_x.shape[:2], dtype=np.float32)
            else:
                valid_mask = _numpy_feature_activity_mask(input_x)
        else:
            valid_mask = self._coerce_valid_mask(input_x, valid_mask)

        outputs = self.module.apply(
            {"params": self.params},
            jnp.asarray(input_x),
            jnp.asarray(valid_mask),
            train=False,
            return_outputs=True,
        )
        assert isinstance(outputs, dict)
        ordered_names = [
            "learned_normalization",
            "h_local",
            "h_global",
            "h_fused",
            "probabilities",
        ]
        return [{name: np.asarray(outputs[name], dtype=np.float32)} for name in ordered_names]

    def _flatten_params(self, params: FrozenDict[str, Any]) -> tuple[list[str], list[np.ndarray]]:
        flat = traverse_util.flatten_dict(unfreeze(params), sep="/")
        names = sorted(flat.keys())
        arrays = [np.asarray(flat[name]) for name in names]
        return names, arrays

    def parameter_count(self) -> int:
        if self.params is None:
            raise ValueError("an unbuilt model has no parameters")
        _names, arrays = self._flatten_params(self.params)
        return sum(int(array.size) for array in arrays)

    def _restore_from_manifest(self, names: Iterable[str], arrays: Iterable[np.ndarray]) -> FrozenDict[str, Any]:
        flat = {tuple(name.split("/")): jnp.asarray(array) for name, array in zip(names, arrays)}
        return freeze(traverse_util.unflatten_dict(flat))

    def save_weights_to_directory(self, weights_dir: Path) -> None:
        if self.params is None or self.input_shape is None:
            return
        names, arrays = self._flatten_params(self.params)
        np.savez(weights_dir / "model_weights.npz", *arrays)
        (weights_dir / "input_shape.json").write_text(json.dumps(list(self.input_shape)))
        (weights_dir / "weight_manifest.json").write_text(json.dumps(names, indent=2))

    def save_weights_safetensors(self, path: Path) -> None:
        """Write the fitted parameters to a `safetensors` file.

        The public release format: a flat ``{name: tensor}`` map with the same
        slash-separated names ``weight_manifest.json`` records, but in a
        documented, non-executable container. Metadata carries the input shape
        so a loader can rebuild the module before restoring the tree.
        """
        try:
            from safetensors.numpy import save_file
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via CLI
            raise ModuleNotFoundError(
                "saving safetensors needs the optional dependency: "
                "pip install 'arcasleep[publish]'",
                name="safetensors",
            ) from exc
        if self.params is None or self.input_shape is None:
            raise ValueError("cannot export safetensors from an unbuilt model")
        names, arrays = self._flatten_params(self.params)
        tensors = {name: np.ascontiguousarray(array) for name, array in zip(names, arrays)}
        save_file(tensors, str(path), metadata={"input_shape": json.dumps(list(self.input_shape))})

    def load_weights_safetensors(self, path: Path, input_shape: tuple[int, ...] | None = None) -> None:
        """Restore parameters from `safetensors`, rebuilding the module first."""
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via CLI
            raise ModuleNotFoundError(
                "loading safetensors needs the optional dependency: "
                "pip install 'arcasleep[inference]'",
                name="safetensors",
            ) from exc
        with safe_open(str(path), framework="np") as handle:
            if input_shape is None:
                metadata = handle.metadata() or {}
                if "input_shape" not in metadata:
                    raise ValueError(f"{path} carries no input_shape metadata; pass input_shape=")
                input_shape = tuple(json.loads(metadata["input_shape"]))
            names = sorted(handle.keys())
            arrays = [handle.get_tensor(name) for name in names]
        self._ensure_built(tuple(input_shape))
        self.params = self._restore_from_manifest(names, arrays)
        self.model = {"params": self.params}
        self.input_shape = tuple(input_shape)

    def load_weights_from_directory(
        self,
        weights_dir: Path,
        input_shape: tuple[int, ...] | None = None,
        allow_partial: bool = False,
        init_seed: int | None = None,
    ) -> None:
        weights_path = weights_dir / "model_weights.npz"
        shape_path = weights_dir / "input_shape.json"
        manifest_path = weights_dir / "weight_manifest.json"

        if not weights_path.exists() and not shape_path.exists():
            return
        if not weights_path.exists() or not shape_path.exists():
            raise ValueError(f"Expected both model_weights.npz and input_shape.json in {weights_dir}")
        if not manifest_path.exists():
            raise ValueError(
                f"Missing weight_manifest.json in {weights_dir}; a weights directory "
                "holds model_weights.npz, input_shape.json and weight_manifest.json."
            )

        if input_shape is None:
            input_shape = tuple(json.loads(shape_path.read_text()))

        # Seed the build, not just the fit. Under a partial load the params that
        # the checkpoint does not carry keep whatever this build drew. Without a
        # seed here that is always PRNGKey(_build_seed=0), and the later
        # _ensure_built(seed=fit_seed) inside fit_with_callbacks is a no-op
        # because self.params is already set. Every "seed" of such a run would
        # then share one readout init and vary only in batch order.
        self._ensure_built(tuple(input_shape), seed=init_seed)
        data = np.load(weights_path, allow_pickle=False)
        arrays = [data[key] for key in sorted(data.files, key=lambda value: int(value[4:]))]
        names = json.loads(manifest_path.read_text())
        if not allow_partial:
            self.params = self._restore_from_manifest(names, arrays)
        else:
            assert self.params is not None
            source = dict(zip(names, arrays))
            params = unfreeze(self.params)
            target = traverse_util.flatten_dict(params, sep="/")
            for name, value in source.items():
                if name in target and target[name].shape == value.shape:
                    target[name] = jnp.asarray(value)
            self.params = freeze(traverse_util.unflatten_dict({tuple(name.split("/")): value for name, value in target.items()}))
        self.model = {"params": self.params}
        self.input_shape = tuple(input_shape)

    def _run_epoch(
        self,
        state: TrainState,
        train_x: jax.Array,
        train_mask: jax.Array,
        train_y: jax.Array,
        train_step: Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array]],
        run_full_batches: Callable[[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[TrainState, jax.Array, jnp.ndarray]],
        batch_size: int,
        rng: np.random.Generator,
        train_rng_key: jax.Array,
        planned_batches: np.ndarray | None = None,
    ) -> tuple[TrainState, float, jax.Array]:
        num_examples = int(train_x.shape[0])
        if planned_batches is None:
            indices = rng.permutation(num_examples)
            num_full_batches = len(indices) // batch_size
            full_size = num_full_batches * batch_size
            full_batches = indices[:full_size].reshape(num_full_batches, batch_size) if num_full_batches else None
            tail_indices = indices[full_size:]
        else:
            full_batches = np.asarray(planned_batches, dtype=np.int32)
            if full_batches.ndim != 2 or full_batches.shape[1] != batch_size:
                raise ValueError(
                    "planned_batches must have shape (num_batches, batch_size); "
                    f"got {full_batches.shape} for batch_size={batch_size}"
                )
            tail_indices = np.empty((0,), dtype=np.int32)

        current_key = train_rng_key
        loss_sum = jnp.array(0.0, dtype=jnp.float32)
        loss_count = 0

        if full_batches is not None and len(full_batches):
            state, current_key, full_losses = run_full_batches(
                state,
                train_x,
                train_mask,
                train_y,
                jnp.asarray(full_batches, dtype=jnp.int32),
                current_key,
            )
            loss_sum = loss_sum + jnp.sum(full_losses)
            loss_count += int(full_losses.shape[0])

        if len(tail_indices):
            current_key, step_key = jax.random.split(current_key)
            state, tail_loss = train_step(
                state,
                train_x[jnp.asarray(tail_indices, dtype=jnp.int32)],
                train_mask[jnp.asarray(tail_indices, dtype=jnp.int32)],
                train_y[jnp.asarray(tail_indices, dtype=jnp.int32)],
                step_key,
            )
            loss_sum = loss_sum + tail_loss
            loss_count += 1

        mean_loss = float(jax.device_get(loss_sum / loss_count)) if loss_count else float("nan")
        return state, mean_loss, current_key

    def _run_weighted_curriculum_epoch(
        self,
        state: TrainState,
        train_x: jax.Array,
        train_mask: jax.Array,
        train_stage_labels: jax.Array,
        loss_weights: jax.Array,
        train_step: Callable[..., tuple[TrainState, jax.Array]],
        run_full_batches: Callable[..., tuple[TrainState, jax.Array, jax.Array]],
        batch_size: int,
        rng: np.random.Generator,
        train_rng_key: jax.Array,
        planned_batches: np.ndarray | None = None,
    ) -> tuple[TrainState, float, jax.Array]:
        """Run one epoch with dynamic, one-hot curriculum loss weights."""
        num_examples = int(train_x.shape[0])
        if planned_batches is None:
            indices = rng.permutation(num_examples)
            num_full_batches = len(indices) // batch_size
            full_size = num_full_batches * batch_size
            full_batches = (
                indices[:full_size].reshape(num_full_batches, batch_size)
                if num_full_batches
                else None
            )
            tail_indices = indices[full_size:]
        else:
            full_batches = np.asarray(planned_batches, dtype=np.int32)
            if full_batches.ndim != 2 or full_batches.shape[1] != batch_size:
                raise ValueError(
                    "planned_batches must have shape (num_batches, batch_size); "
                    f"got {full_batches.shape} for batch_size={batch_size}"
                )
            tail_indices = np.empty((0,), dtype=np.int32)

        current_key = train_rng_key
        loss_sum = jnp.array(0.0, dtype=jnp.float32)
        loss_count = 0
        if full_batches is not None and len(full_batches):
            state, current_key, full_losses = run_full_batches(
                state,
                train_x,
                train_mask,
                train_stage_labels,
                loss_weights,
                jnp.asarray(full_batches, dtype=jnp.int32),
                current_key,
            )
            loss_sum = loss_sum + jnp.sum(full_losses)
            loss_count += int(full_losses.shape[0])
        if len(tail_indices):
            current_key, step_key = jax.random.split(current_key)
            batch_idx = jnp.asarray(tail_indices, dtype=jnp.int32)
            state, tail_loss = train_step(
                state,
                train_x[batch_idx],
                train_mask[batch_idx],
                train_stage_labels[batch_idx],
                loss_weights,
                step_key,
            )
            loss_sum = loss_sum + tail_loss
            loss_count += 1
        mean_loss = float(jax.device_get(loss_sum / loss_count)) if loss_count else float("nan")
        return state, mean_loss, current_key

    def _run_adjustment_target_epoch(
        self,
        state: TrainState,
        feedback_x: jax.Array,
        feedback_mask: jax.Array,
        feedback_y: jax.Array,
        target_x: np.ndarray,
        target_mask: np.ndarray,
        target_logits: np.ndarray,
        train_step: Callable[
            [TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            tuple[TrainState, jax.Array, jax.Array, jax.Array],
        ],
        batch_size: int,
        rng: np.random.Generator,
        train_rng_key: jax.Array,
    ) -> tuple[TrainState, float, float, float, jax.Array]:
        num_feedback = int(feedback_x.shape[0])
        num_target = int(target_x.shape[0])
        num_steps = max(
            math.ceil(num_feedback / batch_size),
            math.ceil(num_target / batch_size),
            1,
        )
        feedback_indices = rng.permutation(num_feedback)
        target_indices = rng.permutation(num_target)

        current_key = train_rng_key
        loss_sum = jnp.array(0.0, dtype=jnp.float32)
        feedback_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        target_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        loss_count = 0

        for step_idx in range(num_steps):
            start = step_idx * batch_size
            stop = start + batch_size
            feedback_batch = feedback_indices[start: min(stop, num_feedback)]
            if len(feedback_batch) == 0:
                feedback_batch = rng.choice(
                    num_feedback,
                    size=min(batch_size, num_feedback),
                    replace=num_feedback < batch_size,
                )
            target_batch = target_indices[start: min(stop, num_target)]
            if len(target_batch) == 0:
                target_batch = rng.choice(
                    num_target,
                    size=min(batch_size, num_target),
                    replace=num_target < batch_size,
                )
            feedback_batch_indices = jnp.asarray(feedback_batch, dtype=jnp.int32)
            target_batch_indices = np.asarray(target_batch, dtype=np.int32)
            current_key, step_key = jax.random.split(current_key)
            state, loss, feedback_loss, target_loss = train_step(
                state,
                feedback_x[feedback_batch_indices],
                feedback_mask[feedback_batch_indices],
                feedback_y[feedback_batch_indices],
                jax.device_put(target_x[target_batch_indices]),
                jax.device_put(target_mask[target_batch_indices]),
                jax.device_put(target_logits[target_batch_indices]),
                step_key,
            )
            loss_sum = loss_sum + loss
            feedback_loss_sum = feedback_loss_sum + feedback_loss
            target_loss_sum = target_loss_sum + target_loss
            loss_count += 1

        if not loss_count:
            return state, float("nan"), float("nan"), float("nan"), current_key
        mean_loss, mean_feedback_loss, mean_target_loss = jax.device_get(
            (
                loss_sum / loss_count,
                feedback_loss_sum / loss_count,
                target_loss_sum / loss_count,
            )
        )
        return (
            state,
            float(mean_loss),
            float(mean_feedback_loss),
            float(mean_target_loss),
            current_key,
        )

    def _evaluate_loss(
        self,
        params: FrozenDict[str, Any],
        eval_x: jax.Array,
        eval_mask: jax.Array,
        eval_y: jax.Array,
        eval_step: Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
        run_eval_batches: Callable[[FrozenDict[str, Any], jax.Array, jax.Array, jax.Array, jax.Array], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        eval_batch_size: int = 16,
    ) -> tuple[float, np.ndarray]:
        num_examples = int(eval_x.shape[0])
        indices = np.arange(num_examples, dtype=np.int32)
        num_full_batches = len(indices) // eval_batch_size
        full_size = num_full_batches * eval_batch_size
        full_batches = indices[:full_size].reshape(num_full_batches, eval_batch_size) if num_full_batches else None
        tail_indices = indices[full_size:]

        loss_sum_total = jnp.array(0.0, dtype=jnp.float32)
        valid_count_total = jnp.array(0.0, dtype=jnp.float32)
        logits_chunks: list[jax.Array] = []

        if full_batches is not None:
            loss_sums, valid_counts, logits = run_eval_batches(
                params,
                eval_x,
                eval_mask,
                eval_y,
                jnp.asarray(full_batches, dtype=jnp.int32),
            )
            loss_sum_total = loss_sum_total + jnp.sum(loss_sums)
            valid_count_total = valid_count_total + jnp.sum(valid_counts)
            logits_chunks.append(logits.reshape((-1, logits.shape[-1])))

        if len(tail_indices):
            tail_loss_sum, tail_valid_count, tail_logits = eval_step(
                params,
                eval_x[jnp.asarray(tail_indices, dtype=jnp.int32)],
                eval_mask[jnp.asarray(tail_indices, dtype=jnp.int32)],
                eval_y[jnp.asarray(tail_indices, dtype=jnp.int32)],
            )
            loss_sum_total = loss_sum_total + tail_loss_sum
            valid_count_total = valid_count_total + tail_valid_count
            logits_chunks.append(tail_logits.reshape((-1, tail_logits.shape[-1])))

        if logits_chunks:
            logits_all = jnp.concatenate(logits_chunks, axis=0)
            total_valid, total_loss_sum, logits_host = jax.device_get(
                (valid_count_total, loss_sum_total, logits_all)
            )
        else:
            total_valid, total_loss_sum, logits_host = jax.device_get(
                (valid_count_total, loss_sum_total, jnp.empty((0,), dtype=jnp.float32))
            )

        mean_loss = float(total_loss_sum / total_valid) if total_valid > 0 else 0.0
        return mean_loss, np.asarray(logits_host)

    def _evaluate_weighted_curriculum_loss(
        self,
        params: FrozenDict[str, Any],
        eval_x: jax.Array,
        eval_mask: jax.Array,
        eval_stage_labels: jax.Array,
        loss_weights: jax.Array,
        eval_step: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
        run_eval_batches: Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        eval_batch_size: int = 16,
    ) -> tuple[float, np.ndarray]:
        """Evaluate the same dynamic curriculum objective used for training."""
        num_examples = int(eval_x.shape[0])
        indices = np.arange(num_examples, dtype=np.int32)
        num_full_batches = len(indices) // eval_batch_size
        full_size = num_full_batches * eval_batch_size
        full_batches = (
            indices[:full_size].reshape(num_full_batches, eval_batch_size)
            if num_full_batches
            else None
        )
        tail_indices = indices[full_size:]
        loss_sum_total = jnp.array(0.0, dtype=jnp.float32)
        valid_count_total = jnp.array(0.0, dtype=jnp.float32)
        logits_chunks: list[jax.Array] = []
        if full_batches is not None:
            loss_sums, valid_counts, logits = run_eval_batches(
                params,
                eval_x,
                eval_mask,
                eval_stage_labels,
                loss_weights,
                jnp.asarray(full_batches, dtype=jnp.int32),
            )
            loss_sum_total = loss_sum_total + jnp.sum(loss_sums)
            valid_count_total = valid_count_total + jnp.sum(valid_counts)
            logits_chunks.append(logits.reshape((-1, logits.shape[-1])))
        if len(tail_indices):
            loss_sum, valid_count, logits = eval_step(
                params,
                eval_x[jnp.asarray(tail_indices, dtype=jnp.int32)],
                eval_mask[jnp.asarray(tail_indices, dtype=jnp.int32)],
                eval_stage_labels[jnp.asarray(tail_indices, dtype=jnp.int32)],
                loss_weights,
            )
            loss_sum_total = loss_sum_total + loss_sum
            valid_count_total = valid_count_total + valid_count
            logits_chunks.append(logits.reshape((-1, logits.shape[-1])))
        if logits_chunks:
            logits_all = jnp.concatenate(logits_chunks, axis=0)
            total_valid, total_loss_sum, logits_host = jax.device_get(
                (valid_count_total, loss_sum_total, logits_all)
            )
        else:
            total_valid, total_loss_sum, logits_host = jax.device_get(
                (valid_count_total, loss_sum_total, jnp.empty((0,), dtype=jnp.float32))
            )
        mean_loss = float(total_loss_sum / total_valid) if total_valid > 0 else 0.0
        return mean_loss, np.asarray(logits_host)

    def train_with_callbacks(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
        initial_lr: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        callbacks: List,
        checkpoint_path: Path | None = None,
        train_verbose: int = 1,
        fit_seed: int | None = None,
        sliding_copies: int = 1,
        shift_pairing_weight: float = 0.0,
        deblip_x: np.ndarray | None = None,
        deblip_epochs: int = 0,
        deblip_schedule: str = "interleaved",
        deblip_pairing_weight: float = 0.0,
        deblip_epoch_seconds: float = 30.0,
        deblip_min_wake_block_minutes: float = 15.0,
        deblip_max_sleep_minutes: float = 5.0,
        deblip_epsilon: float = 1e-7,
        stage_maps_override: list | None = None,
        disable_swa: bool = False,
        checkpoint_epoch_dir: "Path | None" = None,
        checkpoint_epochs: tuple[int, ...] = (),
        train_dataset_labels: np.ndarray | None = None,
        train_batch_sampling: str | None = None,
        subsample_to_smallest: bool = False,
        save_history_plot: bool = True,
        lr_schedule: str = "cosine",
        lr_decay_rate: float = 1.0,
        curriculum_loss_mode: str = "staged",
        trainable_scope: str = "all",
        anchor_l2_weight: float = 0.0,
        epoch_observer: Callable | None = None,
        validation_fraction: float = VAL_SUBSAMPLE_FRACTION,
        validation_batch_size: int = 4,
    ) -> dict[str, list[float]] | None:
        del callbacks
        if not 0.0 < validation_fraction <= 1.0:
            raise ValueError("validation_fraction must be in (0, 1]")
        if validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive")
        sliding_copies = int(sliding_copies)
        if sliding_copies < 1:
            raise ValueError(f"Expected sliding_copies >= 1, got {sliding_copies}")
        curriculum_loss_mode = str(curriculum_loss_mode)
        if curriculum_loss_mode not in {"staged", "weighted"}:
            raise ValueError("curriculum_loss_mode must be 'staged' or 'weighted'")
        trainable_scope = str(trainable_scope)
        if trainable_scope not in TRAINABLE_SCOPES:
            raise ValueError(f"Unknown trainable scope: {trainable_scope}")
        anchor_l2_weight = float(anchor_l2_weight)
        if anchor_l2_weight < 0.0:
            raise ValueError("anchor_l2_weight must be non-negative")
        shift_pairing_weight = float(shift_pairing_weight)
        deblip_pairing_weight = float(deblip_pairing_weight)
        if curriculum_loss_mode == "weighted" and stage_maps_override is not None:
            raise ValueError("weighted curriculum does not support stage_maps_override")
        if shift_pairing_weight < 0.0:
            raise ValueError("shift_pairing_weight must be non-negative")
        if deblip_pairing_weight < 0.0:
            raise ValueError("deblip_pairing_weight must be non-negative")
        deblip_schedule = str(deblip_schedule)
        if deblip_schedule not in {"interleaved", "final"}:
            raise ValueError(
                "deblip_schedule must be 'interleaved' or 'final', got "
                f"{deblip_schedule!r}"
            )

        train_x = np.asarray(train_x, dtype=np.float32)
        train_y = np.asarray(train_y, dtype=np.int32)
        train_dataset_labels = (
            None
            if train_dataset_labels is None
            else np.asarray(train_dataset_labels).astype(str)
        )
        train_batch_sampling = "disabled" if train_batch_sampling is None else str(train_batch_sampling)
        if train_batch_sampling not in {"disabled", "balanced_by_dataset"}:
            raise ValueError(f"Unsupported train_batch_sampling mode: {train_batch_sampling!r}")
        subsample_to_smallest = bool(subsample_to_smallest)
        if subsample_to_smallest and train_batch_sampling == "balanced_by_dataset":
            raise ValueError(
                "subsample_to_smallest and train_batch_sampling='balanced_by_dataset' both "
                "plan the epoch's batches; enable only one"
            )
        if train_batch_sampling == "balanced_by_dataset" or subsample_to_smallest:
            requirement = (
                "subsample_to_smallest"
                if subsample_to_smallest
                else "balanced_by_dataset sampling"
            )
            if train_dataset_labels is None:
                raise ValueError(f"train_dataset_labels are required for {requirement}")
            if train_dataset_labels.shape != (train_x.shape[0],):
                raise ValueError(
                    "train_dataset_labels must align to train_x rows; "
                    f"got labels={train_dataset_labels.shape}, train_x rows={train_x.shape[0]}"
                )
        val_x = None if val_x is None else np.asarray(val_x, dtype=np.float32)
        val_y = None if val_y is None else np.asarray(val_y, dtype=np.int32)
        deblip_x = None if deblip_x is None else np.asarray(deblip_x, dtype=np.float32)

        resolved_seed = _resolve_seed(fit_seed)
        self._ensure_built(tuple(train_x.shape[1:]), seed=resolved_seed)
        assert self.params is not None
        seed_sequence = np.random.SeedSequence(resolved_seed)
        slide_seed, batch_seed, sliding_copy_seed = seed_sequence.spawn(3)

        # Gap epochs are a trainable class only when the model has a
        # Gap output; otherwise they are masked out like unlabeled epochs.
        if self.config.score_gap:
            train_mask = ((train_y >= 0) | (train_y == GAP_LABEL)).astype(np.float32)
            val_mask = (
                ((val_y >= 0) | (val_y == GAP_LABEL)).astype(np.float32)
                if val_y is not None
                else None
            )
        else:
            train_mask = (train_y >= 0).astype(np.float32)
            val_mask = (val_y >= 0).astype(np.float32) if val_y is not None else None

        if self.config.temporal_slide:
            from arcasleep.data.batching import epoch_padding

            rng_np = np.random.default_rng(slide_seed)
            width = train_x.shape[1]
            train_x = train_x.copy()
            train_y = train_y.copy()
            train_mask = train_mask.copy()
            for row_idx in range(len(train_x)):
                valid_len = int(np.sum(train_mask[row_idx]))
                if valid_len < width:
                    offset = int(rng_np.integers(0, width - valid_len + 1))
                    # Filled with absent epochs (gap row at 1), like the
                    # batch padding, not zeros the model would read as data.
                    shifted_x = epoch_padding(train_x[row_idx, :valid_len], width)
                    shifted_y = np.full_like(train_y[row_idx], -2)
                    shifted_mask = np.zeros_like(train_mask[row_idx])
                    shifted_x[offset : offset + valid_len] = train_x[row_idx, :valid_len]
                    shifted_y[offset : offset + valid_len] = train_y[row_idx, :valid_len]
                    shifted_mask[offset : offset + valid_len] = train_mask[row_idx, :valid_len]
                    train_x[row_idx] = shifted_x
                    train_y[row_idx] = shifted_y
                    train_mask[row_idx] = shifted_mask

        weighted_curriculum = curriculum_loss_mode == "weighted"
        # Without a gap output the curriculum runs on maps that have no
        # gap group, so those epochs stay masked instead of being folded
        # into the last sleep class.
        curriculum_stage_maps = (
            CURRICULUM_STAGE_MAPS
            if self.config.score_gap
            else tuple(stage_map_without_gap(m) for m in CURRICULUM_STAGE_MAPS)
        )
        train_stage_labels = None
        val_stage_labels = None
        if weighted_curriculum:
            train_stage_labels = np.stack(
                [remap_classes(train_y, stage_map) for stage_map in curriculum_stage_maps],
                axis=-1,
            )
            val_stage_labels = (
                None
                if val_y is None
                else np.stack(
                    [remap_classes(val_y, stage_map) for stage_map in curriculum_stage_maps],
                    axis=-1,
                )
            )

        train_x_device = jax.device_put(train_x) if sliding_copies == 1 else None
        train_mask_device = jax.device_put(train_mask) if sliding_copies == 1 else None
        val_x_device = None if val_x is None else jax.device_put(val_x)
        val_mask_device = None if val_mask is None else jax.device_put(val_mask)
        train_stage_labels_device = (
            jax.device_put(train_stage_labels)
            if train_stage_labels is not None and sliding_copies == 1
            else None
        )
        val_stage_labels_device = (
            None if val_stage_labels is None else jax.device_put(val_stage_labels)
        )

        # ``stage_maps_override`` enables single-stage warm-start fine-tuning: run only
        # the supplied stage(s) for exactly ``epochs`` epochs, skipping the curriculum's
        # per-stage fraction split (which would otherwise give the lone stage ~0.10*epochs).
        if stage_maps_override is not None:
            stage_maps = list(stage_maps_override)
            epoch_counts = [int(epochs)] * len(stage_maps)
        else:
            stage_maps = list(curriculum_stage_maps)
            fractions = np.asarray(self.config.stage_epoch_fractions, dtype=np.float32)
            fractions = fractions / fractions.sum()
            epoch_counts = [max(1, round(float(frac) * epochs)) for frac in fractions]
            epoch_counts[-1] = max(1, epochs - sum(epoch_counts[:-1]))

        history: dict[str, list[float]] = {"loss": []}
        if val_x is not None:
            history["val_loss"] = []

        best_params = self.params
        batch_rng = np.random.default_rng(batch_seed)
        sliding_copy_rng = np.random.default_rng(sliding_copy_seed)
        train_rng_key = jax.random.PRNGKey(resolved_seed)
        params = self.params
        anchor_params = self.params if anchor_l2_weight > 0.0 else None
        final_stage_idx = len(stage_maps) - 1
        swa_params_accum: list[FrozenDict[str, Any]] = []
        # Running mean of the SWA window, kept only so validation can score the
        # weights that actually ship. See the swa_val_loss block below.
        swa_running_mean: FrozenDict[str, Any] | None = None
        # Validation is scored on a fresh uniform VAL_SUBSAMPLE_FRACTION of the
        # records each epoch rather than all of them: a full pass over a
        # test-sized validation set costs about as much as the training step.
        # Its own RNG, deliberately -- drawing from the training stream would
        # change the fitted weights.
        val_subsample_rng = np.random.default_rng(resolved_seed + VAL_SUBSAMPLE_SEED_OFFSET)
        weighted_train_step = None
        weighted_run_full_batches = None
        weighted_eval_step = None
        weighted_run_eval_batches = None
        if weighted_curriculum:
            curriculum_alphas = tuple(
                None if alpha is None else tuple(alpha)
                for alpha in self.config.stage_focal_alphas
            )
            curriculum_gammas = tuple(float(gamma) for gamma in self.config.stage_focal_gammas)
            weighted_train_step = _make_weighted_curriculum_train_step(
                self.module,
                curriculum_alphas,
                curriculum_gammas,
                trainable_scope=trainable_scope,
                anchor_params=anchor_params,
                anchor_l2_weight=anchor_l2_weight,
                stage_maps=curriculum_stage_maps,
            )
            weighted_run_full_batches = _make_run_weighted_curriculum_full_batches(
                weighted_train_step
            )
            weighted_eval_step = _make_weighted_curriculum_eval_step(
                self.module,
                curriculum_alphas,
                curriculum_gammas,
                stage_maps=curriculum_stage_maps,
            )
            weighted_run_eval_batches = _make_run_weighted_curriculum_eval_batches(
                weighted_eval_step
            )

        for stage_idx, stage_map in enumerate(stage_maps):
            stage_epochs = epoch_counts[stage_idx]
            # A single-stage override runs at final granularity, so use the final-stage
            # focal settings rather than stage_idx==0's (which is STAGE_1's coarse config).
            focal_idx = (
                len(self.config.stage_focal_alphas) - 1
                if stage_maps_override is not None
                else stage_idx
            )
            stage_alpha = self.config.stage_focal_alphas[focal_idx]
            stage_gamma = float(self.config.stage_focal_gammas[focal_idx])
            stage_loss_weights_device = (
                jax.device_put(np.eye(len(curriculum_stage_maps), dtype=np.float32)[stage_idx])
                if weighted_curriculum
                else None
            )
            train_y_stage = remap_classes(train_y, stage_map)
            val_y_stage = remap_classes(val_y, stage_map) if val_y is not None else None
            train_y_stage_device = jax.device_put(train_y_stage) if sliding_copies == 1 else None
            val_y_stage_device = None if val_y_stage is None else jax.device_put(val_y_stage)
            # Built outside the epoch loop so every epoch reuses the same executable.
            train_step = _make_train_step(
                self.module,
                stage_map,
                None if stage_alpha is None else tuple(stage_alpha),
                stage_gamma,
                trainable_scope=trainable_scope,
                anchor_params=anchor_params,
                anchor_l2_weight=anchor_l2_weight,
            )
            run_full_batches = _make_run_full_batches(train_step)
            eval_step = _make_eval_step(
                self.module,
                stage_map,
                None if stage_alpha is None else tuple(stage_alpha),
                stage_gamma,
            )
            run_eval_batches = _make_run_eval_batches(eval_step)

            balanced_batches = None
            balanced_quotas = None
            balanced_counts = None
            subsample_sampled_counts = None
            subsample_total_counts = None
            balanced_dataset_labels = train_dataset_labels
            if train_batch_sampling == "balanced_by_dataset" or subsample_to_smallest:
                assert balanced_dataset_labels is not None
                if sliding_copies > 1:
                    balanced_dataset_labels = _expand_dataset_labels_for_sliding_copies(
                        balanced_dataset_labels,
                        sliding_copies,
                    )
                preview_rng = np.random.default_rng(
                    np.random.SeedSequence([resolved_seed, stage_idx, 0])
                )
                if subsample_to_smallest:
                    (
                        balanced_batches,
                        subsample_sampled_counts,
                        subsample_total_counts,
                    ) = _make_subsample_to_smallest_batches(
                        balanced_dataset_labels,
                        batch_size,
                        preview_rng,
                    )
                else:
                    balanced_batches, balanced_quotas, balanced_counts = _make_balanced_dataset_batches(
                        balanced_dataset_labels,
                        batch_size,
                        preview_rng,
                    )
            train_examples_per_epoch = int(train_x.shape[0]) * sliding_copies
            optimizer_examples_per_epoch = (
                int(balanced_batches.size)
                if balanced_batches is not None
                else train_examples_per_epoch)
            steps_per_epoch = max(math.ceil(optimizer_examples_per_epoch / batch_size), 1)
            if stage_idx == final_stage_idx:
                if lr_schedule == "cosine":
                    sgdr_cycle_epochs = max(1, stage_epochs // 2)
                    swa_start_epoch = None if disable_swa else sgdr_cycle_epochs
                    stage_lr = RuntimeLearningRate.cosine(
                        initial_lr,
                        max(steps_per_epoch * max(sgdr_cycle_epochs, 1), 1),
                    )
                elif lr_schedule == "exponential":
                    sgdr_cycle_epochs = 0
                    swa_start_epoch = None
                    stage_lr = RuntimeLearningRate.exponential(
                        initial_lr,
                        lr_decay_rate,
                        steps_per_epoch,
                    )
                else:  # "constant"
                    sgdr_cycle_epochs = 0
                    swa_start_epoch = None
                    stage_lr = RuntimeLearningRate.constant(initial_lr)
            else:
                sgdr_cycle_epochs = 0
                swa_start_epoch = None
                stage_lr = RuntimeLearningRate.constant(initial_lr)
            # One shared executable across stages: only the LR hyperparams differ,
            # and they live in opt_state as device scalars, not in the treedef.
            state = _make_train_state(
                self.module.apply,
                params,
                stage_lr,
                weight_decay,
                trainable_scope=trainable_scope,
            )
            best_val_loss = float("inf") if stage_idx == final_stage_idx else None

            if train_verbose:
                class_count = f"{len(stage_map)-1}+1" if self.config.score_gap else str(len(stage_map))
                print(f"\nStage {stage_idx + 1}/{len(stage_maps)}: {class_count} grouped classes for {stage_epochs} epochs")
                if weighted_curriculum:
                    print("  curriculum loss: dynamic one-hot weights (shared compiled objective)")
                elif sliding_copies > 1:
                    print(f"  sliding copies: {sliding_copies} ({train_examples_per_epoch} examples per epoch)")
                if train_batch_sampling == "balanced_by_dataset":
                    assert balanced_quotas is not None
                    assert balanced_counts is not None
                    print(
                        "  batch sampling: balanced_by_dataset "
                        f"quotas={balanced_quotas} counts={balanced_counts} "
                        f"batches_per_epoch={steps_per_epoch}"
                    )
                if subsample_to_smallest:
                    assert subsample_sampled_counts is not None
                    assert subsample_total_counts is not None
                    print(
                        "  batch sampling: subsample_to_smallest "
                        f"sampled={subsample_sampled_counts} counts={subsample_total_counts} "
                        f"batches_per_epoch={steps_per_epoch} (resampled every epoch)"
                    )
                print(f"  LR (runtime hyperparams): {stage_lr.describe()}")
                if stage_idx == final_stage_idx:
                    if lr_schedule == "cosine":
                        print(f"  final-stage LR schedule: cosine (cycle {sgdr_cycle_epochs} epochs)")
                        print(f"  SWA accumulation starts at epoch {swa_start_epoch}")
                    elif lr_schedule == "exponential":
                        print(f"  final-stage LR schedule: exponential (decay_rate={lr_decay_rate} per epoch)")
                    else:
                        print(f"  final-stage LR schedule: constant")
                if anchor_l2_weight > 0.0:
                    print(
                        "  anchor L2 regularization: "
                        f"weight={anchor_l2_weight:.3g} scope={trainable_scope}"
                    )

            for epoch_idx in range(stage_epochs):
                active_train_step = train_step
                active_run_full_batches = run_full_batches
                active_weighted_train_step = weighted_train_step
                active_weighted_run_full_batches = weighted_run_full_batches
                if sliding_copies > 1:
                    labels_for_epoch = train_stage_labels if weighted_curriculum else train_y_stage
                    assert labels_for_epoch is not None
                    epoch_train_x, epoch_train_mask, epoch_train_y = _make_sliding_epoch_copies(
                        train_x,
                        train_mask,
                        labels_for_epoch,
                        sliding_copies,
                        sliding_copy_rng,
                    )
                    epoch_train_x_device = jax.device_put(epoch_train_x)
                    epoch_train_mask_device = jax.device_put(epoch_train_mask)
                    epoch_train_y_device = jax.device_put(epoch_train_y)
                else:
                    assert train_x_device is not None
                    assert train_mask_device is not None
                    if weighted_curriculum:
                        assert train_stage_labels_device is not None
                    else:
                        assert train_y_stage_device is not None
                    epoch_train_x_device = train_x_device
                    epoch_train_mask_device = train_mask_device
                    epoch_train_y_device = (
                        train_stage_labels_device if weighted_curriculum else train_y_stage_device
                    )
                planned_batches = None
                if train_batch_sampling == "balanced_by_dataset" or subsample_to_smallest:
                    assert balanced_dataset_labels is not None
                    epoch_plan_rng = np.random.default_rng(
                        np.random.SeedSequence([resolved_seed, stage_idx, epoch_idx])
                    )
                    plan_batches = (
                        _make_subsample_to_smallest_batches
                        if subsample_to_smallest
                        else _make_balanced_dataset_batches
                    )
                    planned_batches, _, _ = plan_batches(
                        balanced_dataset_labels,
                        batch_size,
                        epoch_plan_rng,
                    )
                epoch_start = time.perf_counter()
                if weighted_curriculum:
                    assert active_weighted_train_step is not None
                    assert active_weighted_run_full_batches is not None
                    assert stage_loss_weights_device is not None
                    state, train_loss, train_rng_key = self._run_weighted_curriculum_epoch(
                        state,
                        epoch_train_x_device,
                        epoch_train_mask_device,
                        epoch_train_y_device,
                        stage_loss_weights_device,
                        active_weighted_train_step,
                        active_weighted_run_full_batches,
                        batch_size,
                        batch_rng,
                        train_rng_key,
                        planned_batches=planned_batches,
                    )
                    ce_loss = adv_pair_loss = shift_pair_loss = float("nan")
                else:
                    state, train_loss, train_rng_key = self._run_epoch(
                        state,
                        epoch_train_x_device,
                        epoch_train_mask_device,
                        epoch_train_y_device,
                        active_train_step,
                        active_run_full_batches,
                        batch_size,
                        batch_rng,
                        train_rng_key,
                        planned_batches=planned_batches,
                    )
                    ce_loss = adv_pair_loss = shift_pair_loss = float("nan")
                train_elapsed = time.perf_counter() - epoch_start
                params = state.params
                history["loss"].append(train_loss)

                message = f"stage {stage_idx + 1} epoch {epoch_idx + 1}/{stage_epochs} loss={train_loss:.4f}"
                if weighted_curriculum:
                    has_validation = (
                        val_x_device is not None
                        and val_stage_labels_device is not None
                        and val_mask_device is not None
                    )
                else:
                    has_validation = (
                        val_x_device is not None
                        and val_y_stage_device is not None
                        and val_mask_device is not None
                    )
                # Draw this epoch's validation subset once, and reuse it for both
                # the live-iterate and SWA-average passes so the two numbers on a
                # given line are directly comparable.
                val_x_ep = val_x_device
                val_mask_ep = val_mask_device
                val_y_stage_ep = val_y_stage_device
                val_stage_labels_ep = val_stage_labels_device
                val_y_ep = val_y
                if has_validation:
                    n_val = int(val_x_device.shape[0])
                    k_val = max(1, int(round(validation_fraction * n_val)))
                    if k_val < n_val:
                        sel = np.sort(val_subsample_rng.choice(n_val, size=k_val, replace=False))
                        sel_device = jnp.asarray(sel, dtype=jnp.int32)
                        val_x_ep = val_x_device[sel_device]
                        val_mask_ep = val_mask_device[sel_device]
                        if val_y_stage_device is not None:
                            val_y_stage_ep = val_y_stage_device[sel_device]
                        if val_stage_labels_device is not None:
                            # (records, time, stage) -- stacked with axis=-1, so the
                            # record axis is 0 here, same as the other val tensors.
                            val_stage_labels_ep = val_stage_labels_device[sel_device]
                        if val_y is not None:
                            val_y_ep = val_y[sel]
                if has_validation:
                    val_start = time.perf_counter()
                    if weighted_curriculum:
                        assert val_stage_labels_device is not None
                        assert stage_loss_weights_device is not None
                        assert weighted_eval_step is not None
                        assert weighted_run_eval_batches is not None
                        val_loss, val_logits = self._evaluate_weighted_curriculum_loss(
                            params,
                            val_x_ep,
                            val_mask_ep,
                            val_stage_labels_ep,
                            stage_loss_weights_device,
                            weighted_eval_step,
                            weighted_run_eval_batches,
                            eval_batch_size=validation_batch_size,
                        )
                    else:
                        assert val_y_stage_device is not None
                        val_loss, val_logits = self._evaluate_loss(
                            params,
                            val_x_ep,
                            val_mask_ep,
                            val_y_stage_ep,
                            eval_step,
                            run_eval_batches,
                            eval_batch_size=validation_batch_size,
                        )
                    val_elapsed = time.perf_counter() - val_start
                    history["val_loss"].append(val_loss)
                    message += f" val_loss={val_loss:.4f}"
                    if stage_idx == len(stage_maps) - 1:
                        class_metrics = masked_class_accuracies(val_logits, val_y_ep.reshape(-1), self.config.num_classes)
                        for key, value in class_metrics.items():
                            history.setdefault(f"val_{key}", []).append(value)
                        message += " " + _format_class_accuracy_metrics(class_metrics, prefix="val")
                    if stage_idx == final_stage_idx and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = params
                        if checkpoint_path is not None:
                            checkpoint_npz = checkpoint_path.with_suffix(".npz")
                            checkpoint_npz.parent.mkdir(parents=True, exist_ok=True)
                            names, arrays = self._flatten_params(best_params)
                            np.savez(checkpoint_npz, *arrays)
                            checkpoint_npz.with_suffix(".manifest.json").write_text(json.dumps(names, indent=2))
                else:
                    val_elapsed = 0.0
                    if stage_idx == final_stage_idx:
                        best_params = params

                if checkpoint_epoch_dir is not None and (epoch_idx + 1) in checkpoint_epochs:
                    # Save the live end-of-epoch weights (not best-val/SWA) so each
                    # checkpoint reflects exactly N epochs of fine-tuning. Restore
                    # self.params afterward so end-of-function bookkeeping is unchanged.
                    _saved_params = self.params
                    self.params = params
                    self.model = {"params": self.params}
                    self.savez(Path(checkpoint_epoch_dir) / f"epoch_{epoch_idx + 1:03d}.pisces")
                    self.params = _saved_params
                    self.model = {"params": self.params}

                if stage_idx == final_stage_idx and swa_start_epoch is not None and epoch_idx >= swa_start_epoch:
                    swa_params_accum.append(params)
                    # val_loss above scores `params`, the live SGD iterate -- but the
                    # model that ships is the SWA average (see `self.params =
                    # _average_param_trees(...)` at the end of this function). Those
                    # are different weights, so the printed val_loss curve says
                    # nothing about the delivered model. Track the same average
                    # incrementally and score it too.
                    #
                    # The running mean is monitor-only: the final weights still come
                    # from _average_param_trees(swa_params_accum), unchanged.
                    if swa_running_mean is None:
                        swa_running_mean = params
                    else:
                        swa_n = float(len(swa_params_accum))
                        swa_running_mean = jax.tree_util.tree_map(
                            lambda acc, new: acc + (new - acc) / swa_n,
                            swa_running_mean,
                            params,
                        )
                    if has_validation:
                        swa_val_start = time.perf_counter()
                        if weighted_curriculum:
                            swa_val_loss, _ = self._evaluate_weighted_curriculum_loss(
                                swa_running_mean,
                                val_x_ep,
                                val_mask_ep,
                                val_stage_labels_ep,
                                stage_loss_weights_device,
                                weighted_eval_step,
                                weighted_run_eval_batches,
                                eval_batch_size=validation_batch_size,
                            )
                        else:
                            swa_val_loss, _ = self._evaluate_loss(
                                swa_running_mean,
                                val_x_ep,
                                val_mask_ep,
                                val_y_stage_ep,
                                eval_step,
                                run_eval_batches,
                                eval_batch_size=validation_batch_size,
                            )
                        val_elapsed += time.perf_counter() - swa_val_start
                        history.setdefault("swa_val_loss", []).append(swa_val_loss)
                        message += f" swa_val_loss={swa_val_loss:.4f}"

                if epoch_observer is not None:
                    # Monitoring sees the live iterate without advancing any training
                    # RNG or changing best-checkpoint/SWA selection. Epochs are global
                    # across curriculum stages.
                    saved_params, saved_model = self.params, self.model
                    try:
                        self.params = params
                        self.model = {"params": params}
                        epoch_observer(self, {
                            "epoch": len(history["loss"]),
                            "stage": stage_idx + 1,
                            "stage_epoch": epoch_idx + 1,
                            "final_stage": stage_idx == final_stage_idx,
                            "weight_state": "live",
                            "stage_optimizer_step": int(state.step),
                            "steps_per_epoch": steps_per_epoch,
                            "lr_schedule": vars(stage_lr),
                            "val_logits": val_logits if has_validation else None,
                            "validation_fraction": validation_fraction,
                        }, history)
                    finally:
                        self.params, self.model = saved_params, saved_model

                epoch_elapsed = time.perf_counter() - epoch_start
                message += (
                    f" epoch_time={epoch_elapsed:.3f}s"
                    f" train_time={train_elapsed:.3f}s"
                    f" val_time={val_elapsed:.3f}s"
                )

                if train_verbose:
                    print(message)

            # "final" defers deblipping to a single pass after the fit, which the
            # caller runs; only "interleaved" deblips between curriculum stages.
            if (
                deblip_schedule == "interleaved"
                and stage_idx < final_stage_idx
                and deblip_x is not None
                and deblip_epochs > 0
            ):
                self.params = params
                self.model = {"params": self.params}
                # Deblip carries no stage labels; every epoch is masked and the
                # phase learns only from the soft targets below.
                deblip_y = np.full(deblip_x.shape[:2], -1, dtype=np.int32)
                deblip_target_logits = None
                target_x = None
                if deblip_pairing_weight > 0.0:
                    target_x = deblip_x
                    deblip_target_logits = self.soft_deblipped_target_logits(
                        target_x,
                        epoch_seconds=deblip_epoch_seconds,
                        min_wake_block_minutes=deblip_min_wake_block_minutes,
                        max_sleep_minutes=deblip_max_sleep_minutes,
                        epsilon=deblip_epsilon,
                        batch_size=batch_size,
                    )
                self.adjust_with_callbacks(
                    train_x=deblip_x,
                    train_y=deblip_y,
                    initial_lr=initial_lr,
                    weight_decay=weight_decay,
                    epochs=deblip_epochs,
                    batch_size=batch_size,
                    train_verbose=train_verbose,
                    fit_seed=fit_seed,
                    target_x=target_x,
                    target_logits=deblip_target_logits,
                    target_logit_pairing_weight=deblip_pairing_weight,
                )
                assert self.params is not None
                params = self.params

        self.params = best_params
        if swa_params_accum:
            self.params = _average_param_trees(swa_params_accum)
            if train_verbose:
                print(f"Applied SWA over {len(swa_params_accum)} final-stage checkpoints")
        self.model = {"params": self.params}
        # The plot is named after the checkpoint, so it needs one. Both the
        # path and the write stay inside the guard because callers that pass
        # save_history_plot=False (the training driver does) may pass no
        # checkpoint_path at all.
        if save_history_plot:
            if checkpoint_path is None:
                if train_verbose:
                    print("No checkpoint_path; skipping training history plot")
            else:
                history_plot_path = checkpoint_path.with_suffix(".history.png")
                _save_training_history_plot(history, history_plot_path)
                if train_verbose:
                    print(f"Saved training history plot to {history_plot_path}")
        return history

    def fit_balanced(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        eval_x: np.ndarray,
        eval_y: np.ndarray,
        *,
        batch_size: int,
        initial_lr: float,
        weight_decay: float,
        balance_beta: float,
        patience: Patience,
        max_epochs: int,
        fit_seed: int,
        lr_drop_factor: float = LR_DROP_FACTOR,
        class_sample_max_attempts: int = CLASS_SAMPLE_MAX_ATTEMPTS,
        eval_batch_size: int = 4,
        soft_targets: np.ndarray | None = None,
        score_start: bool = False,
        segment_placement: str = "start",
        shift_consistency_weight: float = 0.0,
        crop: np.ndarray | None = None,
        crop_epochs: tuple[int, int] = (0, 0),
    ) -> dict:
        """Fit with the class-balanced recipe, early-stopping on ``eval_*``'s Cohen's kappa.

        Rows are padded recordings ``(B, T, ...)`` with labels ``(B, T)`` in
        ``0..num_classes-1`` plus the gap label and ``-1`` for unscored. Each
        epoch draws as many rows as there are training rows through the
        pseudo-class-balanced sampler. A row padded to the fit length is a
        whole recording that nearly always holds every stage, so the draws are
        close to uniform and the class balance comes from the effective-number
        class-balanced cross-entropy. AdamW starts at ``initial_lr``; the rate
        is divided by ``lr_drop_factor`` and the fit stops as ``patience``
        decides, and the parameters of the epoch with the best eval kappa are
        kept. The eval kappa counts a scored epoch predicted as gap as a miss.

        With ``soft_targets`` ``(B, T, output_classes)``, the fit distills: each
        step draws whole rows uniformly and the loss is the cross-entropy against
        those probabilities over every epoch of the row, scored or not. The labels
        then serve only the eval kappa.

        With ``score_start`` the starting parameters are scored as epoch 0 and kept
        unless a later epoch has a higher eval kappa.

        With ``segment_placement="random"``, each drawn row's recording moves to a
        random epoch of the row (:func:`~arcasleep.training.balanced.place_rows`),
        drawn anew at every draw, so the model sees recordings at every position of
        the fit length. The eval rows are placed the same way once, from ``fit_seed``.
        ``segment_placement="circular"`` rotates each row by any number of epochs
        instead, so a recording can wrap from the row's end to its start;
        ``"wake-cut"`` rotates only recordings longer than half the row, and only at
        cuts with no sleep nearby (:func:`~arcasleep.training.balanced.placement_offsets`).

        With ``shift_consistency_weight`` above 0, each drawn row is placed twice,
        with offsets drawn independently. The loss is the mean of the two views'
        cross-entropies plus ``shift_consistency_weight * (1 - balance_beta)`` times
        the symmetric KL divergence between the second view's output and the first
        view's output rolled into the second view's epochs, averaged over epochs
        more than ``CONSISTENCY_EDGE_EPOCHS`` from either view's context edges. With
        ``balance_beta`` near 1, ``1 - balance_beta`` is about the cross-entropy's
        weight on an epoch of a class common in the batch. Not with ``soft_targets``
        or ``segment_placement="start"``.

        With ``crop`` ``(B,)``, each drawn row whose ``crop`` is true is replaced, before
        it is placed, by a contiguous crop of its recording whose length is drawn from
        ``crop_epochs`` (:func:`~arcasleep.training.balanced.crop_rows`), drawn anew at
        every draw. The eval rows are not cropped. Not with ``soft_targets``.
        """
        cfg = self.config
        train_x = np.asarray(train_x, dtype=np.float32)
        train_y = np.asarray(train_y, dtype=np.int32)
        eval_x = np.asarray(eval_x, dtype=np.float32)
        eval_y = np.asarray(eval_y, dtype=np.int32)
        if segment_placement not in SEGMENT_PLACEMENTS:
            raise ValueError(
                f"segment_placement must be one of {SEGMENT_PLACEMENTS}, got {segment_placement!r}"
            )
        shift_consistency_weight = float(shift_consistency_weight)
        if shift_consistency_weight < 0.0:
            raise ValueError("shift_consistency_weight must be nonnegative")
        paired = shift_consistency_weight > 0.0
        if paired and (soft_targets is not None or segment_placement == "start"):
            raise ValueError(
                "shift_consistency_weight needs labels and a segment_placement other than 'start'"
            )
        if crop is not None:
            crop = np.asarray(crop, dtype=bool)
            if crop.shape != train_y.shape[:1]:
                raise ValueError(f"crop must be {train_y.shape[:1]}, got {crop.shape}")
            if soft_targets is not None:
                raise ValueError("crop needs labels, not soft_targets")
            if not 1 <= crop_epochs[0] <= crop_epochs[1]:
                raise ValueError(f"crop_epochs must be 1 <= min <= max, got {crop_epochs}")
        place = segment_placement != "start"
        if place:
            eval_x, eval_y, _ = place_rows(
                eval_x, eval_y, np.random.default_rng([fit_seed, 1]), cfg.frames_per_epoch,
                placement=segment_placement,
            )
        self._ensure_built(tuple(train_x.shape[1:]), seed=fit_seed)
        assert self.params is not None

        rng = np.random.default_rng(fit_seed)
        rng_key = jax.random.PRNGKey(fit_seed)
        num_classes = cfg.num_classes
        output_classes = cfg.output_classes
        # With score_gap, gap and padding epochs train the extra logit; without
        # it they are masked like unscored epochs.
        gap_class = num_classes if cfg.score_gap else -1
        module = self.module

        tx = optax.inject_hyperparams(optax.adamw)(
            learning_rate=initial_lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=weight_decay
        )
        params = self.params
        opt_state = tx.init(params)

        @jax.jit
        def train_step(params, opt_state, xb, mb, yb, key):
            def loss_fn(p):
                logits = module.apply(
                    {"params": p}, xb, mb, train=True, rngs={"dropout": key, "augment": key}
                )
                return balanced_cross_entropy(logits, yb, output_classes, balance_beta)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        consistency_scale = shift_consistency_weight * (1.0 - balance_beta)

        @jax.jit
        def pair_step(params, opt_state, xa, ma, ya, xb, mb, yb, shift, keep, key):
            key_a, key_b = jax.random.split(key)

            def loss_fn(p):
                logits_a = module.apply(
                    {"params": p}, xa, ma, train=True, rngs={"dropout": key_a, "augment": key_a}
                )
                logits_b = module.apply(
                    {"params": p}, xb, mb, train=True, rngs={"dropout": key_b, "augment": key_b}
                )
                ce = 0.5 * (
                    balanced_cross_entropy(logits_a, ya, output_classes, balance_beta)
                    + balanced_cross_entropy(logits_b, yb, output_classes, balance_beta)
                )
                log_a = jax.nn.log_softmax(_roll_time_by_offsets(logits_a, shift), axis=-1)
                log_b = jax.nn.log_softmax(logits_b, axis=-1)
                divergence = 0.5 * jnp.sum((jnp.exp(log_a) - jnp.exp(log_b)) * (log_a - log_b), axis=-1)
                consistency = jnp.sum(divergence * keep) / jnp.maximum(jnp.sum(keep), 1.0)
                return ce + consistency_scale * consistency, consistency

            (loss, consistency), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss, consistency

        @jax.jit
        def distill_step(params, opt_state, xb, mb, qb, key):
            def loss_fn(p):
                logits = module.apply(
                    {"params": p}, xb, mb, train=True, rngs={"dropout": key, "augment": key}
                )
                return -(qb * jax.nn.log_softmax(logits, axis=-1)).sum(axis=-1).mean()

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        if soft_targets is not None:
            soft_targets = np.asarray(soft_targets, dtype=np.float32)
            if soft_targets.shape != (*train_y.shape, output_classes):
                raise ValueError(
                    f"soft_targets must be {(*train_y.shape, output_classes)}, got {soft_targets.shape}"
                )

        @jax.jit
        def eval_logits(params, xb, mb):
            return module.apply({"params": params}, xb, mb, train=False)

        def valid_mask(y: np.ndarray) -> np.ndarray:
            if cfg.score_gap:
                return ((y >= 0) | (y == GAP_LABEL)).astype(np.float32)
            return (y >= 0).astype(np.float32)

        eval_mask = (
            np.ones(eval_x.shape[:2], dtype=np.float32)
            if cfg.score_gap
            else _numpy_feature_activity_mask(eval_x)
        )

        def eval_kappa(params) -> float:
            cm = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
            for i in range(0, eval_x.shape[0], eval_batch_size):
                batch = slice(i, i + eval_batch_size)
                logits = eval_logits(params, jnp.asarray(eval_x[batch]), jnp.asarray(eval_mask[batch]))
                pred = np.argmax(np.asarray(logits), axis=-1)
                cm += selection_confusion(eval_y[batch], pred, num_classes)
            return cohen_kappa_from_confusion(cm)

        fpe = cfg.frames_per_epoch
        rows = [(train_x[i], train_y[i]) for i in range(train_x.shape[0])]
        steps_per_epoch = max(1, -(-len(rows) // batch_size))
        lr = float(initial_lr)
        best_kappa = -np.inf
        best_params = None
        best_epoch = -1
        plateau = Plateau(patience, steps_per_epoch)
        history: dict[str, Any] = {
            "train_loss": [], "eval_kappa": [], "smoothed_eval_kappa": [], "lr": []
        }
        if score_start:
            # Starting weights are a candidate: epoch 0, outside the patience schedule.
            best_kappa = eval_kappa(params)
            best_params = params
            best_epoch = 0
            print(f"    epoch   0  eval_kappa {best_kappa:.4f}  (starting weights)", flush=True)

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            if soft_targets is not None:
                # Whole rows, drawn uniformly; the loss covers every epoch of a row.
                drawn = rng.integers(train_x.shape[0], size=steps_per_epoch * batch_size)
                xs, targets = train_x[drawn], soft_targets[drawn]
                if place:
                    xs, _, targets = place_rows(
                        xs, train_y[drawn], rng, fpe, extra=targets, placement=segment_placement
                    )
                ms = np.ones((len(drawn), train_y.shape[1] * fpe), dtype=np.float32)
                step_fn = distill_step
            else:
                # Rows are already the fit length, so each draw is a whole row.
                xs, ys, drawn = sample_balanced_segments(
                    rows,
                    rng,
                    n_segments=steps_per_epoch * batch_size,
                    segment_epochs=train_y.shape[1],
                    num_classes=num_classes,
                    frames_per_epoch=fpe,
                    max_attempts=class_sample_max_attempts,
                    return_nights=True,
                )
                if crop is not None:
                    xs, ys = crop_rows(xs, ys, crop[drawn], rng, fpe, *crop_epochs)
                if paired:
                    offsets_a = placement_offsets(ys, rng, segment_placement)
                    offsets_b = placement_offsets(ys, rng, segment_placement)
                    xb_all, yb_all, _ = roll_rows(xs, ys, offsets_b, fpe)
                    xs, ys, _ = roll_rows(xs, ys, offsets_a, fpe)
                    shifts = (offsets_b - offsets_a) % ys.shape[1]
                    ms_b = np.repeat(valid_mask(yb_all), fpe, axis=1)
                    targets_b = np.where(yb_all <= GAP_LABEL, gap_class, yb_all)
                elif place:
                    xs, ys, _ = place_rows(xs, ys, rng, fpe, placement=segment_placement)
                # The model pools the mask with its frames, so it runs per frame.
                ms = np.repeat(valid_mask(ys), fpe, axis=1)
                targets = np.where(ys <= GAP_LABEL, gap_class, ys)
                step_fn = train_step
            epoch_loss = 0.0
            epoch_consistency = 0.0
            for step in range(steps_per_epoch):
                batch = slice(step * batch_size, (step + 1) * batch_size)
                rng_key, step_key = jax.random.split(rng_key)
                if paired:
                    params, opt_state, loss, consistency = pair_step(
                        params,
                        opt_state,
                        jnp.asarray(xs[batch]),
                        jnp.asarray(ms[batch]),
                        jnp.asarray(targets[batch]),
                        jnp.asarray(xb_all[batch]),
                        jnp.asarray(ms_b[batch]),
                        jnp.asarray(targets_b[batch]),
                        jnp.asarray(shifts[batch]),
                        jnp.asarray(consistency_keep(shifts[batch], ys.shape[1])),
                        step_key,
                    )
                    epoch_loss += float(loss)
                    epoch_consistency += float(consistency)
                    continue
                params, opt_state, loss = step_fn(
                    params,
                    opt_state,
                    jnp.asarray(xs[batch]),
                    jnp.asarray(ms[batch]),
                    jnp.asarray(targets[batch]),
                    step_key,
                )
                epoch_loss += float(loss)
            epoch_loss /= steps_per_epoch
            epoch_consistency /= steps_per_epoch

            kappa = eval_kappa(params)
            decision = plateau.update(kappa)
            history["train_loss"].append(epoch_loss)
            history["eval_kappa"].append(kappa)
            history["smoothed_eval_kappa"].append(decision.smoothed_kappa)
            history["lr"].append(lr)
            if paired:
                history.setdefault("consistency", []).append(epoch_consistency)

            if np.isfinite(kappa) and kappa > best_kappa:
                best_kappa = kappa
                best_epoch = epoch
                best_params = params

            print(
                f"    epoch {epoch:3d}  step {plateau.step}  loss {epoch_loss:.3e}  "
                + (f"consistency {epoch_consistency:.3e}  " if paired else "")
                + f"eval_kappa {kappa:.4f}  smoothed {decision.smoothed_kappa:.4f}  "
                f"best {best_kappa:.4f}@{best_epoch}  lr {lr:.2e}  "
                f"epoch_time={time.time() - epoch_start:.3f}s",
                flush=True,
            )

            if decision.stop:
                print(f"    early stop: {plateau.describe_stop()}")
                break
            if decision.drop_lr:
                lr /= lr_drop_factor
                opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
                print(f"    lr -> {lr:.2e}")

        if best_params is None:
            raise ValueError("no epoch reached a finite eval kappa; nothing to select")
        self.params = best_params
        self.model = {"params": self.params}
        history["best_eval_kappa"] = float(best_kappa)
        history["best_epoch"] = int(best_epoch)
        return history

    def adjust_with_callbacks(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        initial_lr: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        train_verbose: int = 1,
        fit_seed: int | None = None,
        focal_alpha: tuple[float, float] | None = DEFAULT_FEEDBACK_FOCAL_ALPHA,
        focal_gamma: float | None = None,
        target_x: np.ndarray | None = None,
        target_logits: np.ndarray | None = None,
        target_logit_pairing_weight: float = 0.0,
    ) -> dict[str, list[float]]:
        """Fine-tune on sparse feedback data using binary sleep/wake labels."""
        if epochs <= 0:
            return {"adjustment_loss": []}

        train_x = np.asarray(train_x, dtype=np.float32)
        train_y = np.asarray(train_y, dtype=np.int32)
        target_x = None if target_x is None else np.asarray(target_x, dtype=np.float32)
        target_logits = None if target_logits is None else np.asarray(target_logits, dtype=np.float32)
        target_logit_pairing_weight = float(target_logit_pairing_weight)
        if target_logit_pairing_weight < 0.0:
            raise ValueError("target_logit_pairing_weight must be non-negative")
        target_enabled = target_logits is not None and target_logit_pairing_weight > 0.0
        if target_enabled:
            if target_x is None:
                target_x = train_x
            if target_x.shape[1:] != train_x.shape[1:]:
                raise ValueError(
                    "target_x must have the same per-example shape as train_x; "
                    f"got target_x={target_x.shape}, train_x={train_x.shape}"
                )
            if target_logits.shape != (*target_x.shape[:2], self.config.output_classes):
                raise ValueError(
                    "target_logits must have shape "
                    f"{(*target_x.shape[:2], self.config.output_classes)}, got {target_logits.shape}"
                )
        if train_x.shape[0] == 0:
            return {"adjustment_loss": []}

        resolved_seed = _resolve_seed(fit_seed)
        self._ensure_built(tuple(train_x.shape[1:]), seed=resolved_seed)
        assert self.params is not None

        valid_feedback = (train_y == 0) | (train_y == 1) | (train_y <= -2)
        if not np.any(valid_feedback) and not target_enabled:
            if train_verbose:
                print("Skipping adjustment training: no WAKE/SLEEP feedback labels")
            return {"adjustment_loss": []}

        # Sparse -1 labels are masked out of the loss; the model still sees their
        # accelerometer context.
        train_y_feedback = np.where(valid_feedback, train_y, -1).astype(np.int32)
        train_mask = (train_y != -1).astype(np.float32)
        target_mask = (
            _numpy_feature_activity_mask(target_x)
            if target_x is not None
            else None
        )

        seed_sequence = np.random.SeedSequence(resolved_seed)
        batch_seed, rng_seed = seed_sequence.spawn(2)
        batch_rng = np.random.default_rng(batch_seed)
        train_rng_key = jax.random.PRNGKey(int(rng_seed.generate_state(1)[0]))

        train_x_device = jax.device_put(train_x)
        train_mask_device = jax.device_put(train_mask)
        train_y_device = jax.device_put(train_y_feedback)

        optimizer_examples = max(
            len(train_x),
            int(target_x.shape[0]) if target_enabled and target_x is not None else 0,
        )
        steps_per_epoch = max(math.ceil(optimizer_examples / batch_size), 1)
        # Same shared tx as the curriculum stages, so a zipper phase boundary
        # resets optimizer state without also forcing a retrace.
        state = _make_train_state(
            self.module.apply,
            self.params,
            RuntimeLearningRate.constant(initial_lr),
            weight_decay,
        )

        active_focal_alpha = focal_alpha
        active_focal_gamma = float(self.config.focal_gamma if focal_gamma is None else focal_gamma)
        # Reuse compiled adjustment functions across zipper phases; the boundary
        # resets optimizer state without forcing another JAX trace.
        adjustment_cache_key = (
            None if active_focal_alpha is None else tuple(active_focal_alpha),
            active_focal_gamma,
            target_enabled,
            target_logit_pairing_weight,
            "gate_head",
        )
        cached_steps = self._adjustment_step_cache.get(adjustment_cache_key)
        if cached_steps is None:
            train_step = _make_adjustment_train_step(
                self.module,
                active_focal_alpha,
                active_focal_gamma,
                trainable_scope="gate_head",
            )
            run_full_batches = _make_run_full_batches(train_step)
            target_train_step = (
                _make_adjustment_target_train_step(
                    self.module,
                    active_focal_alpha,
                    active_focal_gamma,
                    target_logit_pairing_weight,
                    trainable_scope="gate_head",
                )
                if target_enabled
                else None
            )
            self._adjustment_step_cache[adjustment_cache_key] = (
                train_step,
                run_full_batches,
                target_train_step,
            )
        else:
            train_step, run_full_batches, target_train_step = cached_steps

        history: dict[str, list[float]] = {"adjustment_loss": []}
        if target_enabled:
            history["adjustment_feedback_loss"] = []
            history["adjustment_target_logit_loss"] = []
        if train_verbose:
            n_labels = int(valid_feedback.sum())
            print(
                f"\nAdjustment training: {epochs} epochs on {n_labels} sparse feedback labels "
                "(trainable_scope=gate_head)"
            )
            if target_enabled:
                assert target_x is not None
                print(
                    "  deblip target-logit pairing: "
                    f"loss_weight={target_logit_pairing_weight:.3g} "
                    f"target_examples={target_x.shape[0]}"
                )
        for epoch_idx in range(epochs):
            epoch_start = time.perf_counter()
            if target_enabled:
                assert target_train_step is not None
                assert target_x is not None
                assert target_mask is not None
                assert target_logits is not None
                state, train_loss, feedback_loss, target_loss, train_rng_key = self._run_adjustment_target_epoch(
                    state,
                    train_x_device,
                    train_mask_device,
                    train_y_device,
                    target_x,
                    target_mask,
                    target_logits,
                    target_train_step,
                    batch_size,
                    batch_rng,
                    train_rng_key,
                )
            else:
                state, train_loss, train_rng_key = self._run_epoch(
                    state,
                    train_x_device,
                    train_mask_device,
                    train_y_device,
                    train_step,
                    run_full_batches,
                    batch_size,
                    batch_rng,
                    train_rng_key,
                )
                feedback_loss = train_loss
                target_loss = float("nan")
            history["adjustment_loss"].append(train_loss)
            if target_enabled:
                history["adjustment_feedback_loss"].append(feedback_loss)
                history["adjustment_target_logit_loss"].append(target_loss)
            if train_verbose:
                elapsed = time.perf_counter() - epoch_start
                message = f"adjustment epoch {epoch_idx + 1}/{epochs} loss={train_loss:.4f}"
                if target_enabled:
                    message += (
                        f" feedback_loss={feedback_loss:.4f} "
                        f"target_logit_loss={target_loss:.4f}"
                    )
                message += f" epoch_time={elapsed:.3f}s"
                print(message)

        self.params = state.params
        self.model = {"params": self.params}
        return history

    def savez(self, filepath: str | Path) -> None:
        """Save model config and weights to a .pisces archive (zip)."""
        import shutil
        import tempfile

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            metadata = {
                "model_class": self.__class__.__name__,
                "config_class": self.config.__class__.__name__,
                "config": self.config.to_dict(),
                "backend": getattr(self, "backend", "jax"),
                "is_trained": self.is_trained,
            }
            (tmpdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            weights_dir = tmpdir / "weights"
            weights_dir.mkdir()
            self.save_weights_to_directory(weights_dir)

            archive_base = filepath.with_suffix("")
            shutil.make_archive(str(archive_base), "zip", tmpdir)
            Path(str(archive_base) + ".zip").rename(filepath)

    @classmethod
    def loadz(
        cls,
        filepath: str | Path,
        input_shape: tuple[int, ...] | None = None,
        config_override: BiDozeNetConfig | None = None,
        init_seed: int | None = None,
    ) -> "CVModel":
        """Load a model from a .pisces archive (zip)."""
        import tempfile
        import zipfile

        filepath = Path(filepath)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(tmpdir)

            metadata = json.loads((tmpdir / "metadata.json").read_text())
            config = config_override or cls.config_class.from_dict(metadata["config"])
            model = cls(config=config)
            model.load_weights_from_directory(
                tmpdir / "weights", input_shape=input_shape,
                allow_partial=config_override is not None, init_seed=init_seed,
            )
            return model


if __name__ == "__main__":
    cfg = BiDozeNetConfig(num_classes=4)
    wrapper = BiDozeNetModel(cfg)
    x = np.random.randn(2, 120, 64).astype(np.float32)
    y = np.random.randint(0, 4, size=(2, 120), dtype=np.int32)
    wrapper.train_with_callbacks(x, y, x, y, initial_lr=1e-3, weight_decay=1e-4, epochs=1, batch_size=2, callbacks=[])
    logits = wrapper.infer(x)
    print(logits.shape)

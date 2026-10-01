"""A temporal U-Net sleep-stage model for accelerometer spectrograms.

The architecture is the accelerometry branch described in M. Olsen et al.,
"A flexible deep learning architecture for temporal sleep stage
classification using accelerometry and photoplethysmography," IEEE
Transactions on Biomedical Engineering, 2022, doi:10.1109/TBME.2022.3187945,
with the settings of its Table II (K=(16, 3), L=(2, 2), C_U=16, M=10).

The input is ``(B, T, F, C)``: ``T`` spectrogram frames, ``frames_per_epoch``
of them per 30 s epoch, over ``F`` frequency bins and ``C`` channels. ``T``
must be a multiple of ``2**m_blocks`` and of ``frames_per_epoch``.

- Conformation: the frequency axis is zero-padded to the next power of two.
- Encoder block m (1..M): Conv(round(2^((m-1)/3)·C_U), (16, 3)) -> GELU ->
  BatchNorm, then a strided Conv(round(2^(m/3)·C_U), (2, 2)). Once the
  frequency axis reaches 1 its stride drops to 1.
- Bottleneck: the same conv block without reduction.
- Decoder block m (M..1): ConvTranspose up, concatenate the encoder block's
  pre-downsample features, conv block.
- Classification: crop the frequency padding, flatten frequency x channels,
  Dense + GELU, average the frames of each epoch, Dense + GELU, then the
  per-epoch logits.

Training follows the paper's Section II-E: online pseudo-class-balanced
segment sampling, the effective-number class-balanced cross-entropy of its
eq. 1, and Adam with plateau learning-rate drops and early stopping on a
held-out set. With ``score_gap`` the head has one more logit, trained on gap
and padding epochs; only unscored ``-1`` epochs are masked from the loss.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.core import unfreeze

from arcasleep.data.constants import GAP_LABEL, SPECTROGRAM_PADDING_VALUE
from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion, pooled_confusion

#: The learning rate is divided by this after ``lr_drop_patience`` epochs
#: without an improvement in eval kappa.
LR_DROP_FACTOR = math.sqrt(10.0)
#: Recording draws per segment before the sampler gives up on its target class.
CLASS_SAMPLE_MAX_ATTEMPTS = 100
#: Recordings per forward pass when scoring.
EVAL_BATCH_SIZE = 4


@dataclass
class UNetConfig:
    num_classes: int = 4
    # With this `True` the net learns to predict sections that have
    # no accelerometer samples to a 5th class.
    score_gap: bool = True
    frames_per_epoch: int = 15
    c_u: int = 16
    kernel_temporal: int = 16
    kernel_freq: int = 3
    m_blocks: int = 10
    head_hidden: int = 64

    @classmethod
    def from_dict(cls, data: dict) -> UNetConfig:
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"UNetConfig: unknown keys {sorted(unknown)}")
        kwargs = {
            key: bool(value) if key == "score_gap" else int(value) for key, value in data.items()
        }
        config = cls(**kwargs)
        config.validate()
        return config

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.num_classes not in (2, 3, 4):
            raise ValueError(
                "UNetConfig.num_classes is the sleep-stage interface size; expected 2, 3, or 4"
            )
        for name in self.__dataclass_fields__:
            if name not in ("num_classes", "score_gap") and getattr(self, name) < 1:
                raise ValueError(f"UNetConfig.{name} must be at least 1")

    @property
    def output_classes(self) -> int:
        return self.num_classes + int(self.score_gap)

    @property
    def temporal_multiple(self) -> int:
        """The frame count every input must be a multiple of."""
        return 2**self.m_blocks

    @property
    def padded_epoch_multiple(self) -> int:
        """The epoch count every input must be a multiple of."""
        return self.temporal_multiple // math.gcd(self.frames_per_epoch, self.temporal_multiple)


def _next_pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def _filters(exponent_thirds: int, c_u: int) -> int:
    """``round(2^(exponent_thirds / 3) * C_U)``: filters grow by the cube root of 2."""
    return int(round(2.0 ** (exponent_thirds / 3.0) * c_u))


class UNetModule(nn.Module):
    cfg: UNetConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        cfg = self.cfg
        _, t_len, f_len, _ = x.shape
        if t_len % cfg.temporal_multiple != 0:
            raise ValueError(
                f"UNetModule: temporal length {t_len} must be a multiple of "
                f"2^{cfg.m_blocks}={cfg.temporal_multiple}"
            )
        if t_len % cfg.frames_per_epoch != 0:
            raise ValueError(
                f"UNetModule: temporal length {t_len} must be a multiple of "
                f"frames_per_epoch={cfg.frames_per_epoch}"
            )

        f_pad = _next_pow2(f_len)
        x = jnp.pad(x, ((0, 0), (0, 0), (0, f_pad - f_len), (0, 0)))

        freq_strides: list[int] = []
        f_dim = f_pad
        for _ in range(cfg.m_blocks):
            stride = 2 if f_dim > 1 else 1
            freq_strides.append(stride)
            f_dim = -(-f_dim // stride)

        kernel = (cfg.kernel_temporal, cfg.kernel_freq)

        def conv_block(h: jnp.ndarray, features: int, name: str) -> jnp.ndarray:
            h = nn.Conv(features, kernel, padding="SAME", name=f"{name}_conv")(h)
            h = nn.gelu(h)
            return nn.BatchNorm(use_running_average=not train, name=f"{name}_bn")(h)

        skips: list[jnp.ndarray] = []
        h = x
        for m in range(1, cfg.m_blocks + 1):
            h = conv_block(h, _filters(m - 1, cfg.c_u), name=f"enc{m}")
            skips.append(h)
            fs = freq_strides[m - 1]
            h = nn.Conv(
                _filters(m, cfg.c_u), (2, fs), strides=(2, fs), padding="SAME", name=f"enc{m}_down"
            )(h)

        h = conv_block(h, _filters(cfg.m_blocks, cfg.c_u), name="bottleneck")

        for m in range(cfg.m_blocks, 0, -1):
            fs = freq_strides[m - 1]
            h = nn.ConvTranspose(
                _filters(m - 1, cfg.c_u),
                (2, fs),
                strides=(2, fs),
                padding="SAME",
                name=f"dec{m}_up",
            )(h)
            h = jnp.concatenate([h, skips[m - 1]], axis=-1)
            h = conv_block(h, _filters(m - 1, cfg.c_u), name=f"dec{m}")

        h = h[:, :, :f_len, :]
        h = h.reshape(h.shape[0], h.shape[1], -1)
        h = nn.gelu(nn.Dense(cfg.head_hidden, name="head_pre_pool")(h))
        if cfg.frames_per_epoch > 1:
            h = h.reshape(
                h.shape[0], h.shape[1] // cfg.frames_per_epoch, cfg.frames_per_epoch, h.shape[2]
            ).mean(axis=2)
        h = nn.gelu(nn.Dense(cfg.head_hidden, name="head_post_pool")(h))
        return nn.Dense(cfg.output_classes, name="head_logits")(h)


def balanced_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    beta: float,
) -> jnp.ndarray:
    """Effective-number class-balanced cross-entropy (Olsen et al., eq. 1).

    Class counts are the batch's validly labelled epochs; labels outside
    ``0..num_classes-1`` are masked. The sum is normalised by every epoch in
    the batch, masked ones included.
    """
    valid = (labels >= 0) & (labels < num_classes)
    safe_labels = jnp.where(valid, labels, 0)
    counts = jnp.zeros(num_classes).at[safe_labels].add(valid.astype(jnp.float32))
    class_weights = jnp.where(counts > 0, (1.0 - beta) / (1.0 - jnp.power(beta, counts)), 0.0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
    weighted = class_weights[safe_labels] * ce * valid.astype(jnp.float32)
    return jnp.sum(weighted) / float(labels.size)


def sample_balanced_segments(
    nights: Sequence[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    n_segments: int,
    segment_epochs: int,
    num_classes: int,
    frames_per_epoch: int,
    max_attempts: int = CLASS_SAMPLE_MAX_ATTEMPTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-class-balanced online sampling (Olsen et al., Section II-E).

    Each segment draws a target stage uniformly, then draws (recording, start)
    pairs until the segment holds at least one epoch of it; ``max_attempts``
    bounds the search for a stage a short night may lack. Nights are
    ``(frames, labels)`` with ``frames_per_epoch`` frames per label. Segments
    shorter than ``segment_epochs`` are padded with absent frames and the gap
    label.
    """
    fpe = frames_per_epoch
    f_bins, channels = nights[0][0].shape[1:]
    xs = np.full(
        (n_segments, segment_epochs * fpe, f_bins, channels),
        SPECTROGRAM_PADDING_VALUE,
        dtype=np.float32,
    )
    ys = np.full((n_segments, segment_epochs), GAP_LABEL, dtype=np.int32)
    for i in range(n_segments):
        target_class = int(rng.integers(num_classes))
        for _ in range(max_attempts):
            night_x, night_y = nights[int(rng.integers(len(nights)))]
            start = int(rng.integers(max(night_y.shape[0] - segment_epochs, 0) + 1))
            seg_x = night_x[start * fpe : (start + segment_epochs) * fpe]
            seg_y = night_y[start : start + segment_epochs]
            if np.any(seg_y == target_class):
                break
        xs[i, : seg_x.shape[0]] = seg_x
        ys[i, : seg_y.shape[0]] = seg_y
    return xs, ys


def pad_nights(
    nights: Sequence[tuple[np.ndarray, np.ndarray]],
    n_epochs: int,
    frames_per_epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack whole nights, padded with absent frames and the gap label to ``n_epochs``."""
    f_bins, channels = nights[0][0].shape[1:]
    xs = np.full(
        (len(nights), n_epochs * frames_per_epoch, f_bins, channels),
        SPECTROGRAM_PADDING_VALUE,
        dtype=np.float32,
    )
    ys = np.full((len(nights), n_epochs), GAP_LABEL, dtype=np.int32)
    for i, (night_x, night_y) in enumerate(nights):
        xs[i, : night_x.shape[0]] = night_x
        ys[i, : night_y.shape[0]] = night_y
    return xs, ys


class UNetModel:
    config_class: ClassVar[type[UNetConfig]] = UNetConfig

    def __init__(self, config: UNetConfig):
        self.config = config
        self.module = UNetModule(config)
        self.variables: dict[str, Any] | None = None
        self.input_shape: tuple[int, ...] | None = None
        self._apply = jax.jit(lambda variables, x: self.module.apply(variables, x, train=False))

    @property
    def is_trained(self) -> bool:
        return self.variables is not None

    def returns_logits(self) -> bool:
        return True

    def _init_variables(self, rng: jax.Array, input_shape: tuple[int, ...]) -> dict[str, Any]:
        sample = jnp.zeros((1, *input_shape), dtype=jnp.float32)
        return unfreeze(self.module.init(rng, sample, train=False))

    def _variable_shapes(self, input_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Every variable's flat name and shape, without running the network."""
        sample = jax.ShapeDtypeStruct((1, *input_shape), jnp.float32)
        shapes = jax.eval_shape(
            lambda x: self.module.init(jax.random.PRNGKey(0), x, train=False), sample
        )
        flat = traverse_util.flatten_dict(unfreeze(shapes), sep="/")
        return {name: tuple(value.shape) for name, value in flat.items()}

    def parameter_count(self) -> int:
        if self.variables is None:
            raise ValueError("an unbuilt model has no parameters")
        return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(self.variables["params"]))

    def infer(self, input_x: np.ndarray, batch_size: int = EVAL_BATCH_SIZE) -> np.ndarray:
        """Logits ``(B, T // frames_per_epoch, output_classes)`` for ``(B, T, F, C)``."""
        if self.variables is None:
            raise ValueError("UNetModel: the model is untrained")
        input_x = np.asarray(input_x, dtype=np.float32)
        outputs = [
            np.asarray(self._apply(self.variables, jnp.asarray(input_x[i : i + batch_size])))
            for i in range(0, input_x.shape[0], batch_size)
        ]
        return np.concatenate(outputs, axis=0).astype(np.float32)

    # -- persistence --

    def _flat_variables(self) -> dict[str, np.ndarray]:
        if self.variables is None or self.input_shape is None:
            raise ValueError("cannot save an unbuilt model")
        flat = traverse_util.flatten_dict(self.variables, sep="/")
        return {name: np.asarray(flat[name]) for name in sorted(flat)}

    def _restore(self, arrays: dict[str, np.ndarray], input_shape: tuple[int, ...]) -> None:
        """Set the variables from flat ``params/...`` and ``batch_stats/...`` arrays.

        The arrays must match, name for name and shape for shape, the network
        this config builds.
        """
        expected = self._variable_shapes(input_shape)
        found = {name: tuple(array.shape) for name, array in arrays.items()}
        if found != expected:
            missing = sorted(set(expected) - set(found))
            extra = sorted(set(found) - set(expected))
            mismatched = sorted(
                name for name in set(found) & set(expected) if found[name] != expected[name]
            )
            raise ValueError(
                "weights do not match the network this config builds: "
                f"missing {missing[:3]}, unexpected {extra[:3]}, mismatched {mismatched[:3]}"
            )
        flat = {tuple(name.split("/")): jnp.asarray(array) for name, array in arrays.items()}
        self.variables = traverse_util.unflatten_dict(flat)
        self.input_shape = tuple(input_shape)

    def savez(self, filepath: str | Path) -> None:
        """Save config and variables to a ``.pisces`` archive (zip)."""
        import io
        import zipfile

        arrays = self._flat_variables()
        names = list(arrays)
        weights = io.BytesIO()
        np.savez(weights, *(arrays[name] for name in names))
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_class": type(self).__name__,
            "config_class": type(self.config).__name__,
            "config": self.config.to_dict(),
            "backend": "jax",
            "is_trained": self.is_trained,
        }
        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            archive.writestr("weights/model_weights.npz", weights.getvalue())
            archive.writestr("weights/input_shape.json", json.dumps(list(self.input_shape)))
            archive.writestr("weights/weight_manifest.json", json.dumps(names, indent=2))

    @classmethod
    def loadz(
        cls, filepath: str | Path, input_shape: tuple[int, ...] | None = None
    ) -> UNetModel:
        """Load a model saved by :meth:`savez`.

        ``input_shape`` defaults to the one the model was saved with; any
        shape with the same frequency and channel dims loads the same weights.
        """
        import io
        import zipfile

        with zipfile.ZipFile(filepath) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            names = json.loads(archive.read("weights/weight_manifest.json"))
            saved_shape = tuple(json.loads(archive.read("weights/input_shape.json")))
            with np.load(io.BytesIO(archive.read("weights/model_weights.npz"))) as data:
                arrays = [data[f"arr_{i}"] for i in range(len(names))]
        model = cls(cls.config_class.from_dict(metadata["config"]))
        model._restore(dict(zip(names, arrays, strict=True)), tuple(input_shape or saved_shape))
        return model

    def save_weights_safetensors(self, path: Path) -> None:
        """Write the variables to a ``safetensors`` file.

        The release format: a flat ``{name: tensor}`` map over both the
        ``params`` and the BatchNorm ``batch_stats`` collections. Metadata
        carries the input shape.
        """
        try:
            from safetensors.numpy import save_file
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via CLI
            raise ModuleNotFoundError(
                "saving safetensors needs the optional dependency: "
                "pip install 'arcasleep[publish]'",
                name="safetensors",
            ) from exc
        tensors = {
            name: np.ascontiguousarray(array) for name, array in self._flat_variables().items()
        }
        save_file(tensors, str(path), metadata={"input_shape": json.dumps(list(self.input_shape))})

    def load_weights_safetensors(
        self, path: Path, input_shape: tuple[int, ...] | None = None
    ) -> None:
        """Restore the variables from ``safetensors``."""
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
            arrays = {name: handle.get_tensor(name) for name in handle.keys()}
        self._restore(arrays, tuple(input_shape))

    # -- training --

    def fit(
        self,
        train_nights: Sequence[tuple[np.ndarray, np.ndarray]],
        eval_nights: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        segment_epochs: int,
        batch_size: int,
        initial_lr: float,
        balance_beta: float,
        lr_drop_patience: int,
        early_stop_patience: int,
        max_epochs: int,
        fit_seed: int,
        lr_drop_factor: float = LR_DROP_FACTOR,
        class_sample_max_attempts: int = CLASS_SAMPLE_MAX_ATTEMPTS,
        eval_batch_size: int = EVAL_BATCH_SIZE,
    ) -> dict:
        """Fit on ``train_nights``, early-stopping on ``eval_nights``' Cohen's kappa.

        Nights are ``(frames, labels)`` pairs at their own length, labels in
        ``0..num_classes-1`` plus the gap label and ``-1`` for unscored. Each
        epoch draws one batch of balanced segments per ``batch_size`` training
        nights. Adam starts at ``initial_lr``; the rate is divided by
        ``lr_drop_factor`` after ``lr_drop_patience`` epochs without an eval
        improvement, training stops after ``early_stop_patience``, and the
        best epoch's variables are kept.
        """
        cfg = self.config
        fpe = cfg.frames_per_epoch
        if (segment_epochs * fpe) % cfg.temporal_multiple != 0:
            raise ValueError(
                f"segment_epochs*frames_per_epoch={segment_epochs * fpe} must be "
                f"a multiple of {cfg.temporal_multiple}"
            )
        for night_x, night_y in [*train_nights, *eval_nights]:
            if night_x.shape[0] != night_y.shape[0] * fpe:
                raise ValueError(
                    f"night has {night_x.shape[0]} frames for {night_y.shape[0]} "
                    f"epochs; expected frames_per_epoch={fpe}"
                )
        rng = np.random.default_rng(fit_seed)
        input_shape = (segment_epochs * fpe, *train_nights[0][0].shape[1:])
        if self.variables is None:
            self.variables = self._init_variables(jax.random.PRNGKey(fit_seed), input_shape)
            self.input_shape = input_shape

        # inject_hyperparams keeps the learning rate in the optimizer state, so
        # a plateau drop changes data rather than retracing the train step.
        tx = optax.inject_hyperparams(optax.adam)(
            learning_rate=initial_lr, b1=0.9, b2=0.999, eps=1e-8
        )
        params = self.variables["params"]
        batch_stats = self.variables["batch_stats"]
        opt_state = tx.init(params)

        module = self.module
        num_classes = cfg.num_classes
        output_classes = cfg.output_classes
        # With score_gap, gap and padding epochs train the extra logit; without
        # it they are masked like unscored epochs.
        gap_class = cfg.num_classes if cfg.score_gap else -1

        @jax.jit
        def train_step(params, batch_stats, opt_state, xb, yb):
            def loss_fn(p):
                logits, updates = module.apply(
                    {"params": p, "batch_stats": batch_stats},
                    xb,
                    train=True,
                    mutable=["batch_stats"],
                )
                loss = balanced_cross_entropy(logits, yb, output_classes, balance_beta)
                return loss, updates["batch_stats"]

            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_batch_stats, opt_state, loss

        # Every eval night is padded to one length, so scoring compiles once.
        eval_epochs = max(night_y.shape[0] for _, night_y in eval_nights)
        eval_epochs = -(-eval_epochs // cfg.padded_epoch_multiple) * cfg.padded_epoch_multiple
        eval_x, eval_y = pad_nights(eval_nights, eval_epochs, fpe)

        def eval_kappa(params, batch_stats) -> float:
            variables = {"params": params, "batch_stats": batch_stats}
            cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            for i in range(0, eval_x.shape[0], eval_batch_size):
                logits = self._apply(variables, jnp.asarray(eval_x[i : i + eval_batch_size]))
                pred = np.argmax(np.asarray(logits), axis=-1)
                cm += pooled_confusion(eval_y[i : i + eval_batch_size], pred, num_classes)
            return cohen_kappa_from_confusion(cm)

        steps_per_epoch = max(1, -(-len(train_nights) // batch_size))
        lr = float(initial_lr)
        best_kappa = -np.inf
        best_variables = None
        best_epoch = -1
        epochs_since_best = 0
        epochs_since_lr_drop = 0
        history: dict[str, Any] = {"train_loss": [], "eval_kappa": [], "lr": []}

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            xs, ys = sample_balanced_segments(
                train_nights,
                rng,
                n_segments=steps_per_epoch * batch_size,
                segment_epochs=segment_epochs,
                num_classes=num_classes,
                frames_per_epoch=fpe,
                max_attempts=class_sample_max_attempts,
            )
            ys = np.where(ys <= GAP_LABEL, gap_class, ys)
            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                batch = slice(step * batch_size, (step + 1) * batch_size)
                params, batch_stats, opt_state, loss = train_step(
                    params, batch_stats, opt_state, jnp.asarray(xs[batch]), jnp.asarray(ys[batch])
                )
                epoch_loss += float(loss)
            epoch_loss /= steps_per_epoch

            kappa = eval_kappa(params, batch_stats)
            history["train_loss"].append(epoch_loss)
            history["eval_kappa"].append(kappa)
            history["lr"].append(lr)

            if np.isfinite(kappa) and kappa > best_kappa:
                best_kappa = kappa
                best_epoch = epoch
                best_variables = jax.device_get({"params": params, "batch_stats": batch_stats})
                epochs_since_best = 0
                epochs_since_lr_drop = 0
            else:
                epochs_since_best += 1
                epochs_since_lr_drop += 1

            print(
                f"    epoch {epoch:3d}  loss {epoch_loss:.3e}  eval_kappa {kappa:.4f}  "
                f"best {best_kappa:.4f}@{best_epoch}  lr {lr:.2e}  "
                f"epoch_time={time.time() - epoch_start:.3f}s",
                flush=True,
            )

            if epochs_since_best >= early_stop_patience:
                print(f"    early stop: no eval improvement in {early_stop_patience} epochs")
                break
            if epochs_since_lr_drop >= lr_drop_patience:
                lr /= lr_drop_factor
                opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
                epochs_since_lr_drop = 0
                print(f"    lr -> {lr:.2e}")

        if best_variables is None:
            best_variables = jax.device_get({"params": params, "batch_stats": batch_stats})
        self.variables = unfreeze(best_variables)
        history["best_eval_kappa"] = float(best_kappa)
        history["best_epoch"] = int(best_epoch)
        return history

"""Strands that differ in attention type, with the mean or a fusion transformer over their outputs.

A strand is one CNN plus forward/reverse transformer streams plus a 5-way head, as in
``cooperating_arms/model.py``. Each strand's ``attention`` is ``"bicausal"`` (the forward stream
sees only earlier epochs, the reverse stream only later ones) or ``"full"`` (both streams see every
present epoch); the two have the same weights. All attention uses ALiBi and hides absent epochs
(gap row = 1) as keys; every head's output is gap exactly where the epoch is absent.

Without ``fusion`` the model's output is the mean of the strands' probabilities, trained alone.
With ``fusion`` the output is the fusion head's: per epoch each strand's penultimate features
(GELU of its ``head_hidden`` layer) are projected to ``fusion_proj`` dims, a learned strand
embedding is added, the projections are concatenated and mapped to ``fusion_width``, and
``fusion_layers`` full-attention ALiBi transformer layers over epochs precede a 5-way head. The
loss is the class-balanced cross-entropy of the fusion head plus ``strand_loss_weight`` times the
same loss of each strand head.
"""
from dataclasses import dataclass
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cooperating_arms"))

from arcasleep.training.balanced import balanced_cross_entropy  # noqa: E402
from arcasleep_bidoze.model import BiDozeNetConfig  # noqa: E402
from arcasleep_bidoze.model.definition import (  # noqa: E402
    ConvBankModule, TransformerBlockModule, GatedFusionModule,
    _causal_mask, _reverse_causal_mask, _flatten_timestep_features,
    bypass_learned_normalization, present_from_gap_row, with_fixed_gap,
    GLOROT_UNIFORM, ZEROS, LAYER_NORM_EPS, _gelu,
)
from model import CooperatingModel  # noqa: E402  (cooperating_arms/model.py)

G3_PARAMETERS = 6_709_644  # three BiDoze-G seeds, 2,236,548 each
ATTENTION = ("bicausal", "full")


@dataclass(frozen=True)
class StrandSpec:
    width: int = 124
    ffn: int = 0  # transformer feed-forward width of a strand; 0 means ``width``
    attention: tuple[str, ...] = ("bicausal",) * 3
    feature_dropout: float = 0.2
    fusion: bool = False
    fusion_proj: int = 64
    fusion_width: int = 128
    fusion_layers: int = 2
    fusion_heads: int = 4
    fusion_ffn: int = 256
    strand_loss_weight: float = 0.3

    def __post_init__(self):
        object.__setattr__(self, "attention", tuple(self.attention))
        if not self.attention or any(a not in ATTENTION for a in self.attention):
            raise ValueError(f"strand attention must be in {ATTENTION}")
        if self.width < 4 or self.width % 4:
            raise ValueError("strand width divisible by four required")

    @property
    def members(self) -> int:
        return len(self.attention)

    @property
    def strand_ffn(self) -> int:
        return self.ffn or self.width


class FusionTransformer(nn.Module):
    spec: StrandSpec
    cfg: BiDozeNetConfig

    @nn.compact
    def __call__(self, features, present, train=False):
        s = self.spec
        embed = self.param("strand_embedding", nn.initializers.normal(0.02), (s.members, s.fusion_proj))
        h = jnp.concatenate([
            nn.Dense(s.fusion_proj, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name=f"project_{i}")(f) + embed[i]
            for i, f in enumerate(features)], axis=-1)
        h = nn.Dense(s.fusion_width, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="mix")(h)
        hidden = ~present[:, None, :]
        for layer in range(s.fusion_layers):
            h = TransformerBlockModule(s.fusion_width, s.fusion_heads, s.fusion_ffn, self.cfg.dropout,
                                       self.cfg.attn_dropout, name=f"layer_{layer}")(h, causal_mask=hidden, train=train)
        h = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name="norm")(h)
        h = nn.Dropout(self.cfg.dropout, name="head_drop")(h, deterministic=not train)
        return nn.Dense(self.cfg.num_classes, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name="head_logits")(h)


class StrandFusionModule(nn.Module):
    cfg: BiDozeNetConfig
    spec: StrandSpec

    @nn.compact
    def __call__(self, x, valid_mask=None, train=False, return_outputs=False):
        cfg, s = self.cfg, self.spec
        if cfg.frames_per_epoch != 1 or not cfg.gap_from_row:
            raise ValueError("this experiment requires pooled epochs and the fixed gap rule")
        present = present_from_gap_row(x)
        mask = present.astype(x.dtype)
        if valid_mask is not None:
            mask = mask * valid_mask
        x = bypass_learned_normalization(_flatten_timestep_features(x), valid_mask=mask)
        length = x.shape[1]
        locals_ = []
        for i in range(s.members):
            h = ConvBankModule(s.width, cfg.conv_kernels, cfg.dropout, name=f"arm_{i}_cnn")(x, train=train)
            locals_.append(nn.Dropout(s.feature_dropout, broadcast_dims=(1,),
                                      name=f"arm_{i}_feature_drop")(h, deterministic=not train))
        d = s.width // 2
        absent = ~present[:, None, :]
        outputs = []
        for direction, causal in [("fwd", _causal_mask(length)), ("rev", _reverse_causal_mask(length))]:
            states = []
            for i, h in enumerate(locals_):
                h = nn.Dense(d, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name=f"arm_{i}_{direction}_in")(h)
                hidden = causal[None] | absent if s.attention[i] == "bicausal" else absent
                for layer in range(cfg.n_transformer_layers):
                    h = TransformerBlockModule(d, cfg.n_heads // 2, s.strand_ffn, cfg.dropout, cfg.attn_dropout,
                                               name=f"arm_{i}_{direction}_{layer}")(h, causal_mask=hidden, train=train)
                states.append(h)
            outputs.append(states)
        logps, features = [], []
        for i in range(s.members):
            h = jnp.concatenate([outputs[0][i], outputs[1][i]], axis=-1)
            h = nn.Dense(s.width, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name=f"arm_{i}_out")(h)
            h = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name=f"arm_{i}_out_norm")(h)
            h = GatedFusionModule(s.width, name=f"arm_{i}_fusion")(locals_[i], h)
            h = _gelu(nn.Dense(s.width // 2, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS, name=f"arm_{i}_head_hidden")(h))
            features.append(h)
            h = nn.Dropout(cfg.dropout, name=f"arm_{i}_head_drop")(h, deterministic=not train)
            logits = nn.Dense(cfg.num_classes, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS,
                              name=f"arm_{i}_head_logits")(h)
            logps.append(jax.nn.log_softmax(with_fixed_gap(logits, present), axis=-1))
        logps = jnp.stack(logps)
        log_mean = jax.nn.logsumexp(logps, axis=0) - jnp.log(s.members)
        if s.fusion:
            logits = FusionTransformer(s, cfg, name="fusion")(features, present, train=train)
            log_head = jax.nn.log_softmax(with_fixed_gap(logits, present), axis=-1)
        else:
            log_head = log_mean
        if not return_outputs:
            return log_head
        return {"log_head": log_head, "log_mean": log_mean, "strand_log_probabilities": logps,
                "strand_probabilities": jnp.exp(logps)}


class StrandFusionModel(CooperatingModel):
    """Research model; weights live in a directory beside ``config.json`` (spec is a StrandSpec)."""

    def __init__(self, config, spec):
        super().__init__(config, spec)
        self.module = StrandFusionModule(config, spec)

    def _balanced_loss(self, params, x, mask, targets, key, output_classes, beta, weights=None):
        s = self.spec
        out = self.module.apply({"params": params}, x, mask, train=True, return_outputs=True,
                                rngs={"dropout": key, "augment": key})
        loss = balanced_cross_entropy(out["log_head"], targets, output_classes, beta, weights)
        if s.fusion:
            for i in range(s.members):
                loss = loss + s.strand_loss_weight * balanced_cross_entropy(
                    out["strand_log_probabilities"][i], targets, output_classes, beta, weights)
        return loss

    @classmethod
    def load_research(cls, directory):
        import json
        directory = Path(directory)
        metadata = json.loads((directory / "config.json").read_text())
        model = cls(BiDozeNetConfig.from_dict(metadata["config"]), StrandSpec(**metadata["spec"]))
        model.load_weights_from_directory(directory / "weights")
        return model


def model_config(spec: StrandSpec) -> BiDozeNetConfig:
    return BiDozeNetConfig(num_classes=4, score_gap=True, d_model=spec.width, n_heads=4, d_ff=2 * spec.width,
                           n_transformer_layers=4, conv_kernels=(3, 7, 31, 63), dropout=0.2, attn_dropout=0,
                           gap_from_row=True)


def count_parameters(spec: StrandSpec) -> int:
    module = StrandFusionModule(model_config(spec), spec)
    x = jnp.zeros((1, 8, 61, 2))
    params = jax.eval_shape(lambda: module.init(jax.random.PRNGKey(0), x, jnp.ones((1, 8)))["params"])
    return int(sum(l.size for l in jax.tree.leaves(params)))


# Widths put each arm's total within 2% of its target: F0 for F1-F3, G x3 for F4 (see tests).
MIXED = ("full", "bicausal", "bicausal")
ARMS = {  # name: (label, StrandSpec fields)
    "final3": ("F0", dict(width=124)),
    "final3_mixed": ("F1", dict(width=124, attention=MIXED)),
    "final3_fuse": ("F2", dict(width=112, fusion=True)),
    "final3_mixed_fuse": ("F3", dict(width=112, fusion=True, attention=MIXED)),
    "final3_large": ("F4", dict(width=256)),
}


def arm_spec(name: str) -> StrandSpec:
    return StrandSpec(**ARMS[name][1])

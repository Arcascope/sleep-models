"""Research-only BiDoze arms: one supervised loss on mean probabilities."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel
from arcasleep_bidoze.model.definition import (
    ConvBankModule, TransformerBlockModule, GatedFusionModule,
    _causal_mask, _reverse_causal_mask, _flatten_timestep_features,
    bypass_learned_normalization, present_from_gap_row, with_fixed_gap,
    GLOROT_UNIFORM, ZEROS, LAYER_NORM_EPS, _gelu,
)


@dataclass(frozen=True)
class ArmSpec:
    members: int = 7
    width: int = 68
    shared_cnn: bool = False
    cnn_width: int = 256
    exchange_after: tuple[int, ...] = ()
    exchange_rank: int = 16
    exchange_scale: float = 0.1
    feature_dropout: float = 0.2

    def __post_init__(self):
        if self.members < 1 or self.width < 4 or self.width % 4:
            raise ValueError("positive arm count and width divisible by four required")
        if self.exchange_after and self.members < 2:
            raise ValueError("exchange needs multiple arms")


class ResidualExchange(nn.Module):
    members: int
    width: int
    rank: int
    scale: float

    @nn.compact
    def __call__(self, states):
        written = [nn.Dense(self.rank, name=f"write_{i}")(
            nn.LayerNorm(name=f"norm_{i}")(h)) for i, h in enumerate(states)]
        # Leave-self-out keeps this an explicit between-arm connection.
        total = sum(written)
        return [h + self.scale * nn.Dense(self.width, name=f"read_{i}")(
            (total - written[i]) / (self.members - 1)) for i, h in enumerate(states)]


class CooperatingModule(nn.Module):
    cfg: BiDozeNetConfig
    spec: ArmSpec

    @nn.compact
    def __call__(self, x, valid_mask=None, train=False, return_outputs=False,
                 exchange_enabled=True):
        cfg, s = self.cfg, self.spec
        if cfg.frames_per_epoch != 1 or not cfg.gap_from_row:
            raise ValueError("this experiment requires pooled epochs and the fixed gap rule")
        present = present_from_gap_row(x)
        mask = present.astype(x.dtype)
        if valid_mask is not None:
            mask = mask * valid_mask
        x = bypass_learned_normalization(_flatten_timestep_features(x), valid_mask=mask)
        if s.shared_cnn:
            shared = ConvBankModule(s.cnn_width, cfg.conv_kernels, cfg.dropout,
                                    name="shared_cnn")(x, train=train)
        locals_ = []
        for i in range(s.members):
            if s.shared_cnn:
                h = nn.Dense(s.width, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS,
                             name=f"arm_{i}_input")(shared)
            else:
                h = ConvBankModule(s.width, cfg.conv_kernels, cfg.dropout,
                                   name=f"arm_{i}_cnn")(x, train=train)
            # Different module paths draw independent masks, shared over time per row.
            h = nn.Dropout(s.feature_dropout, broadcast_dims=(1,),
                           name=f"arm_{i}_feature_drop")(h, deterministic=not train)
            locals_.append(h)
        d = s.width // 2
        outputs = []
        for direction, causal in [("fwd", _causal_mask(x.shape[1])),
                                  ("rev", _reverse_causal_mask(x.shape[1]))]:
            hidden = causal[None] | ~present[:, None, :]
            states = [nn.Dense(d, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS,
                               name=f"arm_{i}_{direction}_in")(h)
                      for i, h in enumerate(locals_)]
            for layer in range(cfg.n_transformer_layers):
                states = [TransformerBlockModule(
                    d, cfg.n_heads // 2, s.width, cfg.dropout, cfg.attn_dropout,
                    name=f"arm_{i}_{direction}_{layer}")(h, causal_mask=hidden, train=train)
                    for i, h in enumerate(states)]
                if layer + 1 in s.exchange_after:
                    mixed = ResidualExchange(s.members, d, s.exchange_rank,
                                             s.exchange_scale,
                                             name=f"exchange_{direction}_{layer}")(states)
                    if exchange_enabled:
                        states = mixed
            outputs.append(states)
        logps = []
        for i in range(s.members):
            h = jnp.concatenate([outputs[0][i], outputs[1][i]], axis=-1)
            h = nn.Dense(s.width, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS,
                         name=f"arm_{i}_out")(h)
            h = nn.LayerNorm(epsilon=LAYER_NORM_EPS, name=f"arm_{i}_out_norm")(h)
            h = GatedFusionModule(s.width, name=f"arm_{i}_fusion")(locals_[i], h)
            h = nn.Dense(s.width // 2, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS,
                         name=f"arm_{i}_head_hidden")(h)
            h = nn.Dropout(cfg.dropout, name=f"arm_{i}_head_drop")(_gelu(h), deterministic=not train)
            logits = nn.Dense(cfg.num_classes, kernel_init=GLOROT_UNIFORM, bias_init=ZEROS,
                              name=f"arm_{i}_head_logits")(h)
            logps.append(jax.nn.log_softmax(with_fixed_gap(logits, present), axis=-1))
        logps = jnp.stack(logps)
        mean = jax.nn.logsumexp(logps, axis=0) - jnp.log(s.members)
        return {"log_mean": mean, "arm_probabilities": jnp.exp(logps)} if return_outputs else mean


class CooperatingModel(BiDozeNetModel):
    def __init__(self, config, spec):
        super().__init__(config)
        self.spec = spec
        self.module = CooperatingModule(config, spec)

    # Inherited balanced loss scores only module output, the log of the probability mean.
    # Keep the architecture beside the weight directory; never masquerade as a plain .pisces.

"""Several BiDoze networks trained as one: the training loss is applied to their mean output.

A :class:`JointEnsemble` holds ``members`` copies of the BiDoze network with their weights
stacked on a leading axis, runs them with ``jax.vmap`` and returns the log of their mean 5-way
softmax, the output :class:`arcasleep_bidoze.adult.StackedModel` gives for separately trained
seeds. ``fit_balanced`` therefore applies the class-balanced loss to that mean, and early-stops
on the mean's eval κ. Member m starts from the weights a separate fit of seed ``fit_seed + m``
starts from, and draws its own dropout.

``savez`` writes one ordinary BiDoze checkpoint per member, so the members load and score like
separately trained seeds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from arcasleep.training.balanced import balanced_cross_entropy
from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel


class JointEnsemble(BiDozeNetModel):
    """``member_loss_weight`` β anchors each member: the loss is (1 − β) times the loss of the
    members' mean output plus β times the mean of each member's own loss. β = 0 trains the mean
    alone; β = 1 trains each member on its own output, in one fit."""

    def __init__(self, config: BiDozeNetConfig, members: int, member_loss_weight: float = 0.0):
        super().__init__(config)
        if members < 2:
            raise ValueError("a joint ensemble needs at least 2 members")
        if not 0.0 <= member_loss_weight <= 1.0:
            raise ValueError("member_loss_weight must be in [0, 1]")
        if config.architecture != "bidoze":
            raise ValueError("a joint ensemble's members are single BiDoze networks")
        self.members = members
        self.member_loss_weight = float(member_loss_weight)

    def _build_model(self, input_shape: tuple[int, ...]) -> FrozenDict[str, Any]:
        trees = []
        for m in range(self.members):
            member = BiDozeNetModel(self.config)
            member._build_seed = self._build_seed + m
            trees.append(member._build_model(input_shape))
        self.input_shape = tuple(input_shape)
        self.params = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *trees)
        self.model = {"params": self.params}
        return self.params

    def _apply(
        self,
        params: FrozenDict[str, Any],
        x: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        train: bool,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """The log of the members' mean softmax, which a softmax or log-softmax leaves unchanged."""
        log_proba = self._member_log_probabilities(params, x, mask, train=train, key=key)
        return jax.nn.logsumexp(log_proba, axis=0) - jnp.log(self.members)

    def _member_log_probabilities(self, params, x, mask, *, train: bool, key: jax.Array | None) -> jnp.ndarray:
        """``(members, B, T, classes)``: each member's log softmax."""
        single = BiDozeNetModel._apply
        if key is None:
            logits = jax.vmap(lambda p: single(self, p, x, mask, train=train))(params)
        else:
            keys = jax.random.split(key, self.members)
            logits = jax.vmap(lambda p, k: single(self, p, x, mask, train=train, key=k))(params, keys)
        return jax.nn.log_softmax(logits, axis=-1)

    def _balanced_loss(self, params, x, mask, targets, key, output_classes, beta, weights=None):
        log_proba = self._member_log_probabilities(params, x, mask, train=True, key=key)
        mean = jax.nn.logsumexp(log_proba, axis=0) - jnp.log(self.members)
        loss = balanced_cross_entropy(mean, targets, output_classes, beta, weights)
        if self.member_loss_weight == 0.0:
            return loss
        own = jnp.mean(jnp.stack(
            [balanced_cross_entropy(log_proba[m], targets, output_classes, beta, weights) for m in range(self.members)]
        ))
        return (1.0 - self.member_loss_weight) * loss + self.member_loss_weight * own

    def member_models(self) -> list[BiDozeNetModel]:
        """Each member as a separate BiDoze model."""
        assert self.params is not None
        out = []
        for m in range(self.members):
            member = BiDozeNetModel(self.config)
            member.input_shape = self.input_shape
            member.params = jax.tree_util.tree_map(lambda leaf, m=m: leaf[m], self.params)
            member.model = {"params": member.params}
            out.append(member)
        return out

    def member_paths(self, filepath: str | Path) -> list[Path]:
        filepath = Path(filepath)
        return [filepath.with_name(f"{filepath.stem}_member{m}{filepath.suffix}") for m in range(self.members)]

    def savez(self, filepath: str | Path) -> None:
        """One BiDoze checkpoint per member: ``<stem>_member<m>.pisces`` beside ``filepath``."""
        for member, path in zip(self.member_models(), self.member_paths(filepath), strict=True):
            member.savez(path)

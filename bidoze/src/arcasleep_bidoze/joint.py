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

from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel


class JointEnsemble(BiDozeNetModel):
    def __init__(self, config: BiDozeNetConfig, members: int):
        super().__init__(config)
        if members < 2:
            raise ValueError("a joint ensemble needs at least 2 members")
        self.members = members

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
        single = BiDozeNetModel._apply
        if key is None:
            logits = jax.vmap(lambda p: single(self, p, x, mask, train=train))(params)
        else:
            keys = jax.random.split(key, self.members)
            logits = jax.vmap(lambda p, k: single(self, p, x, mask, train=train, key=k))(params, keys)
        log_proba = jax.nn.log_softmax(logits, axis=-1)
        return jax.nn.logsumexp(log_proba, axis=0) - jnp.log(self.members)

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

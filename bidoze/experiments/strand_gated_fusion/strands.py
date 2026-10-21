"""Run frozen strands on recordings and return each strand's probabilities and penultimate features.

A strand's penultimate features are the GELU of its ``head_hidden`` layer (width // 2), the
input of its logit layer. The three strand sources are:

- ``coop``: a cooperating-arms fit (``CooperatingModel.load_research`` of a fit directory);
- ``pisces``: separate BiDoze ``.pisces`` checkpoints (split7 members, independent seeds).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arcasleep.data.batching import pad_pooled, recording_input
from arcasleep.data.bundle import CacheBundle

PAD = 3000
CACHE = "12Hz_log_PSD_aligned"


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def _hidden_filter(module, method_name):
    return method_name == "__call__" and (module.name or "").endswith("head_hidden")


def _collect(intermediates, names):
    flat = {}

    def walk(tree, path):
        if isinstance(tree, dict):
            for k, v in tree.items():
                walk(v, path + (k,))
        else:
            flat[path] = tree[0] if isinstance(tree, (tuple, list)) else tree

    walk(intermediates, ())
    return [next(v for p, v in flat.items() if p[-2] == n) for n in names]


class CoopStrands:
    def __init__(self, fit_dir: Path):
        from model import CooperatingModel  # cooperating_arms/model.py on sys.path

        self.model = CooperatingModel.load_research(Path(fit_dir))
        self.n = self.model.spec.members
        self.parameters = int(sum(x.size for x in jax.tree.leaves(self.model.params)))
        self.names = [f"arm_{i}_head_hidden" for i in range(self.n)]

        @jax.jit
        def run(x):
            out, state = self.model.module.apply(
                {"params": self.model.params}, x, jnp.ones(x.shape[:2]), return_outputs=True,
                capture_intermediates=_hidden_filter, mutable=["intermediates"])
            feats = jnp.stack([_gelu(h) for h in _collect(state["intermediates"], self.names)], axis=2)
            return out["arm_probabilities"], feats  # (S,B,T,5), (B,T,S,F)

        self._run = run

    def __call__(self, x):
        return self._run(x)


class PiscesStrands:
    def __init__(self, paths):
        from arcasleep_bidoze.model import BiDozeNetModel

        self.models = [BiDozeNetModel.loadz(p) for p in paths]
        self.n = len(self.models)
        self.parameters = int(sum(x.size for m in self.models for x in jax.tree.leaves(m.params)))

        def one(model):
            def run(x):
                logits, state = model.module.apply(
                    {"params": model.params}, x, jnp.ones(x.shape[:2]), train=False,
                    capture_intermediates=_hidden_filter, mutable=["intermediates"])
                (h,) = _collect(state["intermediates"], ["head_hidden"])
                return jax.nn.softmax(logits, axis=-1), _gelu(h)
            return jax.jit(run)

        self._runs = [one(m) for m in self.models]

    def __call__(self, x):
        outs = [r(x) for r in self._runs]
        return jnp.stack([o[0] for o in outs]), jnp.stack([o[1] for o in outs], axis=2)


def run_recording(strands, path: Path):
    """``(labels (n,), probabilities (S, n, 5), features (n, S, F))`` for one cached recording."""
    bundle = CacheBundle.from_npz(Path(path))
    n = bundle.y.shape[1]
    if n > PAD:
        raise ValueError("recording longer than the 3000-epoch window")
    inputs = recording_input(bundle.X[0], n, 1)
    x = jnp.asarray(pad_pooled(inputs, PAD)[None])
    probs, feats = strands(x)
    return (bundle.y_for_n_classes(4)[0], np.asarray(probs)[:, 0, :n], np.asarray(feats)[0, :n])

"""Train the final WatchTree model and export it as complete depth-6 trees for the watch.

    python -m arcasleep_watchtree.export --features results/features-2026-10-27 \\
        --uncropped results/cv-2026-10-27/features --out results/port-2026-10-27 \\
        --model-config config/model.json --training-config config/training.json

* Training set: complete-context scored epochs of all five cohorts, 87 features (no breathing),
  ``max_bin`` 15, the settings of ``--model-config`` and ``--training-config``. The number of
  rounds is the median of the best iterations of five person-wise folds (early stopping as in the CV).
* Export (layout of ``TreeBench.mc`` in the Garmin repo): per tree 63 feature bytes (bit 7 set when
  a missing value goes left) | 63 4-bit thresholds, low nibble first | 64 leaf bytes (int8 + 128).
  Trees come in rounds of four, class 0 to 3. A branch that ends early is padded with splits that
  always go left and leaves that repeat the value.
* A threshold is a bin index of the feature. Bins are counted from the feature's edges, integers
  in the units of :mod:`~arcasleep_watchtree.portspec`: ``bin(F) = #{edges < F}``; a split goes
  left when ``bin <= threshold``. A missing value has bin 15 for splits that send it right and
  bin 0 for splits that send it left (the feature byte's bit 7 selects which).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from arcasleep_watchtree import cv
from arcasleep_watchtree.definition import CONFIG_DIR, TreeSpec, WatchTree
from arcasleep_watchtree.portspec import FEATURES, NA, SCALE

DEPTH = 6
SPLITS = 63
LEAVES = 64
STRIDE = SPLITS + 32 + LEAVES
N_BINS = 15
MISSING_RIGHT, MISSING_LEFT = 15, 0


def pooled_table(features_dir: Path, uncropped_dir: Path) -> pd.DataFrame:
    tables = cv.load_tables(features_dir, uncropped_dir)
    parts = []
    for name in cv.COHORT_ORDER:
        t = tables[name]
        t = t[t.subject.isin(cv.kept_nights(t))]
        parts.append(cv.select(t, True))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------ edges and trees


def real_thresholds(booster: lgb.Booster, n_features: int) -> list[np.ndarray]:
    """Sorted distinct thresholds of each feature over all trees."""
    found: list[set[float]] = [set() for _ in range(n_features)]

    def walk(node: dict) -> None:
        if "split_feature" in node:
            found[node["split_feature"]].add(float(node["threshold"]))
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in booster.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    return [np.array(sorted(s)) for s in found]


def edge_of(threshold, scale: int):
    """Integer ``E`` with ``F <= E`` for the split ``x <= t``: the cell of ``F`` is ``[F - .5, F + .5)`` in units."""
    return np.rint(np.asarray(threshold, np.float64) * scale - 0.5).astype(np.int64)


def integer_edges(real: list[np.ndarray], features: list[str]) -> list[np.ndarray]:
    """Edges in the feature's integer units; ``F <= E`` is the split ``x <= t``."""
    out = []
    for f, name in enumerate(features):
        out.append(np.unique(edge_of(real[f], SCALE[name])))
    return out


def bin_of(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``#{edges < value}``; NaN (or the integer ``NA``) maps to -1."""
    return np.searchsorted(edges, values, side="left")


def nan_goes_left(node: dict) -> bool:
    if node["missing_type"] == "NaN":
        return bool(node["default_left"])
    return 0.0 <= float(node["threshold"])  # missing_type None: NaN is read as 0.0


class Tree:
    """One tree as complete heap-order arrays."""

    def __init__(self) -> None:
        self.feature = np.zeros(SPLITS, np.int64)
        self.nan_left = np.ones(SPLITS, bool)
        self.thr = np.full(SPLITS, N_BINS, np.int64)  # bin threshold; 15 = always left
        self.leaf = np.zeros(LEAVES)


def build_tree(structure: dict, edges: list[np.ndarray], scale_of: list[int]) -> Tree:
    """``edges`` are the integer edges; ``scale_of`` the units per original unit (to find bins)."""
    tree = Tree()

    def fill(node: dict, i: int, depth: int) -> None:
        if depth == DEPTH:
            assert "leaf_value" in node, "tree deeper than 6"
            tree.leaf[i - SPLITS] = node["leaf_value"]
            return
        if "leaf_value" in node:
            fill(node, 2 * i + 1, depth + 1)
            fill(node, 2 * i + 2, depth + 1)
            return
        f = node["split_feature"]
        j = int(np.searchsorted(edges[f], edge_of(float(node["threshold"]), scale_of[f])))
        tree.feature[i], tree.thr[i], tree.nan_left[i] = f, j, nan_goes_left(node)
        fill(node["left_child"], 2 * i + 1, depth + 1)
        fill(node["right_child"], 2 * i + 2, depth + 1)

    fill(structure, 0, 0)
    return tree


def build_trees(booster: lgb.Booster, features: list[str]):
    real = real_thresholds(booster, len(features))
    edges = integer_edges(real, features)
    scale_of = [SCALE[n] for n in features]
    trees = [build_tree(t["tree_structure"], edges, scale_of) for t in booster.dump_model()["tree_info"]]
    return trees, real, edges


def leaf_scale(trees: list[Tree]) -> float:
    top = max(float(np.abs(t.leaf).max()) for t in trees)
    return top / 127.0


def pack(trees: list[Tree], scale: float) -> bytes:
    """Bytes of every tree, ``STRIDE`` each."""
    out = bytearray()
    for t in trees:
        feat = (t.feature + 128 * t.nan_left).astype(np.uint8)
        thr = np.zeros(32, np.uint8)
        padded = np.concatenate([t.thr, [0]]).astype(np.uint8)
        thr[:] = padded[0::2] | (padded[1::2] << 4)
        leaf = (np.clip(np.rint(t.leaf / scale), -127, 127) + 128).astype(np.uint8)
        out += feat.tobytes() + thr.tobytes() + leaf.tobytes()
    assert len(out) == STRIDE * len(trees)
    assert all(int(t.thr.max()) <= N_BINS for t in trees)
    return bytes(out)


def unpack(blob: bytes, n_trees: int):
    a = np.frombuffer(blob, np.uint8).reshape(n_trees, STRIDE)
    feat = a[:, :SPLITS]
    packed = a[:, SPLITS : SPLITS + 32]
    thr = np.stack([packed & 15, packed >> 4], axis=2).reshape(n_trees, 64)[:, :SPLITS]
    leaf = a[:, SPLITS + 32 :].astype(np.int64) - 128
    return feat, thr, leaf


# ------------------------------------------------------------------------------ evaluation


def bin_matrix(values: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """``(n, 256)`` bin lookup: column ``f`` for splits that send missing right, ``128 + f`` left."""
    n, nf = values.shape
    x = np.zeros((n, 256), np.uint8)
    for f in range(nf):
        v = values[:, f]
        missing = ~np.isfinite(v) | (v == NA)
        b = np.searchsorted(edges[f], np.where(missing, 0, v), side="left")
        x[:, f] = np.where(missing, MISSING_RIGHT, b)
        x[:, 128 + f] = np.where(missing, MISSING_LEFT, b)
    return x


def run_trees(x: np.ndarray, feat, thr, leaf, n_class: int = 4) -> np.ndarray:
    """Sum of the leaf values per class; ``leaf`` may be float or int."""
    n = x.shape[0]
    rows = np.arange(n)
    out = np.zeros((n, n_class), leaf.dtype)
    for t in range(feat.shape[0]):
        i = np.zeros(n, np.int64)
        for _ in range(DEPTH):
            go_left = x[rows, feat[t, i]] <= thr[t, i]
            i = np.where(go_left, 2 * i + 1, 2 * i + 2)
        out[:, t % n_class] += leaf[t, i - SPLITS]
    return out


def float_leaf_arrays(trees: list[Tree]):
    feat = np.stack([t.feature + 128 * t.nan_left for t in trees])
    thr = np.stack([t.thr for t in trees])
    leaf = np.stack([t.leaf for t in trees])
    return feat, thr, leaf


def kappa_by_night(table: pd.DataFrame, pred: np.ndarray) -> float:
    out = table[["subject", "y4"]].assign(pred=pred)
    rows = [cv._kappa(g.y4.to_numpy(), g.pred.to_numpy()) for _, g in out.groupby("subject")]
    return float(np.nanmean(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--uncropped", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-config", type=Path, default=CONFIG_DIR / "model.json")
    parser.add_argument("--training-config", type=Path, default=CONFIG_DIR / "training.json")
    args = parser.parse_args()
    spec = TreeSpec.load(args.model_config, args.training_config)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds").mkdir(exist_ok=True)

    table = pooled_table(args.features, args.uncropped)
    persons = sorted(table.person.unique())
    folds = cv.person_folds(persons)
    fold = table.person.map(folds).to_numpy()
    (out / "folds.json").write_text(json.dumps(folds, indent=0) + "\n")
    print(f"epochs {len(table)} persons {len(persons)}", flush=True)

    # Five person-wise folds: best iterations, and out-of-fold predictions of the exported model.
    best, fold_stats = [], []
    oof = {k: np.full(len(table), -1) for k in ("float", "export")}
    for k in range(5):
        test = fold == k
        model = WatchTree(spec, FEATURES, N_BINS, 0).fit(table[~test])
        booster = model.booster
        best.append(booster.best_iteration)
        booster.save_model(str(out / "folds" / f"fold{k}.txt"), num_iteration=booster.best_iteration)
        x = table.loc[test, FEATURES].to_numpy(np.float32)
        oof["float"][test] = booster.predict(x, num_iteration=booster.best_iteration).argmax(1)
        sub = lgb.Booster(model_file=str(out / "folds" / f"fold{k}.txt"))
        trees, real, edges = build_trees(sub, FEATURES)
        scale = leaf_scale(trees)
        feat, thr, leaf = unpack(pack(trees, scale), len(trees))
        ints = np.rint(x.astype(np.float64) * np.array([SCALE[n] for n in FEATURES]))
        ints = np.where(np.isnan(x), NA, ints)
        scores = run_trees(bin_matrix(ints, edges), feat, thr, leaf)
        oof["export"][test] = scores.argmax(1)
        fold_stats.append({"fold": k, "best_iteration": int(booster.best_iteration), "n_test": int(test.sum())})
        print(fold_stats[-1], flush=True)
    rounds = int(np.median(best))
    kappas = {name: kappa_by_night(table, p) for name, p in oof.items()}
    pooled = {name: cv._kappa(table.y4, p) for name, p in oof.items()}
    print("rounds", rounds, kappas, flush=True)

    # Final model on everything.
    final = WatchTree(spec, FEATURES, N_BINS).fit_rounds(table, rounds)
    booster = final.booster
    booster.save_model(str(out / "final_float.txt"))
    trees, real, edges = build_trees(booster, FEATURES)
    scale = leaf_scale(trees)
    blob = pack(trees, scale)
    (out / "trees.bin").write_bytes(blob)
    feat, thr, leaf_int = unpack(blob, len(trees))

    # Exactness of the structure: float leaves, bins from the real (unrounded) thresholds.
    x = table[FEATURES].to_numpy(np.float32)
    raw = booster.predict(x, raw_score=True)
    real_trees = [_build_real(t["tree_structure"], real) for t in booster.dump_model()["tree_info"]]
    exact_xb = bin_matrix(np.where(np.isnan(x), NA, x.astype(np.float64)), real)
    exact_scores = run_trees(exact_xb, *float_leaf_arrays(real_trees))
    # Integer path: features rounded to their units, integer edges, float and int8 leaves.
    ints = np.rint(x.astype(np.float64) * np.array([SCALE[n] for n in FEATURES]))
    ints = np.where(np.isnan(x), NA, ints)
    xb = bin_matrix(ints, edges)
    float_scores = run_trees(xb, *float_leaf_arrays(trees))
    int_scores = run_trees(xb, feat, thr, leaf_int)
    report = {
        "n_features": len(FEATURES),
        "rounds": rounds,
        "n_trees": len(trees),
        "tree_bytes": len(blob),
        "stride": STRIDE,
        "leaf_scale": scale,
        "fold_stats": fold_stats,
        "oof_kappa_mean_over_nights": kappas,
        "oof_kappa_pooled": pooled,
        "train_max_abs_raw_diff_float_bins_vs_lightgbm": float(np.abs(exact_scores - raw).max()),
        "train_argmax_agreement_float_bins_vs_lightgbm": float((exact_scores.argmax(1) == raw.argmax(1)).mean()),
        "train_max_abs_raw_diff_integer_edges_float_leaves": float(np.abs(float_scores - raw).max()),
        "train_argmax_agreement_integer_edges_float_leaves": float((float_scores.argmax(1) == raw.argmax(1)).mean()),
        "train_argmax_agreement_integer_edges_int8_leaves": float((int_scores.argmax(1) == raw.argmax(1)).mean()),
        "train_kappa_lightgbm": kappa_by_night(table, raw.argmax(1)),
        "train_kappa_export": kappa_by_night(table, int_scores.argmax(1)),
        "n_edges_per_feature": {n: int(len(e)) for n, e in zip(FEATURES, edges)},
        "features_with_merged_edges": [n for n, r, e in zip(FEATURES, real, edges) if len(e) != len(r)],
    }
    (out / "export_report.json").write_text(json.dumps(report, indent=1) + "\n")
    edge_json = {
        "features": FEATURES,
        "scale": [SCALE[n] for n in FEATURES],
        "edges": [e.tolist() for e in edges],
        "real_edges": [r.tolist() for r in real],
        "leaf_scale": scale,
        "rounds": rounds,
        "n_trees": len(trees),
        "stride": STRIDE,
    }
    (out / "model.json").write_text(json.dumps(edge_json) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("n_edges_per_feature", "fold_stats")}, indent=1))


def _build_real(structure: dict, real: list[np.ndarray]) -> Tree:
    """Tree whose thresholds are bin indices into the real (unrounded) thresholds."""
    tree = Tree()

    def fill(node: dict, i: int, depth: int) -> None:
        if depth == DEPTH:
            tree.leaf[i - SPLITS] = node["leaf_value"]
            return
        if "leaf_value" in node:
            fill(node, 2 * i + 1, depth + 1)
            fill(node, 2 * i + 2, depth + 1)
            return
        f = node["split_feature"]
        tree.feature[i] = f
        tree.thr[i] = int(np.searchsorted(real[f], float(node["threshold"])))
        tree.nan_left[i] = nan_goes_left(node)
        fill(node["left_child"], 2 * i + 1, depth + 1)
        fill(node["right_child"], 2 * i + 2, depth + 1)

    fill(structure, 0, 0)
    return tree


if __name__ == "__main__":
    main()

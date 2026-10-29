"""The tree model: training, prediction, and its file formats.

A model is a dict, the content of ``model.json``:

* ``inputs``, ``features``, ``scale``: the feature set, its column order and the integer units per
  original unit of each feature (:func:`arcasleep_tree.features.scales`);
* ``edges``: per feature, the sorted integer bin edges; a value ``F`` has bin ``#{edges < F}``;
* ``trees``: ``feature`` ``(n, 63)`` uint8 (bit 7 set when a missing value goes left),
  ``threshold`` ``(n, 63)`` uint8 bin thresholds (a split goes left when ``bin <= threshold``) and
  ``leaf`` ``(n, 64)`` int8; complete depth-6 trees in heap order (children of node ``i`` are
  ``2i + 1`` and ``2i + 2``), in rounds of four, one tree per class (0 wake, 1 light, 2 deep,
  3 REM);
* ``leaf_scale``: the score of one int8 leaf step; ``rounds``, ``n_trees``; ``real_edges`` (the
  LightGBM thresholds the edges came from) and ``training`` (how the model was made).

A missing value has bin 15 for splits that send it right and bin 0 for splits that send it left.
A branch that ends above depth 6 is padded with splits that always go left (threshold 15) and
leaves that repeat its value. :func:`predict` runs these integer trees, so it gives the stages
the watch gives for the same feature values.

LightGBM is only the training step: :func:`fit` trains a booster and converts it to a model.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import sosfilt_zi

from arcasleep_tree import features

DEPTH = 6
SPLITS = 2**DEPTH - 1
LEAVES = 2**DEPTH
N_CLASS = 4
N_BINS = 15
MISSING_RIGHT, MISSING_LEFT = 15, 0
#: Marks a missing value among the integer features.
NA = -(2**31)

#: Training rows have complete windows: 2 h of recording before the epoch and the forward reach
#: after it.
BACK_EPOCHS = 120 * features.EPOCHS_PER_MIN
#: Nights with fewer complete-context scored epochs are not trained on.
MIN_NIGHT_EPOCHS = 120

DEFINITION = {
    "params": {
        "objective": "multiclass",
        "num_class": N_CLASS,
        "max_depth": DEPTH,
        "num_leaves": LEAVES,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "max_bin": N_BINS,
        "verbose": -1,
        "seed": 0,
        "num_threads": 24,
    },
    "max_rounds": 130,
    "early_stopping_rounds": 20,
    #: Share of the training persons held out for early stopping.
    "validation_share": 0.15,
    #: Person-wise folds whose median best iteration is the final model's number of rounds.
    "folds": 5,
    "inputs": features.INPUTS,
}


# ---------------------------------------------------------------------------------- training


def trainable(table: pd.DataFrame) -> np.ndarray:
    """Rows a model is trained on.

    These are scored epochs with complete windows, in nights with at least ``MIN_NIGHT_EPOCHS``
    of them.
    """
    epoch = table.epoch.to_numpy()
    n_rows = table.groupby(["cohort", "subject"]).epoch.transform("size").to_numpy()
    keep = (
        (table.y4.to_numpy() >= 0) & (epoch >= BACK_EPOCHS) & (epoch + features.DELAY <= n_rows - 1)
    )
    per_night = (
        pd.Series(keep)
        .groupby([table.cohort.to_numpy(), table.subject.to_numpy()])
        .transform("sum")
    )
    return keep & (per_night.to_numpy() >= MIN_NIGHT_EPOCHS)


def persons(table: pd.DataFrame) -> np.ndarray:
    """Person of each row, unique across cohorts."""
    return (table.cohort.astype(str) + "/" + table.person.astype(str)).to_numpy()


def fit(tree: dict, table: pd.DataFrame, test: np.ndarray, rounds: int | None = None) -> dict:
    """Train on the :func:`trainable` rows of ``table`` where ``test`` is False.

    With ``rounds`` None, training runs up to ``max_rounds`` rounds and stops early on
    ``validation_share`` of the training persons; the model keeps the best iteration.
    Otherwise it trains exactly ``rounds`` rounds on all training rows.
    """
    names = features.feature_names(tree["inputs"])
    train = table[trainable(table) & ~np.asarray(test, bool)]
    x = train[names].to_numpy(np.float32)
    y = train.y4.to_numpy()
    params = tree["params"]
    data_params = {"max_bin": params["max_bin"]}
    if rounds is None:
        who = persons(train)
        pool = np.unique(who)
        rng = np.random.default_rng(params["seed"])
        n_val = max(1, int(round(tree["validation_share"] * pool.size)))
        is_val = np.isin(who, rng.choice(pool, size=n_val, replace=False))
        dtrain = lgb.Dataset(x[~is_val], y[~is_val], feature_name=names, params=data_params)
        dval = lgb.Dataset(x[is_val], y[is_val], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            tree["max_rounds"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(tree["early_stopping_rounds"], verbose=False)],
        )
        rounds = booster.best_iteration
    else:
        booster = lgb.train(
            params, lgb.Dataset(x, y, feature_name=names, params=data_params), rounds
        )
    model = from_booster(booster, tree["inputs"], rounds)
    model["training"] = {
        "definition": tree,
        "rows": int(len(train)),
        "cohorts": sorted(train.cohort.unique().tolist()),
    }
    return model


def person_folds(table: pd.DataFrame, k: int, seed: int = 0) -> np.ndarray:
    """Fold of each row: persons shuffled with ``seed`` and dealt round-robin into ``k`` folds."""
    who = persons(table)
    order = sorted(set(who))
    np.random.default_rng(seed).shuffle(order)
    fold_of = {p: i % k for i, p in enumerate(order)}
    return np.array([fold_of[p] for p in who])


def train(tree: dict, table: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """The final model and the out-of-fold stage of every row (``-1`` where not predicted).

    Each of ``folds`` person-wise folds is fit with early stopping; the final model trains on
    all rows for the median of their best iterations.
    """
    fold = person_folds(table, tree["folds"], tree["params"]["seed"])
    oof = np.full(len(table), -1)
    best = []
    for k in range(tree["folds"]):
        test = fold == k
        model = fit(tree, table, test)
        best.append(model["rounds"])
        oof[test] = predict(model, table[test])
        print(f"fold {k}: {model['rounds']} rounds", flush=True)
    final = fit(tree, table, np.zeros(len(table), bool), rounds=int(np.median(best)))
    final["training"]["fold_rounds"] = best
    return final, oof


# ---------------------------------------------------------------------------------- from LightGBM


def _edge_of(threshold, scale: int):
    """Integer ``E`` with ``F <= E`` for the split ``x <= t``.

    The cell of ``F`` is ``[F - .5, F + .5)``.

    Edges are kept inside the watch's 32-bit range, so a split LightGBM writes as ``x <= 1e300``
    sends every value left.
    """
    edge = np.rint(np.asarray(threshold, np.float64) * scale - 0.5)
    return np.clip(edge, NA + 1, 2**31 - 1).astype(np.int64)


def _nan_goes_left(node: dict) -> bool:
    if node["missing_type"] == "NaN":
        return bool(node["default_left"])
    return 0.0 <= float(node["threshold"])  # missing_type None: NaN is read as 0.0


def from_booster(booster: lgb.Booster, inputs: dict, rounds: int) -> dict:
    """The model of the first ``rounds`` rounds of ``booster``."""
    names = features.feature_names(inputs)
    scale_of = features.scales(inputs)
    scale = [scale_of[n] for n in names]
    dumped = booster.dump_model(num_iteration=rounds)["tree_info"]

    found: list[set[float]] = [set() for _ in names]

    def collect(node: dict) -> None:
        if "split_feature" in node:
            found[node["split_feature"]].add(float(node["threshold"]))
            collect(node["left_child"])
            collect(node["right_child"])

    for t in dumped:
        collect(t["tree_structure"])
    real = [np.array(sorted(s)) for s in found]
    edges = [np.unique(_edge_of(r, s)) for r, s in zip(real, scale, strict=True)]

    n = len(dumped)
    feat = np.zeros((n, SPLITS), np.int64)
    nan_left = np.ones((n, SPLITS), bool)
    thr = np.full((n, SPLITS), N_BINS, np.int64)
    leaf = np.zeros((n, LEAVES))

    for t, info in enumerate(dumped):

        def fill(node: dict, i: int, depth: int, t: int = t) -> None:
            if depth == DEPTH:
                assert "leaf_value" in node, f"tree deeper than {DEPTH}"
                leaf[t, i - SPLITS] = node["leaf_value"]
                return
            if "leaf_value" in node:
                fill(node, 2 * i + 1, depth + 1)
                fill(node, 2 * i + 2, depth + 1)
                return
            f = node["split_feature"]
            feat[t, i] = f
            thr[t, i] = int(np.searchsorted(edges[f], _edge_of(float(node["threshold"]), scale[f])))
            nan_left[t, i] = _nan_goes_left(node)
            fill(node["left_child"], 2 * i + 1, depth + 1)
            fill(node["right_child"], 2 * i + 2, depth + 1)

        fill(info["tree_structure"], 0, 0)

    assert int(thr.max()) <= N_BINS
    leaf_scale = float(np.abs(leaf).max()) / 127.0
    return {
        "inputs": inputs,
        "features": names,
        "scale": scale,
        "edges": [e.tolist() for e in edges],
        "real_edges": [r.tolist() for r in real],
        "leaf_scale": leaf_scale,
        "rounds": int(rounds),
        "n_trees": n,
        "trees": {
            "feature": (feat + 128 * nan_left).astype(np.uint8),
            "threshold": thr.astype(np.uint8),
            "leaf": np.clip(np.rint(leaf / leaf_scale), -127, 127).astype(np.int8),
        },
    }


# ---------------------------------------------------------------------------------- prediction


def integer_features(model: dict, table: pd.DataFrame) -> np.ndarray:
    """The model's features in their integer units; ``NA`` where missing."""
    x = table[model["features"]].to_numpy(np.float64)
    ints = np.rint(x * np.asarray(model["scale"], np.float64))
    return np.where(np.isfinite(x), ints, NA)


def bins(model: dict, ints: np.ndarray) -> np.ndarray:
    """``(n, 256)`` bin lookup.

    Column ``f`` is for splits that send a missing value right, ``128 + f`` for left.
    """
    x = np.zeros((ints.shape[0], 256), np.uint8)
    for f, edges in enumerate(model["edges"]):
        v = ints[:, f]
        missing = v == NA
        b = np.searchsorted(
            np.asarray(edges, np.int64), np.where(missing, 0, v).astype(np.int64), side="left"
        )
        x[:, f] = np.where(missing, MISSING_RIGHT, b)
        x[:, 128 + f] = np.where(missing, MISSING_LEFT, b)
    return x


def scores(model: dict, table: pd.DataFrame) -> np.ndarray:
    """``(n, 4)`` summed int8 leaves per class.

    Times ``leaf_scale``, this is LightGBM's raw score.
    """
    x = bins(model, integer_features(model, table))
    feat, thr, leaf = (model["trees"][k] for k in ("feature", "threshold", "leaf"))
    rows = np.arange(x.shape[0])
    out = np.zeros((x.shape[0], N_CLASS), np.int64)
    for t in range(feat.shape[0]):
        i = np.zeros(x.shape[0], np.int64)
        for _ in range(DEPTH):
            i = np.where(x[rows, feat[t, i]] <= thr[t, i], 2 * i + 1, 2 * i + 2)
        out[:, t % N_CLASS] += leaf[t, i - SPLITS]
    return out


def predict(model: dict, table: pd.DataFrame) -> np.ndarray:
    """Stage of each row: 0 wake, 1 light, 2 deep, 3 REM."""
    return scores(model, table).argmax(axis=1)


# ---------------------------------------------------------------------------------- files


TREE_DTYPES = {"feature": np.uint8, "threshold": np.uint8, "leaf": np.int8}


def _unpack(blob: bytes, n_trees: int) -> dict:
    """Trees from the watch's packed layout.

    Each tree is 159 bytes: 63 feature bytes | 32 bytes of 4-bit thresholds | 64 leaf bytes
    (int8 + 128).
    """
    a = np.frombuffer(blob, np.uint8).reshape(n_trees, SPLITS + 32 + LEAVES)
    packed = a[:, SPLITS : SPLITS + 32]
    return {
        "feature": a[:, :SPLITS].copy(),
        "threshold": np.stack([packed & 15, packed >> 4], axis=2)
        .reshape(n_trees, 64)[:, :SPLITS]
        .copy(),
        "leaf": (a[:, SPLITS + 32 :].astype(np.int16) - 128).astype(np.int8),
    }


def _pack(model: dict) -> bytes:
    out = bytearray()
    trees = model["trees"]
    for feat, thr, leaf in zip(trees["feature"], trees["threshold"], trees["leaf"], strict=True):
        padded = np.concatenate([thr, [0]]).astype(np.uint8)
        out += (
            feat.tobytes()
            + (padded[0::2] | (padded[1::2] << 4)).tobytes()
            + (leaf.astype(np.int16) + 128).astype(np.uint8).tobytes()
        )
    return bytes(out)


def save(model: dict, path: Path | str) -> None:
    """Write ``model`` as ``.json`` or ``.safetensors``, by the suffix of ``path``."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.numpy import save_file

        tensors = {f"trees/{k}": np.ascontiguousarray(v) for k, v in model["trees"].items()}
        tensors["edges"] = np.array([v for e in model["edges"] for v in e], np.int64)
        tensors["edge_counts"] = np.array([len(e) for e in model["edges"]], np.int32)
        rest = {k: v for k, v in model.items() if k not in ("trees", "edges")}
        save_file(tensors, str(path), metadata={"model": json.dumps(rest)})
        return
    out = {**model, "trees": {k: v.tolist() for k, v in model["trees"].items()}}
    path.write_text(json.dumps(out) + "\n")


def load(path: Path | str) -> dict:
    """A model from ``.json`` or ``.safetensors``.

    A ``model.json`` without ``trees`` reads them from the ``trees.bin`` beside it.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), "numpy") as handle:
            model = json.loads(handle.metadata()["model"])
            model["trees"] = {k: handle.get_tensor(f"trees/{k}") for k in TREE_DTYPES}
            flat, counts = handle.get_tensor("edges"), handle.get_tensor("edge_counts")
        starts = np.concatenate([[0], np.cumsum(counts)])
        model["edges"] = [flat[a:b].tolist() for a, b in zip(starts[:-1], starts[1:], strict=True)]
        return model
    model = json.loads(path.read_text())
    if "trees" in model:
        model["trees"] = {
            k: np.asarray(model["trees"][k], dtype) for k, dtype in TREE_DTYPES.items()
        }
    else:
        model["trees"] = _unpack((path.parent / "trees.bin").read_bytes(), model["n_trees"])
    model.setdefault("inputs", features.INPUTS)
    return model


# ---------------------------------------------------------------------------------- Monkey C


#: Per-epoch signals in the order the Monkey C app numbers its rings.
SIGNALS = (
    "jerk", "jerk_peak", "lfm", "angle", "angle_step", "daz", "sib", "act",
    "hr", "hr_std", "hr_n", "hr_base", "hr_rel", "hr_valid", "tsb",
)  # fmt: skip
STAT_CODE = {"mean": 0, "max": 1, "std": 2}
DIR_CODE = {"b": 0, "f": 1}
TREES_PER_CHUNK = 16
#: Low-pass coefficient bits on the watch.
Q = 24
#: atan(i / 64) in 1/1024 degree, i = 0..65.
ATAN = [int(round(np.degrees(np.arctan(min(i, 64) / 64.0)) * 1024)) for i in range(66)]


def _mc_array(values, long: bool = False) -> str:
    suffix = "l" if long else ""
    return "[" + ", ".join(f"{v}{suffix}" for v in values) + "]"


def to_monkey_c(model: dict, app_dir: Path | str) -> None:
    """Write the model into a Garmin app such as ``stagerlive``.

    Writes ``resources/generated.xml`` (the trees, base64, 16 per string),
    ``resources/data/edges.json``, ``resources/data/edge_counts.json`` and
    ``source/StagerSpec.mc`` (sizes, rings, feature and window tables, filter coefficients and
    the arctangent table).
    """
    app_dir = Path(app_dir)
    n_trees = model["n_trees"]
    assert n_trees % N_CLASS == 0 and TREES_PER_CHUNK % N_CLASS == 0
    blob = _pack(model)
    stride = SPLITS + 32 + LEAVES
    windows = features.windows(model["inputs"])
    sig_id = {s: i for i, s in enumerate(SIGNALS)}
    reach = {s: 0 for s in SIGNALS}
    for w in windows:
        if w["dir"] == "b":
            reach[w["signal"]] = max(reach[w["signal"]], w["epochs"])
    ring_size = [features.DELAY + 2 + reach[s] for s in SIGNALS]
    kind, arg = [], []
    for name in model["inputs"]["features"].get("at_time", []):
        if name == "tnb_min":
            kind.append(2), arg.append(0)
        elif name == "hr_missing":
            kind.append(1), arg.append(0)
        else:
            kind.append(0), arg.append(sig_id[{"tsb_min": "tsb"}.get(name, name)])
    kind += [3] * len(windows)
    arg += list(range(len(windows)))
    assert len(kind) == len(model["features"])

    chunk_bytes = TREES_PER_CHUNK * stride
    tree_ids = [f"T{c:03d}" for c in range((n_trees + TREES_PER_CHUNK - 1) // TREES_PER_CHUNK)]
    xml = ["<resources>", "    <strings>"]
    for c, name in enumerate(tree_ids):
        part = blob[c * chunk_bytes : (c + 1) * chunk_bytes]
        xml.append(f'        <string id="{name}">{base64.b64encode(part).decode()}</string>')
    xml += [
        "    </strings>",
        '    <jsonData id="EdgeCounts" filename="data/edge_counts.json" />',
        '    <jsonData id="Edges" filename="data/edges.json" />',
        "</resources>",
        "",
    ]
    data = app_dir / "resources" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (app_dir / "resources" / "generated.xml").write_text("\n".join(xml))
    (data / "edge_counts.json").write_text(json.dumps([len(e) for e in model["edges"]]))
    (data / "edges.json").write_text(json.dumps([v for e in model["edges"] for v in e]))

    coef = [int(round(v * (1 << Q))) for s in features.SOS for v in (s[0], s[1], s[2], s[4], s[5])]
    zi = [int(round(v * (1 << Q))) for row in sosfilt_zi(features.SOS) for v in row]
    src = f"""import Toybox.Lang;

//! Generated by arcasleep_tree.to_monkey_c from a tree model; do not edit.
//!
//! {n_trees} trees ({model["rounds"]} rounds of 4), leaf scale {model["leaf_scale"]:.9g} per int8 step,
//! {len(model["features"])} features, scoring delay {features.DELAY} epochs.
module StagerSpec {{
    const N_FEATURES = {len(model["features"])};
    const N_TREES = {n_trees};
    const ROUNDS = {model["rounds"]};
    const N_SIGNALS = {len(SIGNALS)};

    //! Epochs each signal ring holds, in SIGNALS order.
    const RING_SIZE = {_mc_array(ring_size)} as Array<Number>;

    //! Feature kinds: 0 current value of ring FEAT_ARG, 1 no heart-rate sample, 2 epochs to the next
    //! bout, 3 window FEAT_ARG.
    const FEAT_KIND = {_mc_array(kind)} as Array<Number>;
    const FEAT_ARG = {_mc_array(arg)} as Array<Number>;

    //! Windows: ring, statistic (0 mean, 1 max, 2 std), direction (0 backward, 1 forward), epochs.
    const WIN_SIGNAL = {_mc_array([sig_id[w["signal"]] for w in windows])} as Array<Number>;
    const WIN_STAT = {_mc_array([STAT_CODE[w["stat"]] for w in windows])} as Array<Number>;
    const WIN_DIR = {_mc_array([DIR_CODE[w["dir"]] for w in windows])} as Array<Number>;
    const WIN_N = {_mc_array([w["epochs"] for w in windows])} as Array<Number>;

    //! Two biquad sections, b0 b1 b2 a1 a2 each, Q24; and the initial state per unit input, Q24.
    const COEF = {_mc_array(coef, True)} as Array<Long>;
    const ZI = {_mc_array(zi, True)} as Array<Long>;

    //! atan(i / 64) in 1/1024 degree, i = 0..65.
    const ATAN = {_mc_array(ATAN)} as Array<Number>;

    const TREE_CHUNKS = [{", ".join(f"Rez.Strings.{n}" for n in tree_ids)}];
}}
"""  # noqa: E501
    (app_dir / "source").mkdir(parents=True, exist_ok=True)
    (app_dir / "source" / "StagerSpec.mc").write_text(src)

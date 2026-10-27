"""The fixed-point reference, the golden stream codec and the tree export."""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np

from arcasleep_watchtree import export as ex
from arcasleep_watchtree.fixed import ATAN_LUT, Engine, atan_q10, rdiv, tdiv
from arcasleep_watchtree.portgolden import decode_chunks, encode_chunks, synthetic_batches
from arcasleep_watchtree.portspec import FEATURES, NA, SCALE


def test_atan_table_error_is_small():
    worst = 0.0
    for z in range(-3000, 3001, 97):
        for h in range(0, 3000, 131):
            want = math.degrees(math.atan2(z, h))
            worst = max(worst, abs(atan_q10(z, h) / 1024 - want))
    assert worst < 0.02
    assert len(ATAN_LUT) == 66


def test_integer_division_rounds_as_documented():
    assert tdiv(-7, 2) == -3 and tdiv(7, 2) == 3
    assert rdiv(5, 2) == 3 and rdiv(-5, 2) == -3 and rdiv(4, 3) == 1


def test_golden_chunks_round_trip():
    batches = list(synthetic_batches(300))
    batches[5] = (batches[5][0], batches[5][1], batches[5][2], 300, batches[5][4])
    chunks = encode_chunks(batches)
    assert all(len(c) <= 3000 for c in chunks)
    back = list(decode_chunks(chunks))
    assert [(list(a), list(b), list(c), g, h) for a, b, c, g, h in back] == [
        (a, b, c, g, h) for a, b, c, g, h in batches
    ]


def test_engine_runs_and_keeps_rings_in_range():
    eng = Engine(None)
    for xs, ys, zs, gap, bpm in synthetic_batches(30 * 400 + 5):
        eng.batch(xs, ys, zs, gap, bpm or None)
    assert eng.e >= 400
    assert len(eng.out) == eng.e - 90
    assert all(len(r["features"]) == 87 for r in eng.out)
    assert all(v == NA or isinstance(v, int) for r in eng.out[:5] for v in r["features"])


def test_exported_trees_reproduce_the_booster():
    rng = np.random.default_rng(0)
    names = FEATURES[:6]
    x = rng.normal(size=(2000, 6)) * [10, 1, 0.1, 5, 2, 1]
    x[rng.random(x.shape) < 0.1] = np.nan
    y = ((np.nan_to_num(x[:, 0]) > 3) + (np.nan_to_num(x[:, 1]) > 0.5)).astype(int)
    params = {"objective": "multiclass", "num_class": 4, "max_depth": 6, "num_leaves": 64,
              "learning_rate": 0.1, "max_bin": 15, "verbose": -1, "min_data_in_leaf": 20}  # fmt: skip
    booster = lgb.train(params, lgb.Dataset(x, y, feature_name=names), 5)
    trees, real, _ = ex.build_trees(booster, names)
    real_trees = [ex._build_real(t["tree_structure"], real) for t in booster.dump_model()["tree_info"]]
    bins = ex.bin_matrix(np.where(np.isnan(x), NA, x), real)
    scores = ex.run_trees(bins, *ex.float_leaf_arrays(real_trees))
    assert np.abs(scores - booster.predict(x, raw_score=True)).max() < 1e-9
    assert len(ex.pack(trees, ex.leaf_scale(trees))) == ex.STRIDE * len(trees)
    assert SCALE[names[0]] > 0

"""The fixed-point reference and the tree export."""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np

from arcasleep_watchtree.features.constants import SAMPLES_PER_EPOCH
from arcasleep_watchtree.watch import export as ex
from arcasleep_watchtree.watch.fixed import ATAN_LUT, Engine, atan_q10, rdiv, tdiv
from arcasleep_watchtree.watch.spec import FEATURES, NA, SCALE

#: Movement levels (milli-g) and wrist orientations of the synthetic stream.
LEVELS = [1, 2, 3, 6, 12, 25, 60, 150, 400]
ORIENT = [(0, 0, 1000), (300, 100, 950), (-500, 200, 800), (100, -700, 700), (0, 900, 400)]


class Lcg:
    """``r = (r * 75 + 74) % 65537``."""

    def __init__(self, seed: int = 1) -> None:
        self.r = seed

    def next(self) -> int:
        self.r = (self.r * 75 + 74) % 65537
        return self.r


def synthetic_batches(n_batches: int):
    """A synthetic wrist stream in integer milli-g: ``(xs, ys, zs, gap, bpm)`` per 25-sample batch.

    The movement level and the orientation change every few epochs, heart rate is sometimes
    missing (``bpm`` 0), a gap of 40 samples comes every 1500 batches and some epochs carry a
    burst on the x axis.
    """
    rng = Lcg()
    n = 0
    for b in range(n_batches):
        epoch = n // SAMPLES_PER_EPOCH
        r = rng.next()
        bpm = 0 if r % 9 == 0 else 50 + (epoch % 7) * 3 + rng.next() % 9
        gap = 40 if b % 1500 == 777 else 0
        level = LEVELS[(epoch * 7 + epoch // 5) % 9]
        ox, oy, oz = ORIENT[(epoch // 6) % 5]
        xs, ys, zs = [], [], []
        for i in range(25):
            xs.append(ox + rng.next() % (2 * level + 1) - level)
            ys.append(oy + rng.next() % (2 * level + 1) - level)
            zs.append(oz + rng.next() % (2 * level + 1) - level)
            if epoch % 13 == 5 and (n + i) % SAMPLES_PER_EPOCH < 100:
                xs[-1] += rng.next() % 801 - 400
        n += 25 + gap
        yield xs, ys, zs, gap, bpm


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

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from arcasleep_tree import features, tree


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    """Six nights of random features whose stage follows two of them."""
    rng = np.random.default_rng(0)
    names = features.feature_names()
    n_nights, n_epochs = 6, 600
    x = rng.normal(size=(n_nights * n_epochs, len(names))).astype(np.float32)
    x[rng.random(x.shape) < 0.05] = np.nan
    y = (np.nan_to_num(x[:, 0]) > 0).astype(int) + 2 * (np.nan_to_num(x[:, 20]) > 0.5)
    out = pd.DataFrame(x, columns=names)
    out.insert(0, "epoch", np.tile(np.arange(n_epochs), n_nights))
    out.insert(0, "y4", y.astype(np.int8))
    out.insert(0, "person", np.repeat([f"p{i // 2}" for i in range(n_nights)], n_epochs))
    out.insert(0, "subject", np.repeat([f"n{i}" for i in range(n_nights)], n_epochs))
    out.insert(0, "cohort", "synthetic")
    return out


@pytest.fixture(scope="module")
def model(table) -> dict:
    return tree.fit(tree.DEFINITION, table, np.zeros(len(table), bool), rounds=20)


def test_predict_matches_lightgbm(table):
    names = features.feature_names()
    rows = table[tree.trainable(table)]
    x = rows[names].to_numpy(np.float32)
    params = tree.DEFINITION["params"]
    booster = lgb.train(params, lgb.Dataset(x, rows.y4.to_numpy(), params={"max_bin": 15}), 20)
    converted = tree.from_booster(booster, features.INPUTS, 20)
    agree = booster.predict(x, raw_score=True).argmax(1) == tree.predict(converted, rows)
    assert agree.mean() >= 0.99


@pytest.mark.parametrize("suffix", [".json", ".safetensors"])
def test_save_load_round_trip(model, table, tmp_path, suffix):
    path = tmp_path / f"model{suffix}"
    tree.save(model, path)
    back = tree.load(path)
    for key in model["trees"]:
        np.testing.assert_array_equal(back["trees"][key], model["trees"][key])
    assert back["edges"] == model["edges"]
    np.testing.assert_array_equal(tree.predict(back, table), tree.predict(model, table))


def test_early_stopping_fit_runs(table):
    fitted = tree.fit(tree.DEFINITION, table, table.subject.eq("n0").to_numpy())
    assert 1 <= fitted["rounds"] <= tree.DEFINITION["max_rounds"]


def test_monkey_c_export_builds(model, tmp_path):
    tree.to_monkey_c(model, tmp_path)
    assert (tmp_path / "source" / "StagerSpec.mc").exists()


def test_compute_features_builds():
    rng = np.random.default_rng(1)
    n_epochs = 20
    t = np.arange(n_epochs * features.SAMPLES_PER_EPOCH) / 32.0
    xyz = rng.normal(scale=0.01, size=(t.size, 3)) + [0.0, 0.0, 1.0]
    acc, valid = features.to_25hz(t, xyz, 0.0, n_epochs)
    hr_t = np.arange(0, n_epochs * features.EPOCH_S, 1.0)
    out = features.compute_features(acc, valid, hr_t, np.full(hr_t.size, 60.0))
    assert list(out.columns) == ["epoch", *features.feature_names()]
    assert len(out) == n_epochs

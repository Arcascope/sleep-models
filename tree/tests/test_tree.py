import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from arcasleep_tree import features, loaders, tree


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


@pytest.mark.parametrize("suffix", [".json", ".safetensors"])
def test_save_writes_float_booster_beside_integer_model(model, tmp_path, suffix):
    path = tmp_path / f"model{suffix}"
    tree.save(model, path)
    assert path.exists()
    assert (tmp_path / "model_float.txt").exists()
    back = tree.load(path)
    booster = lgb.Booster(model_str=back["float_booster"])
    again = tree.from_booster(booster, features.INPUTS, model["rounds"])
    for key in model["trees"]:
        np.testing.assert_array_equal(again["trees"][key], model["trees"][key])
    assert again["edges"] == model["edges"]
    assert again["leaf_scale"] == model["leaf_scale"]


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


@pytest.mark.parametrize("suffix", [".json", ".safetensors"])
def test_train_saves_every_fold_model(table, tmp_path, suffix):
    definition = {**tree.DEFINITION, "folds": 3, "max_rounds": 5, "early_stopping_rounds": 2}
    out = tmp_path / f"model{suffix}"
    _, oof = tree.train(definition, table, folds_to=out)
    fold = tree.person_folds(table, 3, definition["params"]["seed"])
    held_out = set()
    for k in range(3):
        path = tree.fold_path(out, k)
        assert path.exists()
        assert (tmp_path / f"model_fold{k}_float.txt").exists()
        back = tree.load(path)
        assert "float_booster" in back
        record = back["training"]
        assert record["fold"] == k
        assert record["fold_persons"] == sorted(set(tree.persons(table[fold == k])))
        held_out.update(record["fold_persons"])
        np.testing.assert_array_equal(tree.predict(back, table[fold == k]), oof[fold == k])
    assert held_out == set(tree.persons(table))


def test_weighted_fit_runs(table):
    weight = np.where(table.subject.eq("n1"), 7.0, 1.0)
    held_out = table.subject.eq("n0").to_numpy()
    fitted = tree.fit(tree.DEFINITION, table, held_out, weight=weight)
    assert 1 <= fitted["rounds"] <= tree.DEFINITION["max_rounds"]
    fixed = tree.fit(tree.DEFINITION, table, held_out, rounds=5, weight=weight)
    assert fixed["rounds"] == 5
    definition = {**tree.DEFINITION, "folds": 3, "max_rounds": 5, "early_stopping_rounds": 2}
    final, oof = tree.train(definition, table, weight=weight)
    assert final["rounds"] >= 1
    assert oof.shape == (len(table),)


def test_e4_quantize_snaps_to_grid_and_clips():
    a = np.array([[0.0, 0.0101, -0.0101], [1.0, 2.5, -3.0], [0.5, 0.2, 0.3]])
    q = loaders.e4_quantize(a)
    np.testing.assert_allclose(q * 64, np.round(q * 64))
    assert q.max() <= 2.0 and q.min() >= -2.0
    np.testing.assert_array_equal(loaders.e4_quantize(q), q)


def test_smooth_heart_rate_uses_recorded_time():
    t = np.array([0.0, 1.0, 2.0, 3.0, 100.0, 101.0])
    bpm = np.array([60.0, 70.0, 80.0, 90.0, 50.0, 70.0])
    out_t, out = loaders.smooth_heart_rate(t, bpm, 2.5)
    np.testing.assert_array_equal(out_t, t)
    assert out.shape == bpm.shape
    assert out[0] == 60.0 and out[4] == 50.0
    assert out[3] == pytest.approx(80.0)
    assert out[5] == pytest.approx(60.0)

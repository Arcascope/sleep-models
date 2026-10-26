"""The strand-fusion architecture (F2): ``BiDozeNetConfig.strands`` > 0."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from arcasleep.training.balanced import Patience  # noqa: E402
from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel  # noqa: E402

BIDOZE = Path(__file__).resolve().parents[1]
F2_CONFIG = BIDOZE / "config/model-f2.json"
F2_PARAMETERS = 2_196_792
T, F, C = 24, 7, 2
GAP = 4


def small_config(**changes) -> BiDozeNetConfig:
    base = dict(num_classes=4, score_gap=True, gap_from_row=True, n_heads=2, n_transformer_layers=1,
                conv_kernels=[3, 7], dropout=0.0, strand_feature_dropout=0.0, strands=2, strand_width=16, strand_ffn=16,
                fusion_proj=8, fusion_width=16, fusion_layers=1, fusion_heads=2, fusion_ffn=16)
    return BiDozeNetConfig.from_dict({**base, **changes})


def recording(rng: np.random.Generator, present: int, lead: int = 0) -> np.ndarray:
    """``(T, F + 1, C)``: ``present`` epochs with data after ``lead`` absent ones, absent after."""
    x = np.zeros((T, F + 1, C), dtype=np.float32)
    x[:, :F] = rng.normal(size=(T, F, C))
    x[:, -1] = 1.0
    x[lead:lead + present, -1] = 0.0
    return x


def built(config: BiDozeNetConfig, seed: int = 0) -> BiDozeNetModel:
    model = BiDozeNetModel(config)
    model._ensure_built((T, F + 1, C), seed=seed)
    return model


def probabilities(model: BiDozeNetModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.nn.softmax(model.infer(x), axis=-1))


def test_the_f2_config_file_has_the_research_f2_weight_count() -> None:
    config = BiDozeNetConfig.from_dict(json.loads(F2_CONFIG.read_text()))
    model = BiDozeNetModel(config)
    model._ensure_built((8, 61, 2))

    assert config.strands == 3
    assert model.parameter_count() == F2_PARAMETERS


def test_the_g_configs_stay_single_network_models() -> None:
    for name in ("model.json", "model-gap-rule.json"):
        assert BiDozeNetConfig.from_dict(json.loads((BIDOZE / "config" / name).read_text())).strands == 0


def test_strands_need_the_gap_rule_and_one_frame_per_epoch() -> None:
    with pytest.raises(ValueError):
        small_config(gap_from_row=False)
    with pytest.raises(ValueError):
        small_config(frames_per_epoch=3)
    with pytest.raises(ValueError):
        small_config(strand_width=18)


def test_the_output_has_five_classes_and_gap_exactly_where_the_gap_row_says() -> None:
    rng = np.random.default_rng(0)
    x = np.stack([recording(rng, 16, lead=3), recording(rng, 10)])
    model = built(small_config())

    proba = probabilities(model, x)
    absent = x[:, :, -1, 0] >= 1.0

    assert proba.shape == (2, T, 5)
    assert np.all(proba[absent, GAP] == 1.0)
    assert np.all(proba[~absent, GAP] == 0.0)
    np.testing.assert_allclose(proba.sum(axis=-1), 1.0, atol=1e-5)


def test_every_strand_head_is_gap_exactly_where_the_gap_row_says() -> None:
    rng = np.random.default_rng(1)
    x = recording(rng, 15, lead=4)[None]
    model = built(small_config())

    out = model.module.apply({"params": model.params}, jnp.asarray(x), jnp.ones((1, T)), return_outputs=True)
    strands = np.asarray(jax.nn.softmax(out["strand_logits"], axis=-1))
    absent = x[0, :, -1, 0] >= 1.0

    assert strands.shape == (2, 1, T, 5)
    assert np.all(strands[:, 0, absent, GAP] == 1.0)
    assert np.all(strands[:, 0, ~absent, GAP] == 0.0)


def test_absent_epochs_before_the_recording_do_not_change_the_present_epochs() -> None:
    rng = np.random.default_rng(2)
    night = recording(rng, 14)
    shifted = np.tile(night[-1:], (T, 1, 1))
    shifted[6:20] = night[:14]
    model = built(small_config())

    a = probabilities(model, night[None])[0]
    b = probabilities(model, shifted[None])[0]

    np.testing.assert_allclose(a[:14], b[6:20], atol=1e-5)


def test_the_loss_trains_every_strand_head_and_the_fusion_head() -> None:
    rng = np.random.default_rng(3)
    x = jnp.asarray(recording(rng, 20)[None])
    y = jnp.asarray(rng.integers(0, 4, size=(1, T)))
    model = built(small_config(dropout=0.1))
    grads = jax.grad(lambda p: model._balanced_loss(p, x, jnp.ones((1, T)), y, jax.random.PRNGKey(0), 5, 0.999))(
        model.params)

    for i in range(2):
        assert float(jnp.abs(grads[f"arm_{i}_head_logits"]["kernel"]).sum()) > 0
    assert float(jnp.abs(grads["fusion"]["head_logits"]["kernel"]).sum()) > 0


def test_a_zero_strand_loss_weight_leaves_the_strand_heads_untrained() -> None:
    rng = np.random.default_rng(4)
    x = jnp.asarray(recording(rng, 20)[None])
    y = jnp.asarray(rng.integers(0, 4, size=(1, T)))
    model = built(small_config(strand_loss_weight=0.0))
    grads = jax.grad(lambda p: model._balanced_loss(p, x, jnp.ones((1, T)), y, jax.random.PRNGKey(0), 5, 0.999))(
        model.params)

    assert float(jnp.abs(grads["arm_0_head_logits"]["kernel"]).sum()) == 0
    assert float(jnp.abs(grads["fusion"]["head_logits"]["kernel"]).sum()) > 0


def test_the_loss_is_the_output_loss_plus_the_weighted_strand_losses() -> None:
    from arcasleep.training.balanced import balanced_cross_entropy

    rng = np.random.default_rng(5)
    x = jnp.asarray(recording(rng, 20)[None])
    y = jnp.asarray(np.where(np.arange(T) < 20, rng.integers(0, 4, size=T), -1)[None])
    model = built(small_config(strand_loss_weight=0.3))
    out = model.module.apply({"params": model.params}, x, jnp.ones((1, T)), return_outputs=True)

    expected = balanced_cross_entropy(out["logits"], y, 5, 0.999) + 0.3 * sum(
        balanced_cross_entropy(s, y, 5, 0.999) for s in out["strand_logits"])
    loss = model._balanced_loss(model.params, x, jnp.ones((1, T)), y, jax.random.PRNGKey(0), 5, 0.999)

    np.testing.assert_allclose(loss, expected, rtol=1e-5)


def test_a_balanced_fit_runs_and_keeps_the_gap_rule() -> None:
    rng = np.random.default_rng(6)
    x = np.stack([recording(rng, 20), recording(rng, 16, lead=4)])
    y = np.full((2, T), -2, dtype=np.int32)
    y[0, :20] = rng.integers(0, 4, 20)
    y[1, 4:20] = rng.integers(0, 4, 16)
    model = BiDozeNetModel(small_config(dropout=0.1))

    history = model.fit_balanced(
        x, y, x, y, batch_size=2, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
        patience=Patience(lr_drop=10, early_stop=20, unit="steps"), max_epochs=2, fit_seed=0,
    )

    assert np.all(np.isfinite(history["train_loss"]))
    assert np.all(probabilities(model, x)[x[:, :, -1, 0] < 1.0, GAP] == 0.0)


def test_a_checkpoint_round_trip_reproduces_the_probabilities(tmp_path: Path) -> None:
    x = recording(np.random.default_rng(7), 18, lead=2)[None]
    model = built(small_config())
    model.savez(tmp_path / "model.pisces")

    loaded = BiDozeNetModel.loadz(tmp_path / "model.pisces")

    assert loaded.config.strands == model.config.strands
    np.testing.assert_allclose(probabilities(loaded, x), probabilities(model, x), atol=1e-6)


def test_stacked_f2_seeds_give_the_mean_of_their_probabilities() -> None:
    from arcasleep_bidoze.adult import StackedModel

    x = np.stack([recording(np.random.default_rng(8), 18, lead=2)])
    models = [built(small_config(), seed=k) for k in range(3)]

    stacked = np.asarray(jax.nn.softmax(StackedModel(models).infer(x), axis=-1))

    mean = np.mean([probabilities(m, x) for m in models], axis=0)
    np.testing.assert_allclose(stacked, mean, atol=1e-5)


@pytest.mark.parametrize("members", [1, 2])
def test_a_flat_bundle_of_f2_members_predicts_a_recording(tmp_path: Path, members: int) -> None:
    pytest.importorskip("senpy")
    pytest.importorskip("safetensors")
    from arcasleep.artifacts.publish import _save_member_weights
    from arcasleep.models import MODEL_TYPE_KEY
    from arcasleep.predict import Predictor

    config = small_config().to_dict()
    bundle = tmp_path / "adult"
    bundle.mkdir()
    models = []
    for k in range(members):
        model = BiDozeNetModel(BiDozeNetConfig.from_dict(config))
        model._ensure_built((120, 61, 2), seed=k)
        models.append(model)
    _save_member_weights(models, bundle / "model.safetensors")
    (bundle / "config.json").write_text(json.dumps({
        MODEL_TYPE_KEY: "bidoze", **config, "members": members,
        "stage_divisor": {"light": 0.8, "deep": 1.6, "rem": 0.9},
    }))
    (bundle / "preprocessing.json").write_text((BIDOZE / "config/inputs.json").read_text())

    fs, minutes = 25.0, 47
    t = np.arange(int(fs * 60 * minutes)) / fs
    xyz = np.random.default_rng(5).normal(scale=0.05, size=(t.size, 3)) + [0.0, 0.0, 1.0]
    result = Predictor(bundle).predict(np.column_stack([t, xyz]))

    assert result.probabilities.shape == (minutes * 2, 4)
    assert len(result.labels) == minutes * 2


@pytest.mark.parametrize("model, config", [("f2", "config/model-f2.json"), ("g", "config/model-gap-rule.json")])
def test_the_harness_dry_run_trains_the_chosen_release_architecture(model: str, config: str) -> None:
    env = {**os.environ, "BIDOZE_RELEASE": "adult", "BIDOZE_RELEASE_MODEL": model, "BIDOZE_RUN_ID": "dry"}
    out = subprocess.run(["bash", str(BIDOZE / "scripts/harness.sh"), "--dry-run"], cwd=BIDOZE, env=env,
                         capture_output=True, text=True, check=True).stdout

    train_lines = [line.split() for line in out.splitlines() if " bidoze train " in line]
    assert len(train_lines) == 6
    assert all(words[words.index("-mc") + 1] == config for words in train_lines)


def test_an_unknown_release_architecture_is_refused() -> None:
    env = {**os.environ, "BIDOZE_RELEASE": "adult", "BIDOZE_RELEASE_MODEL": "h"}
    status = subprocess.run(["bash", str(BIDOZE / "scripts/harness.sh"), "--dry-run"], cwd=BIDOZE, env=env,
                            capture_output=True).returncode

    assert status != 0


def test_a_research_f2_checkpoint_gives_the_research_probabilities() -> None:
    """Needs F2_PARITY_RUN (a research run directory with ``weights/`` and ``predictions/``),
    F2_PARITY_RECORDING (the cached recording of one of its predictions) and F2_PARITY_PREDICTION
    (that prediction's file name)."""
    run, recording_path = os.environ.get("F2_PARITY_RUN"), os.environ.get("F2_PARITY_RECORDING")
    prediction = os.environ.get("F2_PARITY_PREDICTION")
    if not (run and recording_path and prediction):
        pytest.skip("no research F2 checkpoint given")
    from arcasleep.data.batching import pad_pooled, recording_input
    from arcasleep.data.bundle import CacheBundle

    run = Path(run)
    spec = json.loads((run / "config.json").read_text())["spec"]
    config = BiDozeNetConfig.from_dict(json.loads(F2_CONFIG.read_text()))
    assert (config.strand_width, config.fusion_width) == (spec["width"], spec["fusion_width"])
    model = BiDozeNetModel(config)
    model.load_weights_from_directory(run / "weights")
    bundle = CacheBundle.from_npz(Path(recording_path))
    n = bundle.y.shape[1]
    x = pad_pooled(recording_input(bundle.X[0], n, 1), model.input_shape[0])[None]

    ours = probabilities(model, x)[0, :n]

    saved = np.load(run / "predictions" / prediction)
    # The saved probabilities come from a GPU run at default matmul precision.
    np.testing.assert_allclose(ours, saved["fusion_probabilities"], atol=5e-3)

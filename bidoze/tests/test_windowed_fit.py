"""The windowed mode of the balanced recipe: config, determinism, and an end-to-end fit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from arcasleep.data.bundle import CacheBundle  # noqa: E402
from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR  # noqa: E402
from arcasleep.data.constants import OUTSIDE_PSG_LABEL  # noqa: E402
from arcasleep.training.balanced import Patience  # noqa: E402
from arcasleep.training.driver import RunPaths, run_training  # noqa: E402
from arcasleep.training.windows import WindowMix  # noqa: E402
from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel  # noqa: E402
from arcasleep_bidoze.model.definition import sleep_wake_stage_map  # noqa: E402
from arcasleep_bidoze.model.constants import STAGE_1_MAP  # noqa: E402
from arcasleep_bidoze.training import TrainingConfig  # noqa: E402

from test_balanced_recipe import BALANCED_TRAINING, FRAMES_PER_EPOCH, _configs  # noqa: E402

WINDOW = 16
WINDOWED = {
    **BALANCED_TRAINING,
    "windowed": True,
    "window_epochs": WINDOW,
    "x_length": WINDOW,
    "y_length": WINDOW,
    "weak_label_rules": ["daytime_wake", "daytime_sleep"],
}


def _write_uncropped_cache(cache_dir: Path, dataset: str, subjects: list[str], epochs: int = 48) -> None:
    rng = np.random.default_rng(len(dataset))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        y = np.full((1, epochs), OUTSIDE_PSG_LABEL)
        y[0, 12:36] = rng.choice([0, 1, 2, 3, 5], size=24)
        CacheBundle(
            X=rng.normal(size=(1, epochs * FRAMES_PER_EPOCH, 5, 2)),
            y=y,
            datasets=np.array([dataset]),
            subject_ids=np.array([subject]),
            anglez_5s=np.cumsum(rng.normal(0, 6.0, size=(1, epochs, 6)), axis=2) % 60,
        ).save_npz(cache_dir / f"{dataset}{SUBJECT_SEPARATOR}{subject}{CACHE_SUFFIX}")


def test_mode_off_keeps_the_default_window_settings_inert() -> None:
    config = TrainingConfig.from_dict(BALANCED_TRAINING)
    assert not config.windowed
    for key in ("window_mix", "weak_label_rules", "window_epochs"):
        with pytest.raises(ValueError):
            TrainingConfig.from_dict({**BALANCED_TRAINING, key: {"night": 1.0} if key == "window_mix"
                                      else ["daytime_wake"] if key == "weak_label_rules" else 500})


def test_windowed_config_validates() -> None:
    config = TrainingConfig.from_dict(WINDOWED)
    assert config.window_sampling_mix == WindowMix()
    for bad in (
        {"x_length": WINDOW + 1},
        {"recipe": "curriculum"},
        {"weak_label_rules": ["nope"]},
        {"window_mix": {"bogus": 1.0}},
    ):
        with pytest.raises(ValueError):
            TrainingConfig.from_dict({**WINDOWED, **bad})


def test_8h_config_files_load_together() -> None:
    root = Path(__file__).parents[1] / "config"
    training = TrainingConfig.from_json(root / "training-8h.json")
    assert training.windowed and training.window_epochs == 960
    assert training.weak_label_rules == ["daytime_wake", "daytime_sleep"]
    BiDozeNetConfig.from_dict(json.loads((root / "model-8h.json").read_text()))


def test_sleep_wake_map_is_the_curriculum_first_stage_without_its_gap_group() -> None:
    without_gap = {k: v for k, v in STAGE_1_MAP.items() if k != 2}
    assert sleep_wake_stage_map(4) == without_gap


def test_windowed_recipe_trains_and_scores_end_to_end(tmp_path: Path) -> None:
    config = _configs(tmp_path / "config", {"gap_from_row": True})
    (tmp_path / "config" / "training.json").write_text(json.dumps(WINDOWED))
    config = type(config).from_paths(
        family=config.family,
        inputs=tmp_path / "config" / "inputs.json",
        model=tmp_path / "config" / "model.json",
        cv=tmp_path / "config" / "cv.json",
        training=tmp_path / "config" / "training.json",
        metrics=tmp_path / "config" / "metrics.json",
    )
    cache_root = tmp_path / "feature_cache"
    _write_uncropped_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(6)])
    _write_uncropped_cache(cache_root / "test", "CohortB", ["b0", "b1"])
    results = tmp_path / "results" / "training"
    run_training(config, RunPaths(results=results, feature_cache=cache_root), run_id="test")
    assert list((results / "model_outputs").glob("*.pisces"))


def test_windowed_fit_is_seed_deterministic_and_trains() -> None:
    from arcasleep.data.weak_labels import label_recording
    from arcasleep.training.windows import WindowRecording, WindowSampler

    rng = np.random.default_rng(0)

    def rec(i: int) -> WindowRecording:
        y = np.full(48, OUTSIDE_PSG_LABEL)
        y[12:36] = rng.integers(0, 4, size=24)
        grid = np.cumsum(rng.normal(0, 6.0, size=(48, 6)), axis=1) % 60
        x = rng.normal(size=(48, 6, 2)).astype(np.float32)
        x[:, -1] = 0.0
        return WindowRecording(x, label_recording(y, grid, ["daytime_wake", "daytime_sleep"]))

    recs = [rec(i) for i in range(4)]
    sampler = WindowSampler(recs, WINDOW)
    eval_batch = WindowSampler(recs[:2], WINDOW).centered()

    def fit():
        model = BiDozeNetModel(
            BiDozeNetConfig.from_dict(
                dict(num_classes=4, score_gap=True, gap_from_row=True, d_model=16, n_heads=2,
                     n_transformer_layers=1, d_ff=32, conv_kernels=[3], dropout=0.0)
            )
        )
        history = model.fit_balanced(
            None, None, eval_batch.x, eval_batch.labels, batch_size=2, initial_lr=1e-3,
            weight_decay=0.0, balance_beta=0.999, patience=Patience(5, 10), max_epochs=3,
            fit_seed=7, windows=sampler,
        )
        return history, model

    first, model = fit()
    second, _ = fit()
    assert first["train_loss"] == second["train_loss"]
    assert np.isfinite(first["train_loss"]).all()
    logits = model.infer(eval_batch.x)
    assert logits.shape == (2, WINDOW, 5)


def test_only_a_windowed_run_scores_in_overlapping_windows() -> None:
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from arcasleep_bidoze.family import FAMILY
    from arcasleep_bidoze.training import TrainingConfig

    config_dir = Path(__file__).parents[1] / "config"
    windowed = TrainingConfig.from_json(config_dir / "training-8h.json")
    plain = TrainingConfig.from_json(config_dir / "training-balanced.json")

    assert FAMILY.scoring_window_step(SimpleNamespace(training=windowed)) == windowed.window_epochs // 8
    assert FAMILY.scoring_window_step(SimpleNamespace(training=plain)) is None

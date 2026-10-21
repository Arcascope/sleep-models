"""The BiDoze balanced recipe through the shared driver: split, fit, checkpoint, score."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from pisces_lite.cv import SubjectRef  # noqa: E402

from arcasleep.data.bundle import CacheBundle  # noqa: E402
from arcasleep.data.cache import CACHE_SUFFIX, SUBJECT_SEPARATOR  # noqa: E402
from arcasleep.models import checkpoint_family, get_family  # noqa: E402
from arcasleep.training.balanced import Patience, split_train_eval  # noqa: E402
from arcasleep.training.config import RunConfig  # noqa: E402
from arcasleep.training.driver import FIT_EVAL_SPLITS_DIRNAME, RunPaths, run_training  # noqa: E402
from arcasleep_bidoze.training import TrainingConfig  # noqa: E402

BIDOZE = get_family("bidoze")
FRAMES_PER_EPOCH = 3
BALANCED_TRAINING = {
    "recipe": "balanced",
    "run_seed": 0,
    "seed_count": 1,
    "epochs": 2,
    "batch_size": 2,
    "lr": 0.001,
    "weight_decay": 0.0,
    "x_length": 16,
    "y_length": 16,
}


def _write_cache(cache_dir: Path, dataset: str, subjects: list[str], epochs: int) -> None:
    rng = np.random.default_rng(len(dataset))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        # Raw 0-5 PSG labels: wake, N1..N3, REM.
        y = rng.choice([0, 1, 2, 3, 5], size=(1, epochs))
        X = rng.normal(size=(1, epochs * FRAMES_PER_EPOCH, 5, 2))
        CacheBundle(
            X=X, y=y, datasets=np.array([dataset]), subject_ids=np.array([subject])
        ).save_npz(cache_dir / f"{dataset}{SUBJECT_SEPARATOR}{subject}{CACHE_SUFFIX}")


def _recorded_split(results: Path) -> tuple[list[SubjectRef], list[SubjectRef]]:
    (path,) = (results / FIT_EVAL_SPLITS_DIRNAME).glob("*.json")
    split = json.loads(path.read_text())
    return tuple(
        [SubjectRef(row["dataset"], row["subject_id"]) for row in split[side]]
        for side in ("fit", "eval")
    )


def _configs(
    directory: Path, model_overrides: dict | None = None, cv_overrides: dict | None = None
) -> RunConfig:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "inputs.json": {
            "type": "nufft",
            "cache_prefix": "test",
            "window_seconds": 10,
            "window_step_seconds": 10,
        },
        "model.json": {
            "num_classes": 4,
            "score_gap": True,
            "d_model": 16,
            "n_heads": 2,
            "n_transformer_layers": 1,
            "d_ff": 32,
            "conv_kernels": [3],
            "dropout": 0.0,
            **(model_overrides or {}),
        },
        "cv.json": {
            "mode": "transfer",
            "train_sets": ["CohortA"],
            "test_sets": ["CohortB"],
            **(cv_overrides or {}),
        },
        "training.json": BALANCED_TRAINING,
        "metrics.json": {
            "num_classes": 4,
            "class_names": ["wake", "light", "deep", "rem", "gap"],
            "psg_dt_minutes": 0.5,
            "min_stage_minutes": 5.0,
            "metrics": ["sleep_sensitivity", "wake_specificity"],
        },
    }
    for name, payload in files.items():
        (directory / name).write_text(json.dumps(payload))
    return RunConfig.from_paths(
        family=BIDOZE,
        inputs=directory / "inputs.json",
        model=directory / "model.json",
        cv=directory / "cv.json",
        training=directory / "training.json",
        metrics=directory / "metrics.json",
    )


@pytest.mark.parametrize("model_frames_per_epoch", [1, FRAMES_PER_EPOCH])
def test_balanced_recipe_trains_and_scores_end_to_end(
    tmp_path: Path, model_frames_per_epoch: int
) -> None:
    config = _configs(tmp_path / "config", {"frames_per_epoch": model_frames_per_epoch})
    cache_root = tmp_path / "feature_cache"
    _write_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(6)], epochs=12)
    _write_cache(cache_root / "test", "CohortB", ["b0", "b1"], epochs=9)
    results = tmp_path / "results" / "training"

    run_training(config, RunPaths(results=results, feature_cache=cache_root), run_id="test")

    with (results / "cv_results_4_stages.csv").open(newline="") as handle:
        scored = {row["subject_id"] for row in csv.DictReader(handle)}
    assert scored == {"b0", "b1"}
    (checkpoint,) = (results / "model_outputs").glob("*.pisces")
    assert checkpoint_family(checkpoint) == "bidoze"
    assert _recorded_split(results) == split_train_eval(
        [SubjectRef("CohortA", f"a{i}") for i in range(6)], 0.3, 0
    )

    (fold_dir,) = (results / "predictions").iterdir()
    for subject in ("b0", "b1"):
        with (fold_dir / f"CohortB_{subject}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 9
        assert {row["psg"] for row in rows} <= {"wake", "light", "deep", "rem"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"deblip_adjustment_epochs": 20, "deblip_logit_weight": 0.5},
        {"sliding_copies": 2},
        {"recipe": "curriculum", "early_stop_patience": 5},
    ],
)
def test_options_the_chosen_recipe_does_not_use_are_refused(overrides: dict) -> None:
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({**BALANCED_TRAINING, **overrides})


@pytest.mark.parametrize(
    "cv_overrides",
    [{"deblip_sets": ["CohortA"]}],
)
def test_cv_options_the_balanced_recipe_does_not_use_are_refused(
    tmp_path: Path, cv_overrides: dict
) -> None:
    from pisces_lite.cv import Fold

    config = _configs(tmp_path / "config", cv_overrides=cv_overrides)
    fold = Fold(
        fold_id=0,
        total_folds=1,
        mode="transfer",
        train_subjects=[SubjectRef("CohortA", f"a{i}") for i in range(4)],
        test_subjects=[],
        val_subjects=[],
        description="",
    )

    with pytest.raises(ValueError):
        BIDOZE.train_fold(config, fold=fold, fit_seed=0, split_seed=0, features={}, datasets={})


def test_a_fit_with_no_finite_eval_kappa_refuses_to_pick_an_epoch(tmp_path: Path) -> None:
    from arcasleep.data.constants import GAP_LABEL

    from arcasleep_bidoze.model import BiDozeNetModel

    model = BiDozeNetModel(_configs(tmp_path / "config").model)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 16, 6, 2)).astype(np.float32)
    y = rng.integers(0, 4, size=(4, 16)).astype(np.int32)
    no_scored_epochs = np.full((2, 16), GAP_LABEL, dtype=np.int32)

    with pytest.raises(ValueError):
        model.fit_balanced(
            x, y, x[:2], no_scored_epochs,
            batch_size=2, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.999,
            patience=Patience(lr_drop=1, early_stop=2), max_epochs=3, fit_seed=0,
        )


def test_balanced_recipe_early_stops_on_validation_sets(tmp_path: Path) -> None:
    config = _configs(tmp_path / "config", cv_overrides={"validation_sets": ["CohortC"]})
    cache_root = tmp_path / "feature_cache"
    _write_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(3)], epochs=12)
    _write_cache(cache_root / "test", "CohortB", ["b0"], epochs=9)
    _write_cache(cache_root / "test", "CohortC", ["c0", "c1"], epochs=10)
    results = tmp_path / "results" / "training"

    run_training(config, RunPaths(results=results, feature_cache=cache_root), run_id="test")

    (checkpoint,) = (results / "model_outputs").glob("*.pisces")
    assert checkpoint_family(checkpoint) == "bidoze"
    assert _recorded_split(results) == (
        [SubjectRef("CohortA", f"a{i}") for i in range(3)],
        [SubjectRef("CohortC", "c0"), SubjectRef("CohortC", "c1")],
    )


def _release(directory: Path, config) -> Path:
    from arcasleep.models import MODEL_TYPE_KEY

    from arcasleep_bidoze.model import BiDozeNetModel

    model = BiDozeNetModel(config)
    model._ensure_built((16, 5, 2), seed=7)
    directory.mkdir(parents=True)
    model.save_weights_safetensors(directory / "model.safetensors")
    model_json = json.loads((directory.parent / "config" / "model.json").read_text())
    (directory / "config.json").write_text(json.dumps({MODEL_TYPE_KEY: "bidoze", **model_json}))
    return directory


def test_a_fit_can_start_from_a_release_bundles_weights(tmp_path: Path) -> None:
    import jax

    from arcasleep_bidoze.model import BiDozeNetModel
    from arcasleep_bidoze.training import _load_init_weights

    config = _configs(tmp_path / "config").model
    bundle = _release(tmp_path / "release", config)
    model = BiDozeNetModel(config)

    _load_init_weights(model, bundle)

    source = BiDozeNetModel(config)
    source.load_weights_safetensors(bundle / "model.safetensors")
    for a, b in zip(jax.tree_util.tree_leaves(model.params),
                    jax.tree_util.tree_leaves(source.params), strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_fit_can_start_from_a_checkpoints_weights(tmp_path: Path) -> None:
    import jax

    from arcasleep_bidoze.model import BiDozeNetModel
    from arcasleep_bidoze.training import _load_init_weights

    config = _configs(tmp_path / "config").model
    source = BiDozeNetModel(config)
    source._ensure_built((16, 5, 2), seed=3)
    source.savez(tmp_path / "teacher.pisces")
    model = BiDozeNetModel(config)

    _load_init_weights(model, tmp_path / "teacher.pisces", (16, 5, 2))

    for a, b in zip(jax.tree_util.tree_leaves(model.params),
                    jax.tree_util.tree_leaves(source.params), strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_fit_scored_from_its_start_keeps_the_best_of_start_and_epochs(tmp_path: Path) -> None:
    from arcasleep_bidoze.model import BiDozeNetModel

    config = _configs(tmp_path / "config").model
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 16, 5, 2)).astype(np.float32)
    y = rng.integers(0, 4, size=(4, 16)).astype(np.int32)
    model = BiDozeNetModel(config)
    model._ensure_built(x.shape[1:], seed=0)

    history = model.fit_balanced(
        x, y, x, y, batch_size=2, initial_lr=1e-3, weight_decay=0.0, balance_beta=0.0,
        patience=TrainingConfig.from_dict(BALANCED_TRAINING).patience, max_epochs=2, fit_seed=0,
        score_start=True,
    )

    assert 0 <= history["best_epoch"] <= 2
    assert history["best_eval_kappa"] >= max(history["eval_kappa"])


def test_starting_weights_from_another_architecture_are_refused(tmp_path: Path) -> None:
    from arcasleep_bidoze.model import BiDozeNetModel
    from arcasleep_bidoze.training import _load_init_weights

    bundle = _release(tmp_path / "release", _configs(tmp_path / "config").model)
    wider = _configs(tmp_path / "wider", {"d_model": 32}).model

    with pytest.raises(ValueError):
        _load_init_weights(BiDozeNetModel(wider), bundle)


def test_init_weights_is_a_balanced_recipe_option() -> None:
    TrainingConfig.from_dict({**BALANCED_TRAINING, "init_weights": "release"})
    with pytest.raises(ValueError):
        TrainingConfig.from_dict(
            {**BALANCED_TRAINING, "recipe": "curriculum", "init_weights": "release"}
        )


def test_a_fit_distills_the_mean_output_of_teacher_checkpoints(tmp_path: Path) -> None:
    cache_root = tmp_path / "feature_cache"
    _write_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(6)], epochs=12)
    _write_cache(cache_root / "test", "CohortB", ["b0", "b1"], epochs=9)
    teachers = tmp_path / "teachers" / "training"
    run_training(_configs(tmp_path / "teacher-config"), RunPaths(results=teachers, feature_cache=cache_root),
                 run_id="teacher")

    config = _configs(tmp_path / "config")
    training = {**BALANCED_TRAINING, "distill_from": str(teachers / "model_outputs")}
    (tmp_path / "config" / "training.json").write_text(json.dumps(training))
    config = RunConfig.from_paths(
        family=BIDOZE,
        inputs=tmp_path / "config" / "inputs.json",
        model=tmp_path / "config" / "model.json",
        cv=tmp_path / "config" / "cv.json",
        training=tmp_path / "config" / "training.json",
        metrics=tmp_path / "config" / "metrics.json",
    )
    student = tmp_path / "student" / "training"
    run_training(config, RunPaths(results=student, feature_cache=cache_root), run_id="student")

    (checkpoint,) = (student / "model_outputs").glob("*.pisces")
    assert checkpoint_family(checkpoint) == "bidoze"


def test_teacher_probabilities_are_a_distribution(tmp_path: Path) -> None:
    from arcasleep_bidoze.model import BiDozeNetModel
    from arcasleep_bidoze.training import teacher_probabilities

    config = _configs(tmp_path / "config").model
    for seed in range(2):
        model = BiDozeNetModel(config)
        model._ensure_built((16, 5, 2), seed=seed)
        model.savez(tmp_path / f"model_seed_{seed}.pisces")
    x = np.random.default_rng(0).normal(size=(3, 16, 5, 2)).astype(np.float32)

    p = teacher_probabilities(tmp_path, x)

    assert p.shape == (3, 16, config.output_classes)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-5)


def test_a_joint_fit_trains_scores_and_saves_each_member(tmp_path: Path) -> None:
    cache_root = tmp_path / "feature_cache"
    _write_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(6)], epochs=12)
    _write_cache(cache_root / "test", "CohortB", ["b0", "b1"], epochs=9)
    _configs(tmp_path / "config")
    (tmp_path / "config" / "training.json").write_text(json.dumps({**BALANCED_TRAINING, "joint_members": 3}))
    config = RunConfig.from_paths(
        family=BIDOZE,
        inputs=tmp_path / "config" / "inputs.json",
        model=tmp_path / "config" / "model.json",
        cv=tmp_path / "config" / "cv.json",
        training=tmp_path / "config" / "training.json",
        metrics=tmp_path / "config" / "metrics.json",
    )
    results = tmp_path / "results" / "training"

    run_training(config, RunPaths(results=results, feature_cache=cache_root), run_id="joint")

    members = sorted((results / "model_outputs").glob("*_member*.pisces"))
    assert len(members) == 3
    assert all(checkpoint_family(m) == "bidoze" for m in members)
    (fold_dir,) = (results / "predictions").iterdir()
    assert {p.name for p in fold_dir.iterdir()} == {"CohortB_b0.csv", "CohortB_b1.csv"}


def test_a_joint_ensemble_outputs_its_members_mean_and_starts_from_their_seeds(tmp_path: Path) -> None:
    import jax

    from arcasleep_bidoze.adult import StackedModel
    from arcasleep_bidoze.joint import JointEnsemble
    from arcasleep_bidoze.model import BiDozeNetModel

    config = _configs(tmp_path / "config").model
    joint = JointEnsemble(config, 3)
    joint._ensure_built((16, 5, 2), seed=4)
    seeds = []
    for seed in (4, 5, 6):
        model = BiDozeNetModel(config)
        model._ensure_built((16, 5, 2), seed=seed)
        seeds.append(model)
    x = np.random.default_rng(0).normal(size=(2, 16, 5, 2)).astype(np.float32)

    joint_proba = np.asarray(jax.nn.softmax(joint.infer(x), axis=-1))

    # A vmapped batch and separate calls round differently on GPU.
    stacked_proba = np.asarray(jax.nn.softmax(StackedModel(seeds).infer(x), axis=-1))
    np.testing.assert_allclose(joint_proba, stacked_proba, rtol=0, atol=1e-5)
    for member, seed in zip(joint.member_models(), seeds, strict=True):
        np.testing.assert_allclose(member.infer(x), seed.infer(x), rtol=0, atol=1e-5)


def test_joint_members_start_from_random_weights() -> None:
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({**BALANCED_TRAINING, "joint_members": 2, "init_weights": "release"})


def _run_with_training(tmp_path: Path, training: dict, model_overrides: dict | None = None) -> Path:
    cache_root = tmp_path / "feature_cache"
    _write_cache(cache_root / "test", "CohortA", [f"a{i}" for i in range(6)], epochs=12)
    _write_cache(cache_root / "test", "CohortB", ["b0", "b1"], epochs=9)
    _configs(tmp_path / "config", model_overrides)
    (tmp_path / "config" / "training.json").write_text(json.dumps({**BALANCED_TRAINING, **training}))
    config = RunConfig.from_paths(
        family=BIDOZE,
        inputs=tmp_path / "config" / "inputs.json",
        model=tmp_path / "config" / "model.json",
        cv=tmp_path / "config" / "cv.json",
        training=tmp_path / "config" / "training.json",
        metrics=tmp_path / "config" / "metrics.json",
    )
    results = tmp_path / "results" / "training"
    run_training(config, RunPaths(results=results, feature_cache=cache_root), run_id="test")
    return results


def test_a_bootstrap_fit_trains_and_saves_a_checkpoint(tmp_path: Path) -> None:
    results = _run_with_training(tmp_path, {"bootstrap_seed": 3})
    (checkpoint,) = (results / "model_outputs").glob("*.pisces")
    assert checkpoint_family(checkpoint) == "bidoze"


@pytest.mark.parametrize("training", [{}, {"joint_members": 2, "member_loss_weight": 0.5}])
def test_a_fit_with_epoch_weights_trains_and_saves_a_checkpoint(tmp_path: Path, training: dict) -> None:
    weights = tmp_path / "weights.npz"
    np.savez(weights, **{f"CohortA/a{i}": np.linspace(0.5, 2.0, 12) for i in range(3)})
    results = _run_with_training(tmp_path, {"epoch_weights": str(weights), **training}, {"gap_from_row": True})
    assert list((results / "model_outputs").glob("*.pisces"))


def test_a_bootstrap_resample_draws_its_size_from_the_recordings() -> None:
    from pisces_lite.cv import SubjectRef

    from arcasleep_bidoze.training import bootstrap_recordings

    refs = [SubjectRef("CohortA", f"a{i}") for i in range(20)]
    drawn = bootstrap_recordings(refs, 7)
    assert len(drawn) == len(refs)
    assert set(drawn) <= set(refs)
    assert drawn == bootstrap_recordings(list(reversed(refs)), 7)


def test_unit_epoch_weights_leave_the_balanced_loss_unchanged() -> None:
    import jax.numpy as jnp

    from arcasleep.training.balanced import balanced_cross_entropy

    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(2, 10, 5)))
    labels = jnp.asarray(rng.integers(-1, 5, size=(2, 10)))
    plain = balanced_cross_entropy(logits, labels, 5, 0.999)
    weighted = balanced_cross_entropy(logits, labels, 5, 0.999, jnp.ones((2, 10)))
    assert float(plain) == pytest.approx(float(weighted))

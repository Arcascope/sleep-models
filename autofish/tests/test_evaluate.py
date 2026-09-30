"""Scoring a saved checkpoint on a cohort: the pieces around the model itself.

The scoring loop calls into JAX, so it is exercised by the real run; what these
tests pin down is the wiring that decides *what* gets scored and *what the rows
say* -- the identity lifted from the training run, the fixed input shape, the
cohort loader, and the truncation report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from pisces_lite.model_io import ModelIOBundle

from autofish.cli import _expand_inputs
from autofish.data.cache import load_bundles
from autofish.evaluation import run as run_module
from autofish.evaluation.run import (
    RunIdentity,
    cohort_input_shape,
    lift_identity,
    results_csv_path,
    run_evaluation,
)
from autofish.training.config import RunConfig

CVS_HEADER = [
    "run_id", "fold", "epoch", "subject_id",
    "dataset", "training_dataset", "metric", "value",
]


def _configs(
    directory: Path, *, x_length: int = 8, metrics=("sleep_sensitivity", "wake_specificity")
) -> RunConfig:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs.json").write_text(json.dumps({"type": "nufft", "cache_prefix": "test"}))
    (directory / "model.json").write_text(
        json.dumps(
            {
                "num_classes": 4,
                "d_model": 32,
                "n_heads": 4,
                "n_transformer_layers": 1,
                "d_ff": 64,
                "conv_kernels": [3, 7],
                "dropout": 0.0,
            }
        )
    )
    (directory / "cv.json").write_text(
        json.dumps(
            {
                "mode": "transfer",
                "train_sets": ["CohortA"],
                "test_sets": ["CohortB"],
                "validation_sets": [],
                "fold_label": "final",
            }
        )
    )
    (directory / "training.json").write_text(
        json.dumps(
            {
                "run_seed": 0,
                "seed_count": 1,
                "epochs": 200,
                "batch_size": 1,
                "lr": 0.0005,
                "weight_decay": 0.0005,
                "x_length": x_length,
                "y_length": x_length,
            }
        )
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "num_classes": 4,
                "class_names": ["wake", "light", "deep", "rem", "gap"],
                "psg_dt_minutes": 0.5,
                "min_stage_minutes": 5.0,
                "metrics": list(metrics),
            }
        )
    )
    return RunConfig.from_paths(
        inputs=directory / "inputs.json",
        model=directory / "model.json",
        cv=directory / "cv.json",
        training=directory / "training.json",
        metrics=directory / "metrics.json",
    )


def _bundle(
    dataset: str,
    subject: str,
    epochs: int,
    *,
    frames_per_epoch: int = 5,
    freq: int = 3,
    channels: int = 2,
) -> ModelIOBundle:
    return ModelIOBundle(
        X=np.zeros((1, epochs * frames_per_epoch, freq, channels), dtype=np.float64),
        y=np.zeros((1, epochs), dtype=np.int64),
        datasets=np.array([dataset]),
        subject_ids=np.array([subject]),
    )


def _training_csv(directory: Path, config: RunConfig, rows: list[RunIdentity]) -> Path:
    path = results_csv_path(directory, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CVS_HEADER)
        for identity in rows:
            writer.writerow(
                [
                    identity.run_id,
                    identity.fold,
                    identity.epoch,
                    "s1",
                    "CohortB",
                    identity.training_dataset,
                    "sleep_sensitivity",
                    "0.9",
                ]
            )
    return path


# --- load_bundles -----------------------------------------------------------


def test_load_bundles_takes_names_from_inside_the_npz(tmp_path: Path) -> None:
    """The file is named for the directory that was processed, not the cohort."""
    misleading = tmp_path / "weaver_subject_9999_features.npz"
    _bundle("SleepAccel-Clinical", "aws002", 4).save_npz(misleading)
    honest = tmp_path / "weaver_subject_1111_features.npz"
    _bundle("SleepAccel-Clinical", "aws001", 4).save_npz(honest)

    subjects = load_bundles([misleading, honest])

    assert [ref.subject_id for ref in subjects] == ["aws001", "aws002"]
    assert {ref.dataset_name for ref in subjects} == {"SleepAccel-Clinical"}


def test_load_bundles_refuses_the_same_subject_twice(tmp_path: Path) -> None:
    first = tmp_path / "a_features.npz"
    second = tmp_path / "b_features.npz"
    _bundle("CohortB", "s1", 4).save_npz(first)
    _bundle("CohortB", "s1", 4).save_npz(second)

    with pytest.raises(ValueError):
        load_bundles([first, second])


def test_load_bundles_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundles([tmp_path / "absent_features.npz"])


def test_load_bundles_refuses_an_empty_list() -> None:
    with pytest.raises(ValueError):
        load_bundles([])


# --- cohort_input_shape -----------------------------------------------------


def test_cohort_input_shape_takes_its_context_from_the_config(tmp_path: Path) -> None:
    config = _configs(tmp_path / "cfg", x_length=8)
    subjects = load_bundles(
        [
            _save(tmp_path / "a_features.npz", _bundle("CohortB", "s1", 4, freq=3, channels=2)),
            _save(tmp_path / "b_features.npz", _bundle("CohortB", "s2", 12, freq=3, channels=2)),
        ]
    )

    # 12 epochs on the long recording: the context is still the config's 8.
    # The 3 bins gain the pooled missing-data row.
    assert cohort_input_shape(config, subjects) == (8, 4, 2)


def test_cohort_input_shape_rejects_a_mixed_feature_surface(tmp_path: Path) -> None:
    config = _configs(tmp_path / "cfg")
    subjects = load_bundles(
        [
            _save(tmp_path / "a_features.npz", _bundle("CohortB", "s1", 4, freq=3)),
            _save(tmp_path / "b_features.npz", _bundle("CohortB", "s2", 4, freq=4)),
        ]
    )

    with pytest.raises(ValueError):
        cohort_input_shape(config, subjects)


def _save(path: Path, bundle: ModelIOBundle) -> Path:
    bundle.save_npz(path)
    return path


# --- lift_identity ----------------------------------------------------------


def test_lift_identity_copies_the_training_runs_identity(tmp_path: Path) -> None:
    config = _configs(tmp_path / "cfg")
    expected = RunIdentity("0ad56e74ff13", "final_seed_0", 200, "CohortA+CohortB")
    _training_csv(tmp_path / "training", config, [expected, expected])

    assert lift_identity(tmp_path / "training", config) == expected


def test_lift_identity_refuses_an_ambiguous_run(tmp_path: Path) -> None:
    config = _configs(tmp_path / "cfg")
    _training_csv(
        tmp_path / "training",
        config,
        [
            RunIdentity("id", "fold0_seed_0", 200, "CohortA"),
            RunIdentity("id", "fold1_seed_0", 200, "CohortA"),
        ],
    )

    with pytest.raises(ValueError):
        lift_identity(tmp_path / "training", config)


def test_lift_identity_reconstructs_when_the_run_wrote_no_csv(tmp_path: Path) -> None:
    config = _configs(tmp_path / "cfg")

    identity = lift_identity(tmp_path / "training", config)

    assert identity.fold == "final_seed_0"
    assert identity.epoch == 200
    assert identity.training_dataset == "CohortA"
    assert identity.run_id


# --- run_evaluation ---------------------------------------------------------


class _FakeNet:
    """Stands in for AutofishNetModel: the loop's use of it is one classmethod."""

    @classmethod
    def loadz(cls, filepath, input_shape=None):
        assert input_shape == (8, 4, 2)
        return cls()


def _fake_evaluate_subject(
    model, bundle, *, target_x_length, target_y_length, num_classes, return_context=False
):
    """All-wake truth and prediction, at the fixed lengths the real one pads to."""
    y_true = np.zeros(target_y_length, dtype=np.int64)
    probs = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), (target_x_length, 1))
    if return_context:
        inputs = np.zeros((target_x_length, 4, 2), dtype=np.float64)
        return y_true, probs, inputs, y_true
    return y_true, probs


def test_evaluate_writes_a_row_per_metric_and_reports_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    config = _configs(tmp_path / "cfg", x_length=8)
    checkpoint = tmp_path / "model_final_seed_0.pisces"
    checkpoint.write_bytes(b"not really a zip")
    subjects = load_bundles(
        [
            _save(tmp_path / "a_features.npz", _bundle("CohortB", "long", 12)),
            _save(tmp_path / "b_features.npz", _bundle("CohortB", "short", 4)),
        ]
    )
    monkeypatch.setattr(run_module, "AutofishNetModel", _FakeNet)
    monkeypatch.setattr(run_module, "evaluate_subject", _fake_evaluate_subject)
    output_dir = tmp_path / "validation"

    with pytest.warns(RuntimeWarning):
        run_evaluation(
            config,
            subjects=subjects,
            checkpoint=checkpoint,
            output_dir=output_dir,
            identity=RunIdentity("id", "final_seed_0", 200, "CohortA"),
        )

    with (output_dir / "cv_results_4_stages.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4  # 2 subjects x 2 configured metrics
    assert {row["subject_id"] for row in rows} == {"long", "short"}
    assert {row["dataset"] for row in rows} == {"CohortB"}
    assert {row["run_id"] for row in rows} == {"id"}

    provenance = json.loads((output_dir / "evaluation.json").read_text())
    assert provenance["truncated"] == ["CohortB/long"]
    assert provenance["input_shape"] == [8, 4, 2]
    assert provenance["checkpoint"]["sha256"]


def test_evaluate_draws_hypnograms_into_a_subdirectory(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    config = _configs(tmp_path / "cfg", x_length=8)
    checkpoint = tmp_path / "model.pisces"
    checkpoint.write_bytes(b"not really a zip")
    subjects = load_bundles([_save(tmp_path / "a_features.npz", _bundle("CohortB", "s1", 4))])
    monkeypatch.setattr(run_module, "AutofishNetModel", _FakeNet)
    monkeypatch.setattr(run_module, "evaluate_subject", _fake_evaluate_subject)
    output_dir = tmp_path / "hypno"

    run_evaluation(
        config, subjects=subjects, checkpoint=checkpoint, output_dir=output_dir,
        identity=RunIdentity("id", "final_seed_0", 200, "CohortA"),
        render_hypnograms=True,
    )

    png = output_dir / "hypnograms" / "CohortB_s1.png"
    assert png.is_file()
    provenance = json.loads((output_dir / "evaluation.json").read_text())
    assert provenance["hypnograms"] == {"CohortB/s1": str(png)}


def test_evaluate_refuses_to_reuse_a_csv_unless_overwritten(tmp_path: Path, monkeypatch) -> None:
    config = _configs(tmp_path / "cfg")
    checkpoint = tmp_path / "model.pisces"
    checkpoint.write_bytes(b"not really a zip")
    subjects = load_bundles([_save(tmp_path / "a_features.npz", _bundle("CohortB", "s1", 4))])
    monkeypatch.setattr(run_module, "AutofishNetModel", _FakeNet)
    monkeypatch.setattr(run_module, "evaluate_subject", _fake_evaluate_subject)
    output_dir = tmp_path / "validation"
    identity = RunIdentity("id", "final_seed_0", 200, "CohortA")

    run_evaluation(
        config, subjects=subjects, checkpoint=checkpoint, output_dir=output_dir,
        identity=identity,
    )
    with pytest.raises(FileExistsError):
        run_evaluation(
            config, subjects=subjects, checkpoint=checkpoint, output_dir=output_dir,
            identity=identity,
        )

    # Overwriting replaces the file rather than appending to it.
    run_evaluation(
        config, subjects=subjects, checkpoint=checkpoint, output_dir=output_dir,
        identity=identity, overwrite=True,
    )
    with (output_dir / "cv_results_4_stages.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


# --- fold mode --------------------------------------------------------------


def _split_plan(path: Path, folds: dict[str, list[str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohort": "weaver",
                "n_splits": len(folds),
                "seed": 0,
                "provenance": {},
                "folds": folds,
            }
        )
    )


def _fold_phase(root: Path, name: str, test_sets: list[str], template: Path) -> Path:
    """A fold phase dir: config snapshots plus a placeholder checkpoint."""
    phase = root / name
    (phase / "config").mkdir(parents=True)
    (phase / "config" / "cv.json").write_text(
        json.dumps(
            {
                "mode": "transfer",
                "train_sets": ["AutofishWeaverFold1"],
                "test_sets": test_sets,
                "validation_sets": [],
                "fold_label": name,
            }
        )
    )
    for filename in ("inputs.json", "model.json", "training.json", "metrics.json"):
        (phase / "config" / filename).write_text((template / filename).read_text())
    (phase / "model_outputs").mkdir()
    (phase / "model_outputs" / f"model_{name}_seed_0.pisces").write_bytes(b"not really a zip")
    return phase


def test_evaluate_folds_scores_each_model_on_its_held_out_subjects(
    tmp_path: Path, monkeypatch
) -> None:
    from autofish.cli import _evaluate

    fold_root = tmp_path / "training"
    template = tmp_path / "template"
    _configs(template)
    _fold_phase(fold_root, "fold0", ["AutofishWeaverFold0"], template)
    _fold_phase(fold_root, "fold1", ["AutofishWeaverFold1"], template)
    plan = tmp_path / "plan.json"
    _split_plan(
        plan,
        {
            "AutofishWeaverFold0": ["w1", "w2"],
            "AutofishWeaverFold1": ["w3", "w4"],
        },
    )
    subject_paths = [
        _save(tmp_path / f"{name}_features.npz", _bundle("weaver", name, 4))
        for name in ("w1", "w2", "w3", "w4")
    ]
    monkeypatch.setattr(run_module, "AutofishNetModel", _FakeNet)
    monkeypatch.setattr(run_module, "evaluate_subject", _fake_evaluate_subject)
    output_dir = tmp_path / "out"

    import argparse

    args = argparse.Namespace(
        input=[str(p) for p in subject_paths],
        training_dir=None,
        fold_config=str(plan),
        fold_training=str(fold_root),
        output_dir=str(output_dir),
        checkpoint=None,
        overwrite=False,
        hypnograms=False,
    )
    assert _evaluate(args) == 0

    scored: dict[str, set[str]] = {}
    for fold_name in ("fold0", "fold1"):
        with (output_dir / fold_name / "cv_results_4_stages.csv").open(newline="") as handle:
            scored[fold_name] = {row["subject_id"] for row in csv.DictReader(handle)}
    assert scored == {"fold0": {"w1", "w2"}, "fold1": {"w3", "w4"}}

    held_out = json.loads((output_dir / "fold0" / "evaluation.json").read_text())["held_out"]
    assert held_out == ["AutofishWeaverFold0"]


def test_evaluate_folds_needs_both_fold_flags(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from autofish.cli import _evaluate

    args = argparse.Namespace(
        input=[], training_dir=None, fold_config=str(tmp_path / "plan.json"),
        fold_training=None, output_dir=str(tmp_path / "out"),
        checkpoint=None, overwrite=False, hypnograms=False,
    )
    with pytest.raises(SystemExit):
        _evaluate(args)


def test_evaluate_folds_refuses_a_checkpoint_override(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from autofish.cli import _evaluate

    args = argparse.Namespace(
        input=[], training_dir=None, fold_config=str(tmp_path / "plan.json"),
        fold_training=str(tmp_path / "training"), output_dir=str(tmp_path / "out"),
        checkpoint="model.pisces", overwrite=False, hypnograms=False,
    )
    with pytest.raises(SystemExit):
        _evaluate(args)


# --- -i expansion -----------------------------------------------------------


def test_expand_inputs_accepts_a_pattern_or_the_paths_it_stands_for(tmp_path: Path) -> None:
    first = _save(tmp_path / "a_features.npz", _bundle("CohortB", "s1", 4))
    second = _save(tmp_path / "b_features.npz", _bundle("CohortB", "s2", 4))

    assert _expand_inputs([str(tmp_path / "*_features.npz")]) == [first, second]
    assert _expand_inputs([str(first), str(second)]) == [first, second]


def test_expand_inputs_refuses_a_pattern_that_matches_nothing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _expand_inputs([str(tmp_path / "*_features.npz")])


def _night_with_dropout_and_lapse() -> np.ndarray:
    """Raw accel: still, then thrashing, a 5 min dropout, a 70 min lapse, still again."""
    fs = 4
    rng = np.random.default_rng(0)

    def stretch(start_s: float, seconds: float, moving: bool) -> np.ndarray:
        t = start_s + np.arange(int(seconds * fs)) / fs
        xyz = rng.normal(size=(t.size, 3)) if moving else np.tile([0.0, 0.0, 1.0], (t.size, 1))
        return np.column_stack([t, xyz])

    parts = [stretch(0, 3600, False), stretch(3600, 1800, True)]
    parts.append(stretch(5400 + 300, 1800, False))          # 5 min dropout
    parts.append(stretch(7500 + 4200, 3600, False))         # 70 min lapse
    return np.concatenate(parts)


def test_epoch_bins_place_each_5s_mean_in_its_epoch():
    from autofish import ggir

    accel = _night_with_dropout_and_lapse()
    bin_start, anglez = ggir.anglez_bins(accel)
    n_epochs = int(np.ceil((bin_start[-1] + 5) / 30))
    grid = ggir.epoch_bins(bin_start, anglez, n_epochs)

    assert grid.shape == (n_epochs, 6)
    k, j = 130, 2                                   # inside the 5 min dropout
    assert np.isnan(grid[k, j]) == (not np.any(np.rint((bin_start - bin_start[0]) / 5) == 6 * k + j))
    assert np.isnan(grid[(5400 + 60) // 30]).all()  # mid-dropout epoch is empty
    probe = 40
    in_bin = (accel[:, 0] >= 30 * probe + 5 * 3) & (accel[:, 0] < 30 * probe + 5 * 4)
    expected = ggir.z_angle(*accel[in_bin, 1:].T).mean()
    assert grid[probe, 3] == pytest.approx(expected)


def test_cached_bins_give_the_same_judgement_as_the_raw_signal():
    from autofish import ggir

    accel = _night_with_dropout_and_lapse()
    bin_start, anglez = ggir.anglez_bins(accel)
    n_epochs = int(np.ceil((bin_start[-1] + 5) / 30))
    epoch_start_s = accel[0, 0] + 30.0 * np.arange(n_epochs)

    raw = ggir.evaluate(accel, epoch_start_s, n_epochs)
    grid = ggir.epoch_bins(bin_start, anglez, n_epochs)
    cached = ggir.evaluate_bins(*ggir.bins_from_epochs(grid), 30.0 * np.arange(n_epochs), n_epochs)

    assert raw.is_sib.any() and not raw.is_sib.all()
    np.testing.assert_array_equal(cached.is_sib, raw.is_sib)
    np.testing.assert_array_equal(cached.is_spt, raw.is_spt)


def test_apply_sib_filter_demotes_sleep_outside_sustained_inactivity():
    from autofish import ggir
    from autofish.evaluation.run import apply_sib_filter

    accel = _night_with_dropout_and_lapse()
    bin_start, anglez = ggir.anglez_bins(accel)
    n_epochs = int(np.ceil((bin_start[-1] + 5) / 30))
    grid = ggir.epoch_bins(bin_start, anglez, n_epochs)

    proba = np.zeros((n_epochs + 4, 5))
    proba[:n_epochs, 2] = 1.0          # every real epoch predicted sleep
    proba[10, :] = 0.0
    proba[10, 4] = 1.0                 # ... except one gap epoch
    proba[n_epochs:, 4] = 1.0          # and the padding

    filtered, demoted = apply_sib_filter(proba, grid, n_epochs)

    assert demoted > 0
    assert np.all(filtered[125:175].argmax(axis=1) == 0)     # thrashing hour: wake
    assert np.all(filtered[20:100].argmax(axis=1) == 2)      # still hour: sleep
    assert filtered[10].argmax() == 4 and np.all(filtered[n_epochs:].argmax(axis=1) == 4)
    assert proba[150].argmax() == 2                          # input untouched


def test_bundle_round_trips_anglez_and_the_base_reader_still_loads_it(tmp_path):
    from autofish.data.bundle import AutofishBundle

    grid = np.arange(12, dtype=float).reshape(1, 2, 6)
    grid[0, 1, 3] = np.nan
    bundle = AutofishBundle(
        X=np.zeros((1, 30, 3, 2)), y=np.zeros((1, 2)),
        datasets=np.array(["d"]), subject_ids=np.array(["s"]), anglez_5s=grid,
    )
    path = tmp_path / "d_subject_s_features.npz"
    bundle.save_npz(path)

    np.testing.assert_array_equal(AutofishBundle.from_npz(path).anglez_5s, grid)
    assert ModelIOBundle.from_npz(path).X.shape == (1, 30, 3, 2)
    AutofishBundle(
        X=bundle.X, y=bundle.y, datasets=bundle.datasets, subject_ids=bundle.subject_ids
    ).save_npz(path)
    assert AutofishBundle.from_npz(path).anglez_5s is None


def test_sib_filter_without_anglez_in_the_cache_is_refused_before_scoring(tmp_path):
    from pisces_lite.cv import SubjectRef

    from autofish.data.bundle import AutofishBundle

    config = _configs(tmp_path / "cfg")
    bundle = AutofishBundle(
        X=np.zeros((1, 30, 3, 2)), y=np.zeros((1, 2)),
        datasets=np.array(["d"]), subject_ids=np.array(["s"]),
    )
    identity = RunIdentity("run", "fold", 1, "d")
    with pytest.raises(ValueError):
        run_evaluation(
            config, subjects={SubjectRef("d", "s"): bundle}, checkpoint=tmp_path / "x.pisces",
            output_dir=tmp_path / "out", identity=identity, ggir_sib_filter=True,
        )

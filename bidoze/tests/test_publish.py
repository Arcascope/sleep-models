"""Building and verifying a release bundle from a finished training phase.

The heavy path -- loading a real ``.pisces`` checkpoint and re-exporting
safetensors -- is exercised by the real release run. What these tests pin down
is the wiring: which config snapshot becomes which bundle file, that the card
carries run-derived facts and a review marker, that checksums verify, and that
the PENDING gate blocks an upload.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("safetensors")

import numpy as np

from arcasleep.artifacts.publish import (
    BUNDLE_FILES,
    _glance_score,
    build_release_bundle,
    load_bundle_model,
    load_release_info,
    verify_bundle,
)
from arcasleep_bidoze.model import BiDozeNetConfig, BiDozeNetModel

SMALL_CONFIG = {
    "num_classes": 4,
    "score_gap": True,
    "d_model": 32,
    "n_heads": 4,
    "n_transformer_layers": 1,
    "d_ff": 64,
    "conv_kernels": [3, 7],
    "dropout": 0.0,
}


def _checkpoint(path: Path) -> Path:
    config = BiDozeNetConfig.from_dict(SMALL_CONFIG)
    model = BiDozeNetModel(config)
    # Build with a real input shape, then save the .pisces the trainer writes.
    model._ensure_built((8, 3, 2))
    model.savez(path)
    return path


def _phase(root: Path) -> Path:
    phase = root / "training" / "final"
    config_dir = phase / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "model.json").write_text(json.dumps(SMALL_CONFIG))
    (config_dir / "inputs.json").write_text(
        json.dumps({"type": "nufft", "cache_prefix": "test", "fs": 12.0, "window_step_seconds": 2})
    )
    (config_dir / "metrics.json").write_text(
        json.dumps(
            {
                "num_classes": 4,
                "class_names": ["wake", "light", "deep", "rem", "gap"],
                "metrics": ["sleep_sensitivity", "wake_specificity"],
            }
        )
    )
    (config_dir / "metrics_effective.json").write_text(
        json.dumps({"num_classes": 4, "class_names": ["wake", "light", "deep", "rem", "gap"]})
    )
    (config_dir / "training.json").write_text(
        json.dumps({"run_seed": 0, "seed_count": 1, "epochs": 30, "batch_size": 4,
                    "lr": 0.0005, "weight_decay": 0.0005, "x_length": 8, "y_length": 8})
    )
    (config_dir / "split-manifest.json").write_text(json.dumps({"enabled": False}))
    (phase / "model_outputs").mkdir()
    _checkpoint(phase / "model_outputs" / "model_final_seed_0.pisces")
    return phase


def _results_csv(directory: Path, dataset: str, fold: str, n_subjects: int = 3) -> Path:
    rows = []
    for subject in range(n_subjects):
        for metric, value in (
            ("sleep_sensitivity", 0.9 + subject * 0.01),
            ("wake_specificity", 0.5 + subject * 0.01),
            ("kappa_4stage", 0.4 + subject * 0.01),
        ):
            rows.append(
                [f"{fold}", fold, 30, subject, dataset, "Fold0", metric, value]
            )
    path = directory / "cv_results_4_stages.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["run_id", "fold", "epoch", "subject_id", "dataset",
             "training_dataset", "metric", "value"]
        )
        writer.writerows(rows)
    return path


PEDIATRIC_INFO = Path(__file__).resolve().parents[1] / "cards" / "bidoze-pediatric.toml"
# The sdist leaves the authored cards out, so tests that read one skip there.
needs_pediatric_info = pytest.mark.skipif(
    not PEDIATRIC_INFO.is_file(), reason="bidoze/cards is not in this tree"
)


def _info_without_card(root: Path) -> Path:
    """The pediatric info with its authored card dropped, so the card is PENDING."""
    text = "\n".join(
        line for line in PEDIATRIC_INFO.read_text().splitlines() if not line.startswith("card =")
    )
    path = root / "info.toml"
    path.write_text(text)
    return path


def _summary(path: Path) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["metric", "cohort", "n", "n_missing", "mean", "sd", "sem",
             "ci95_lo", "ci95_hi", "median", "iqr_lo", "iqr_hi", "min", "max"]
        )
        writer.writerow(["sleep_sensitivity", "cv", "202", "0", "96.5", "5.5",
                         "0.4", "95.7", "97.3", "97.9", "96.4", "99.0", "41.0", "100.0"])
        writer.writerow(["wake_specificity", "cv", "202", "0", "68.9", "21.0",
                         "1.5", "66.0", "71.8", "72.4", "56.0", "85.4", "6.6", "100.0"])
    return path


def test_bundle_has_every_required_file_and_verifies(tmp_path: Path) -> None:
    phase = _phase(tmp_path)

    bundle = build_release_bundle(
        phase, tmp_path / "release", version="v1.0.0", metrics_summary=_summary(tmp_path / "s.csv")
    )

    for name in BUNDLE_FILES:
        assert (tmp_path / "release" / name).is_file(), name
    verify_bundle(tmp_path / "release")
    assert bundle.checkpoint_sha256
    assert bundle.input_shape == (8, 3, 2)


def test_config_snapshots_become_the_named_bundle_files(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    assert json.loads((tmp_path / "release" / "config.json").read_text())["d_model"] == 32
    preprocessing = json.loads((tmp_path / "release" / "preprocessing.json").read_text())
    assert preprocessing["fs"] == 12.0
    # The resolved protocol wins over the request when a run wrote both.
    assert "class_names" in json.loads((tmp_path / "release" / "metrics.json").read_text())


def test_provenance_records_the_checkpoint_and_phase_hashes(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    provenance = json.loads((tmp_path / "release" / "provenance.json").read_text())
    assert provenance["package_version"] == "v1.0.0"
    assert provenance["checkpoint_sha256"] == build_release_bundle(
        phase, tmp_path / "release2", version="v1.0.0"
    ).checkpoint_sha256
    assert provenance["split_manifest_sha256"]
    assert provenance["input_shape"] == [8, 3, 2]


@needs_pediatric_info
def test_bundle_builds_with_the_real_info_and_card(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(
        phase,
        tmp_path / "release",
        version="v1.0.0",
        metrics_summary=_summary(tmp_path / "s.csv"),
        info=PEDIATRIC_INFO,
    )
    verify_bundle(tmp_path / "release")


def test_bundle_builds_without_info_or_card(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")
    verify_bundle(tmp_path / "release")


def test_pediatric_glance_score_never_substitutes_an_adult_cohort() -> None:
    adult = {"SleepAccel": {"kappa_sleep_wake": {"mean": 0.49, "n": 30}}}
    cv_and_adult = {
        "SleepAccel": {"kappa_sleep_wake": {"mean": 0.49, "n": 30}},
        "cv": {"kappa_sleep_wake": {"mean": 0.65, "n": 202}},
    }

    assert _glance_score(adult, require_cv=True) is None
    assert _glance_score(cv_and_adult, require_cv=True) is not None


@needs_pediatric_info
def test_card_discovers_the_phases_own_results_and_draws_a_grid(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    phase = _phase(tmp_path)
    _results_csv(phase, "SleepAccel", "final_seed_0")
    # Sibling folds are pooled into the cv cohort without being named.
    for i in range(2):
        fold = phase.parent / f"fold{i}"
        fold.mkdir()
        _results_csv(fold, f"BiDozeWeaverFold{i}", f"fold{i}_seed_0")

    bundle = build_release_bundle(
        phase, tmp_path / "release", version="v1.0.0", info=_info_without_card(tmp_path)
    )

    assert "assets/metrics_grid.png" in bundle.files
    assert (tmp_path / "release" / "assets" / "metrics_grid.png").is_file()
    # The grid travels checksummed with the rest of the bundle.
    verify_bundle(tmp_path / "release")
    checksums = (tmp_path / "release" / "checksums.sha256").read_text()
    assert "assets/metrics_grid.png" in checksums


def test_no_grid_is_bundled_when_no_results_csv_exists(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    bundle = build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    assert "assets/metrics_grid.png" not in bundle.files


def test_authored_card_replaces_the_pending_marker(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    authored = tmp_path / "card.md"
    authored.write_text(
        "## Intended use\n\nWrist-worn pediatric sleep staging. Not a medical device.\n"
    )

    bundle = build_release_bundle(
        phase, tmp_path / "release", version="v1.0.0", card=authored
    )

    verify_bundle(tmp_path / "release")


def test_card_images_are_bundled_checksummed_and_uploaded(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    authored = tmp_path / "card.md"
    authored.write_text(
        "## How it stacks up\n\n![Metrics grid](figure.png)\n\n"
        "External: ![x](https://example.com/x.png)\n"
    )

    bundle = build_release_bundle(
        phase, tmp_path / "release", version="v1.0.0", card=authored
    )

    bundled = tmp_path / "release" / "assets" / "figure.png"
    assert bundled.read_bytes() == figure.read_bytes()
    # Remote images are not part of the bundle.
    assert "assets/figure.png" in bundle.files
    assert not any("example.com" in name for name in bundle.files)
    verify_bundle(tmp_path / "release")

    from arcasleep.artifacts.publish import publish_to_hub

    hub = _FakeHub()
    publish_to_hub(tmp_path / "release", "arcascope/arcasleep-bidoze-pediatric", client=hub)
    patterns = hub.folders[0]["allow_patterns"]
    assert "assets/figure.png" in patterns
    assert "README.md" in patterns


def test_card_with_a_missing_image_is_refused(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    authored = tmp_path / "card.md"
    authored.write_text("![missing](nope.png)\n")

    with pytest.raises(FileNotFoundError):
        build_release_bundle(
            phase, tmp_path / "release", version="v1.0.0", card=authored
        )


def test_preview_renders_card_to_standalone_html(tmp_path: Path) -> None:
    pytest.importorskip("markdown_it")
    from arcasleep.artifacts.publish import render_card_html

    phase = _phase(tmp_path)
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    authored = tmp_path / "card.md"
    authored.write_text("# Custom\n\n![f](figure.png)\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")

    build_release_bundle(
        phase, tmp_path / "release", version="v1.0.0", card=authored, repo_id="arcascope/arcasleep-bidoze-pediatric"
    )
    html = render_card_html(tmp_path / "release")

    assert html


def test_evaluation_defaults_to_the_phase_dir_not_a_validation_sibling(
    tmp_path: Path,
) -> None:
    """A phase's evaluation.json sits in the phase, not under a run-level sibling."""
    from arcasleep.artifacts.publish import _evaluation_for

    phase = _phase(tmp_path)
    # A run-level validation/evaluation.json must be ignored: there is no
    # production pattern that puts a phase's evaluation there.
    sibling = tmp_path / "validation"
    sibling.mkdir()
    (sibling / "evaluation.json").write_text(json.dumps({"subject_count": 999}))
    assert _evaluation_for(phase, None) is None

    (phase / "evaluation.json").write_text(json.dumps({"subject_count": 7}))
    assert _evaluation_for(phase, None)["subject_count"] == 7


def test_checkpoint_is_found_when_model_outputs_is_missing(tmp_path: Path) -> None:
    """A copied phase may not keep its .pisces in model_outputs/."""
    from arcasleep.artifacts.publish import _resolve_checkpoint

    phase = _phase(tmp_path)
    nested = phase / "checkpoints"
    nested.mkdir()
    checkpoint = next((phase / "model_outputs").glob("*.pisces"))
    checkpoint.rename(nested / checkpoint.name)
    (phase / "model_outputs").rmdir()

    assert _resolve_checkpoint(phase, None) == nested / checkpoint.name


def test_missing_checkpoint_names_the_problem(tmp_path: Path) -> None:
    from arcasleep.artifacts.publish import _resolve_checkpoint

    phase = _phase(tmp_path)
    next((phase / "model_outputs").glob("*.pisces")).unlink()

    with pytest.raises(FileNotFoundError):
        _resolve_checkpoint(phase, None)


def test_rebuild_requires_overwrite(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    with pytest.raises(FileExistsError):
        build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    build_release_bundle(phase, tmp_path / "release", version="v1.0.0", overwrite=True)


def test_tampering_is_caught(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    (tmp_path / "release" / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_bundle(tmp_path / "release")


def test_upload_gate_blocks_an_unreviewed_card(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from arcasleep.cli import _publish

    phase = _phase(tmp_path)
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(tmp_path / "release"), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None, info=None, preview=None,
        repo="arcascope/bidoze", revision=None, private=False, dry_run=True,
        allow_unreviewed=False, overwrite=False, family="bidoze",
    )

    with pytest.raises(SystemExit):
        _publish(args)


def test_dry_run_upload_does_not_need_huggingface_hub(tmp_path: Path) -> None:
    import argparse

    from arcasleep.cli import _publish

    phase = _phase(tmp_path)
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(tmp_path / "release"), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None, info=None, preview=None,
        repo="arcascope/bidoze", revision=None, private=False, dry_run=True,
        allow_unreviewed=True, overwrite=False, family="bidoze",
    )

    assert _publish(args) == 0


def test_preview_skips_the_upload_even_with_a_repo(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("markdown_it")
    import argparse

    from arcasleep import cli

    calls: list[dict] = []
    monkeypatch.setattr(
        "arcasleep.artifacts.publish.publish_to_hub",
        lambda *a, **k: calls.append(k),
    )

    phase = _phase(tmp_path)
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(tmp_path / "release"), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None, info=None,
        preview=str(tmp_path / "card.html"),
        repo="arcascope/bidoze", revision=None, private=False, dry_run=False,
        allow_unreviewed=False, overwrite=False, family="bidoze",
    )

    assert cli._publish(args) == 0
    assert (tmp_path / "card.html").is_file()
    assert calls == []


def test_preview_into_a_directory_writes_card_html(tmp_path: Path) -> None:
    pytest.importorskip("markdown_it")
    import argparse

    from arcasleep.cli import _publish, build_parser

    phase = _phase(tmp_path)
    target = tmp_path / "release"
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(target), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None, info=None,
        preview=str(target), repo=None, revision=None, private=False, dry_run=False,
        allow_unreviewed=False, overwrite=False, family="bidoze",
    )

    assert _publish(args) == 0
    assert (target / "card.html").is_file()

    # A bare --preview is accepted and defaults to the bundle directory.
    parsed = build_parser().parse_args(
        ["bidoze", "publish", "-t", str(phase), "-o", str(target), "--version", "v1.0.0",
         "--overwrite", "--preview"]
    )
    assert parsed.preview == ""
    assert _publish(parsed) == 0
    assert (target / "card.html").is_file()


def test_load_bundle_model_reproduces_the_checkpoint(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    from_bundle = load_bundle_model(tmp_path / "release")
    from_checkpoint = BiDozeNetModel.loadz(
        phase / "model_outputs" / "model_final_seed_0.pisces", input_shape=(8, 3, 2)
    )
    x = np.random.default_rng(0).standard_normal((1, 8, 3, 2)).astype(np.float32)

    assert np.array_equal(from_bundle.infer(x), from_checkpoint.infer(x))


def test_safetensors_export_refuses_an_unbuilt_model(tmp_path: Path) -> None:
    model = BiDozeNetModel(BiDozeNetConfig.from_dict(SMALL_CONFIG))
    with pytest.raises(ValueError):
        model.save_weights_safetensors(tmp_path / "model.safetensors")


class _FakeHub:
    def __init__(self) -> None:
        self.repos: list[tuple[str, bool]] = []
        self.folders: list[dict] = []

    def create_repo(self, *, repo_id, repo_type, private, exist_ok):
        self.repos.append((repo_id, private))

    def upload_folder(self, **kwargs):
        self.folders.append(kwargs)


def test_upload_is_one_atomic_folder_commit(tmp_path: Path) -> None:
    from arcasleep.artifacts.publish import publish_to_hub

    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")
    hub = _FakeHub()

    publish_to_hub(tmp_path / "release", "arcascope/bidoze", client=hub)

    assert hub.repos == [("arcascope/bidoze", False)]
    assert len(hub.folders) == 1
    assert set(hub.folders[0]["allow_patterns"]) == set(BUNDLE_FILES)
    assert hub.folders[0]["commit_message"] == "Release release"



def test_upload_refuses_a_tampered_bundle(tmp_path: Path) -> None:
    from arcasleep.artifacts.publish import publish_to_hub

    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")
    (tmp_path / "release" / "README.md").write_text("tampered")

    with pytest.raises(ValueError):
        publish_to_hub(tmp_path / "release", "arcascope/bidoze", client=_FakeHub())


def test_release_info_rejects_unknown_keys_and_missing_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('audiance = "typo"\n')
    with pytest.raises(ValueError):
        load_release_info(bad)
    with pytest.raises(FileNotFoundError):
        load_release_info(tmp_path / "missing.toml")


@needs_pediatric_info
def test_info_names_the_authored_card_and_card_flag_overrides_it(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "a", version="v1.0.0", info=PEDIATRIC_INFO)

    figure = tmp_path / "figure.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    other = tmp_path / "other.md"
    other.write_text("![f](figure.png)\n")
    bundle = build_release_bundle(
        phase, tmp_path / "b", version="v1.0.0", info=PEDIATRIC_INFO, card=other
    )
    assert "assets/figure.png" in bundle.files

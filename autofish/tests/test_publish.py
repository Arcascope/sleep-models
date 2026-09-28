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

from autofish.artifacts.publish import (
    BUNDLE_FILES,
    CARD_REVIEW_PENDING,
    _glance_score,
    _metric_label,
    build_release_bundle,
    load_bundle_model,
    verify_bundle,
)
from autofish.model import AutofishNetConfig, AutofishNetModel

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
    config = AutofishNetConfig.from_dict(SMALL_CONFIG)
    model = AutofishNetModel(config)
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


def test_card_carries_metrics_and_a_pending_marker_without_a_card(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(
        phase, tmp_path / "release", version="v1.0.0", metrics_summary=_summary(tmp_path / "s.csv")
    )

    card = (tmp_path / "release" / "README.md").read_text()
    assert CARD_REVIEW_PENDING in card
    assert "sleep_sensitivity" in card
    assert "Sleep sensitivity (%) — higher is better" in card
    assert "Metrics compare model predictions with reference PSG scoring" in card
    assert "not indicate improvement over a control or baseline model" in card
    # The card leads with the raw-accelerometer path, not a spectrogram API.
    assert "aged 5 to 18" in card
    assert "autofish predict autofish-pediatric" in card
    assert "from autofish.predict import Predictor" in card
    assert "never fill them with zeros" in card
    assert "At least 12 Hz" in card
    # Nothing on the card promises an interface the package does not have.
    assert "opportunity" not in card
    assert "## Using model outputs" not in card
    assert card.count(CARD_REVIEW_PENDING) == 2  # the glance row and one banner
    assert "[review required]" not in card
    assert "95.7 to 97.3" in card
    # The card names the built-for repository, not a hard-coded one.
    assert card.lstrip().startswith("---")
    assert "license: mit" in card
    assert "# arcascope/autofish-pediatric · v1.0.0" in card


def test_metric_labels_spell_out_direction_without_arrows() -> None:
    assert _metric_label("wake_specificity") == "Wake specificity (%) — higher is better"
    assert _metric_label("waso_mape") == "WASO error (%) — lower is better"
    assert _metric_label("tst_error_minutes") == "TST error (min)"


def test_pediatric_glance_score_never_substitutes_an_adult_cohort() -> None:
    adult = {"SleepAccel": {"kappa_sleep_wake": {"mean": 0.49, "n": 30}}}
    cv_and_adult = {
        "SleepAccel": {"kappa_sleep_wake": {"mean": 0.49, "n": 30}},
        "cv": {"kappa_sleep_wake": {"mean": 0.65, "n": 202}},
    }

    assert _glance_score(adult, require_cv=True) is None
    assert "0.65" in _glance_score(cv_and_adult, require_cv=True)
    assert "cross-validation" in _glance_score(cv_and_adult, require_cv=True)


def test_card_discovers_the_phases_own_results_and_draws_a_grid(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    phase = _phase(tmp_path)
    _results_csv(phase, "SleepAccel", "final_seed_0")
    # Sibling folds are pooled into the cv cohort without being named.
    for i in range(2):
        fold = phase.parent / f"fold{i}"
        fold.mkdir()
        _results_csv(fold, f"AutofishWeaverFold{i}", f"fold{i}_seed_0")

    bundle = build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    card = (tmp_path / "release" / "README.md").read_text()
    assert "assets/metrics_grid.png" in card
    assert "No results CSV" not in card
    assert "### SleepAccel" in card
    assert "### Cross-validation" in card
    # The headline table leads with the pooled CV cohort, mean over its CI.
    assert card.index("Cross-validation<br>") < card.index("SleepAccel<br>")
    for heading in ("## Quickstart", "## Performance", "## Input requirements", "## Outputs"):
        assert heading in card
    # The grid is the centrepiece: it comes before every table.
    assert card.index("assets/metrics_grid.png") < card.index("Cross-validation<br>")
    assert "_count" not in card
    assert "five fixed, participant-level folds" in card
    assert "balanced across no-diagnosis, mild, moderate, and severe" in card
    assert "all Weaver data" in card
    assert "`gap`" not in card  # a coverage flag, not a stage
    assert "M parameters" in card
    assert "assets/metrics_grid.png" in bundle.files
    assert (tmp_path / "release" / "assets" / "metrics_grid.png").is_file()
    # The grid travels checksummed with the rest of the bundle.
    verify_bundle(tmp_path / "release")
    checksums = (tmp_path / "release" / "checksums.sha256").read_text()
    assert "assets/metrics_grid.png" in checksums


def test_card_says_so_when_no_results_csv_exists(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    card = (tmp_path / "release" / "README.md").read_text()
    assert "No results CSV" in card
    assert "![Metric comparison]" not in card


def test_authored_card_replaces_the_pending_marker(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    authored = tmp_path / "card.md"
    authored.write_text(
        "## Intended use\n\nWrist-worn pediatric sleep staging. Not a medical device.\n"
    )

    bundle = build_release_bundle(
        phase, tmp_path / "release", version="v1.0.0", card=authored
    )

    card = (tmp_path / "release" / "README.md").read_text()
    assert CARD_REVIEW_PENDING not in card
    assert "Not a medical device" in card
    assert bundle.approved
    # Authored prose sits between the title and the generated facts.
    assert card.index("Not a medical device") < card.index("## Model")


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
    card = (tmp_path / "release" / "README.md").read_text()
    assert "assets/figure.png" in card
    # Remote images are left as-is and are not part of the bundle.
    assert "https://example.com/x.png" in card
    assert "assets/figure.png" in bundle.files
    verify_bundle(tmp_path / "release")

    from autofish.artifacts.publish import publish_to_hub

    hub = _FakeHub()
    publish_to_hub(tmp_path / "release", "arcascope/autofish-pediatric", client=hub)
    patterns = hub.folders[0]["allow_patterns"]
    assert "assets/figure.png" in patterns
    assert "README.md" in patterns


def test_card_with_a_missing_image_is_refused(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    authored = tmp_path / "card.md"
    authored.write_text("![missing](nope.png)\n")

    with pytest.raises(FileNotFoundError, match="nope.png"):
        build_release_bundle(
            phase, tmp_path / "release", version="v1.0.0", card=authored
        )


def test_preview_renders_card_to_standalone_html(tmp_path: Path) -> None:
    pytest.importorskip("markdown_it")
    from autofish.artifacts.publish import render_card_html

    phase = _phase(tmp_path)
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    authored = tmp_path / "card.md"
    authored.write_text("# Custom\n\n![f](figure.png)\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")

    build_release_bundle(phase, tmp_path / "release", version="v1.0.0", card=authored)
    html = render_card_html(tmp_path / "release")

    assert html.startswith("<!doctype html>")
    assert "<table>" in html
    assert "data:image/png;base64," in html
    # The Hub frontmatter is metadata, not page content.
    assert "license: mit" not in html
    # The page is titled from the card's generated heading, not authored prose.
    assert "<title>arcascope/autofish-pediatric · v1.0.0</title>" in html
    assert "—" not in html


def test_evaluation_defaults_to_the_phase_dir_not_a_validation_sibling(
    tmp_path: Path,
) -> None:
    """A phase's evaluation.json sits in the phase, not under a run-level sibling."""
    from autofish.artifacts.publish import _evaluation_for

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
    from autofish.artifacts.publish import _resolve_checkpoint

    phase = _phase(tmp_path)
    nested = phase / "checkpoints"
    nested.mkdir()
    checkpoint = next((phase / "model_outputs").glob("*.pisces"))
    checkpoint.rename(nested / checkpoint.name)
    (phase / "model_outputs").rmdir()

    assert _resolve_checkpoint(phase, None) == nested / checkpoint.name


def test_missing_checkpoint_names_the_problem(tmp_path: Path) -> None:
    from autofish.artifacts.publish import _resolve_checkpoint

    phase = _phase(tmp_path)
    next((phase / "model_outputs").glob("*.pisces")).unlink()

    with pytest.raises(FileNotFoundError, match="expected one .pisces"):
        _resolve_checkpoint(phase, None)


def test_rebuild_requires_overwrite(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    with pytest.raises(FileExistsError, match="already holds files"):
        build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    build_release_bundle(phase, tmp_path / "release", version="v1.0.0", overwrite=True)


def test_tampering_is_caught(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    (tmp_path / "release" / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(tmp_path / "release")


def test_upload_gate_blocks_an_unreviewed_card(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from autofish.cli import _publish

    phase = _phase(tmp_path)
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(tmp_path / "release"), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None, preview=None,
        repo="arcascope/autofish", revision=None, private=False, dry_run=True,
        allow_unreviewed=False, overwrite=False,
    )

    with pytest.raises(SystemExit, match="PENDING review"):
        _publish(args)


def test_dry_run_upload_does_not_need_huggingface_hub(tmp_path: Path) -> None:
    import argparse

    from autofish.cli import _publish

    phase = _phase(tmp_path)
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(tmp_path / "release"), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None, preview=None,
        repo="arcascope/autofish", revision=None, private=False, dry_run=True,
        allow_unreviewed=True, overwrite=False,
    )

    assert _publish(args) == 0


def test_preview_skips_the_upload_even_with_a_repo(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("markdown_it")
    import argparse

    from autofish import cli

    calls: list[dict] = []
    monkeypatch.setattr(
        "autofish.artifacts.publish.publish_to_hub",
        lambda *a, **k: calls.append(k),
    )

    phase = _phase(tmp_path)
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(tmp_path / "release"), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None,
        preview=str(tmp_path / "card.html"),
        repo="arcascope/autofish", revision=None, private=False, dry_run=False,
        allow_unreviewed=False, overwrite=False,
    )

    assert cli._publish(args) == 0
    assert (tmp_path / "card.html").is_file()
    assert calls == []


def test_preview_into_a_directory_writes_card_html(tmp_path: Path) -> None:
    pytest.importorskip("markdown_it")
    import argparse

    from autofish.cli import _publish, build_parser

    phase = _phase(tmp_path)
    target = tmp_path / "release"
    args = argparse.Namespace(
        training_dir=str(phase), output_dir=str(target), version="v1.0.0",
        checkpoint=None, evaluation=None, metrics_summary=None, card=None,
        preview=str(target), repo=None, revision=None, private=False, dry_run=False,
        allow_unreviewed=False, overwrite=False,
    )

    assert _publish(args) == 0
    assert (target / "card.html").is_file()

    # A bare --preview is accepted and defaults to the bundle directory.
    parsed = build_parser().parse_args(
        ["publish", "-t", str(phase), "-o", str(target), "--version", "v1.0.0",
         "--overwrite", "--preview"]
    )
    assert parsed.preview == ""
    assert _publish(parsed) == 0
    assert (target / "card.html").is_file()


def test_load_bundle_model_reproduces_the_checkpoint(tmp_path: Path) -> None:
    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")

    from_bundle = load_bundle_model(tmp_path / "release")
    from_checkpoint = AutofishNetModel.loadz(
        phase / "model_outputs" / "model_final_seed_0.pisces", input_shape=(8, 3, 2)
    )
    x = np.random.default_rng(0).standard_normal((1, 8, 3, 2)).astype(np.float32)

    assert np.array_equal(from_bundle.infer(x), from_checkpoint.infer(x))


def test_safetensors_export_refuses_an_unbuilt_model(tmp_path: Path) -> None:
    model = AutofishNetModel(AutofishNetConfig.from_dict(SMALL_CONFIG))
    with pytest.raises(ValueError, match="unbuilt"):
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
    from autofish.artifacts.publish import publish_to_hub

    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")
    hub = _FakeHub()

    publish_to_hub(tmp_path / "release", "arcascope/autofish", client=hub)

    assert hub.repos == [("arcascope/autofish", False)]
    assert len(hub.folders) == 1
    assert set(hub.folders[0]["allow_patterns"]) == set(BUNDLE_FILES)
    assert hub.folders[0]["commit_message"] == "Release release"



def test_upload_refuses_a_tampered_bundle(tmp_path: Path) -> None:
    from autofish.artifacts.publish import publish_to_hub

    phase = _phase(tmp_path)
    build_release_bundle(phase, tmp_path / "release", version="v1.0.0")
    (tmp_path / "release" / "README.md").write_text("tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        publish_to_hub(tmp_path / "release", "arcascope/autofish", client=_FakeHub())

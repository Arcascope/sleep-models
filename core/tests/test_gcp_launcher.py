"""Local output routing for the GCP launcher; no cloud resources are created."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "us-central1-docker.pkg.dev/example-project/arcasleep-train/bidoze@sha256:" + "a" * 64


@pytest.fixture
def launcher():
    path = ROOT / "deploy/scripts/run_task_gcp.py"
    spec = importlib.util.spec_from_file_location("gcp_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Cloud:
    def __init__(self, run_id="unit-run", phase="random/final", training_exit=0):
        self.run_id = run_id
        self.phase = phase
        self.training_exit = training_exit
        self.calls = []
        self.startup = ""

    def __call__(self, command, **kwargs):
        assert command[0] == "gcloud"
        self.calls.append(command)
        args = command[1:]
        if args[:4] == ["artifacts", "docker", "images", "describe"]:
            content = json.dumps({"image_summary": {"fully_qualified_digest": IMAGE}})
        elif args[:3] == ["compute", "instances", "list"]:
            content = ""
        elif args[:3] == ["compute", "instances", "create"]:
            metadata = next(a for a in args if a.startswith("--metadata-from-file="))
            self.startup = Path(metadata.split("startup-script=", 1)[1]).read_text()
            content = "[]"
        elif args[:3] == ["compute", "instances", "get-serial-port-output"]:
            content = (
                f"ARCASLEEP_FINISHED run={self.run_id} phase={self.phase} "
                f"training_exit={self.training_exit} upload_exit=0\n"
            )
        elif args[:3] == ["compute", "instances", "delete"]:
            content = ""
        elif args[:3] == ["storage", "rsync", "--recursive"]:
            destination = Path(args[4])
            (destination / "training").mkdir()
            (destination / "training" / "downloaded.txt").write_text("cloud result")
            content = ""
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=content, stderr="")


def arguments(*extra):
    return [
        "run_task_gcp.py",
        "bidoze",
        IMAGE,
        "random/final",
        "--zones",
        "us-central1-a",
        "--run-id",
        "unit-run",
        *extra,
    ]


@pytest.mark.parametrize("relative", [False, True])
def test_results_root_routes_downloads_metadata_and_phase_logs(
    launcher, monkeypatch, tmp_path, relative
):
    cloud = Cloud()
    monkeypatch.setattr(launcher.subprocess, "run", cloud)
    monkeypatch.chdir(tmp_path)
    target = Path("chosen results") if relative else tmp_path / "chosen results"
    monkeypatch.setattr(sys, "argv", arguments("--results-dir", str(target)))
    assert launcher.main() == 0

    output = tmp_path / "chosen results" / "unit-run"
    state = json.loads((output / "gcp-run.json").read_text())
    assert state["results_dir"] == str(output.parent)
    assert state["phases"]["random/final"]["stage"] == "succeeded"
    assert state["phases"]["random/final"]["vm_deleted"] is True
    assert (output / "training/downloaded.txt").read_text() == "cloud result"
    assert (output / "gcp/random/final/serial.log").exists()
    assert "chosen results" not in cloud.startup
    assert "_RESULTS_DIR=/results" in cloud.startup
    sync = next(c for c in cloud.calls if c[1:4] == ["storage", "rsync", "--recursive"])
    assert sync[-2] == str(output)


def test_results_root_dry_run_creates_no_directories(launcher, monkeypatch, tmp_path, capsys):
    cloud = Cloud()
    monkeypatch.setattr(launcher.subprocess, "run", cloud)
    target = tmp_path / "not created"
    monkeypatch.setattr(sys, "argv", arguments("--results-dir", str(target), "--dry-run"))
    assert launcher.main() == 0
    assert str(target / "unit-run") in capsys.readouterr().out
    assert not target.exists()
    assert not any(c[1:4] == ["compute", "instances", "create"] for c in cloud.calls)


def test_default_results_root_stays_under_checkout(launcher, monkeypatch, capsys):
    cloud = Cloud()
    monkeypatch.setattr(launcher.subprocess, "run", cloud)
    monkeypatch.setattr(sys, "argv", arguments("--dry-run"))
    assert launcher.main() == 0
    assert str(ROOT / "bidoze/results/unit-run") in capsys.readouterr().out


def test_failed_phase_keeps_diagnostics_in_requested_results_root(launcher, monkeypatch, tmp_path):
    cloud = Cloud(training_exit=1)
    monkeypatch.setattr(launcher.subprocess, "run", cloud)
    monkeypatch.setattr(sys, "argv", arguments("--results-dir", str(tmp_path)))
    assert launcher.main() == 1
    state = json.loads((tmp_path / "unit-run/gcp-run.json").read_text())
    assert state["phases"]["random/final"]["stage"] == "failed"
    assert state["phases"]["random/final"]["vm_deleted"] is True
    assert (tmp_path / "unit-run/gcp/random/final/serial.log").exists()

"""Run diagnostics with relocated inputs from outside the repository."""

import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from full import findings, reuse_cache

HERE = Path(__file__).resolve().parent


def run(script, cwd, *args):
    result = subprocess.run(
        [sys.executable, str(HERE / script), *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    return result


def test_conditioned_errors_cli_uses_relocated_inputs(tmp_path):
    experiment = tmp_path / "arbitrary experiment name"
    cache = experiment / "pilot/cache"
    cache.mkdir(parents=True)
    endpoints = 60.0 + np.arange(80) * 15.0
    np.savez_compressed(
        cache / "child_n1_apple.npz",
        endpoint=endpoints,
        eligible=np.ones(80, dtype=bool),
        reference=np.full(80, 75.0),
        peak_narrow=np.full(80, 70.0),
        peak_wide=np.full(80, 80.0),
        motion=np.arange(80.0),
        peak_narrow_clarity=np.arange(80.0, 0.0, -1.0),
        metadata=np.array(json.dumps({"participant": "child", "device": "apple"})),
    )
    old = tmp_path / "earlier predictions.csv.gz"
    pd.DataFrame(
        {
            "cohort": ["SleepAccel"],
            "subject_id": ["adult"],
            "hr_est": [65.0],
            "watch_hr": [64.0],
            "still_clear": [True],
        }
    ).to_csv(old, index=False)
    output = tmp_path / "new reports"
    run(
        "conditioned_errors.py",
        tmp_path,
        "--weaver-run",
        experiment,
        "--old-epochs",
        old,
        "--output",
        output,
    )
    summary = pd.read_csv(output / "conditioned_error_comparison.csv")
    narrow = summary[
        summary.dataset.eq("Weaver") & summary.subset.eq("all") & summary.method.eq("peak_narrow")
    ].iloc[0]
    assert narrow.windows == 41
    assert narrow.participant_macro_mae == 5.0
    assert summary[summary.dataset.eq("SleepAccel_Oct6")].participant_macro_mae.eq(1.0).all()


def test_reference_audit_cli_uses_selected_people_and_relocated_inputs(tmp_path):
    data = tmp_path / "relocated dataset"
    (data / "raw").mkdir(parents=True)
    pd.DataFrame(
        {
            "Kubios Time": np.arange(75.0),
            "Kubios None Mean HR": np.full(75, 80.0),
            "Kubios Medium Mean HR": np.full(75, 81.0),
            "Apple Time": np.arange(75.0),
            "Apple Heart Rate": np.full(75, 82.0),
        }
    ).to_csv(data / "raw/child_n1.csv", index=False)
    predictions = tmp_path / "predictions.csv.gz"
    pd.DataFrame(
        {
            "participant": ["child", "child", "excluded"],
            "night": [1, 1, 1],
            "device": ["apple"] * 3,
            "endpoint": [60.0, 75.0, 60.0],
            "reference": [80.0] * 3,
            "peak_narrow": [79.0, 83.0, 999.0],
            "peak_wide": [79.0, 83.0, 999.0],
            "seqsmoother": [80.0, 84.0, 999.0],
        }
    ).to_csv(predictions, index=False)
    splits = tmp_path / "selection.json"
    splits.write_text(json.dumps({"folds": [{"test": ["child"]}]}))
    output = tmp_path / "reference reports"
    run(
        "reference_audit.py",
        tmp_path,
        "--data",
        data,
        "--predictions",
        predictions,
        "--splits",
        splits,
        "--output",
        output,
    )
    summary = pd.read_csv(output / "weaver_reference_summary.csv")
    optical = summary[
        summary.reference_source.eq("apple_optical_hr")
        & summary.reduction.eq("mean")
        & summary.method.eq("peak_narrow")
    ].iloc[0]
    assert optical.windows == 2
    assert optical.participants == 1
    assert optical.participant_macro_mae == 2.0
    assert optical.reference_difference_mean == 2.0
    assert json.loads((output / "weaver_hr_column_headers.json").read_text())["appleheartrate"] == 1


def test_cross_filesystem_cache_reuse_preserves_contents(tmp_path, monkeypatch):
    source = tmp_path / "source.npz"
    source.write_bytes(b"cached experiment data")
    destination = tmp_path / "destination.npz"

    def cross_filesystem(*args):
        raise OSError(errno.EXDEV, "different filesystems")

    monkeypatch.setattr("full.os.link", cross_filesystem)
    reuse_cache(source, destination)
    assert source.read_bytes() == destination.read_bytes() == b"cached experiment data"


def test_logging_can_be_disabled_without_a_research_log(tmp_path):
    findings(tmp_path / "unused output", "test stage", None)
    assert list(tmp_path.iterdir()) == []

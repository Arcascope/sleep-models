"""Reading ``inputs.json`` and ``preprocessing.json`` into a ``ProcessingConfig``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pisces_lite")

from arcasleep.training.config import load_inputs, processing_config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
COMMITTED_INPUTS = [REPO / "bidoze" / "config" / "inputs.json", REPO / "unet" / "config" / "inputs.json"]


@pytest.mark.parametrize("path", COMMITTED_INPUTS, ids=lambda p: p.parent.parent.name)
@pytest.mark.parametrize("extra", [{}, {"jerk_lowpass_hz": 6.0}, {"jerk_lowpass_hz": None}])
def test_committed_inputs_build(path: Path, extra: dict, tmp_path: Path) -> None:
    settings = {**json.loads(path.read_text()), **extra}
    processing_config(settings, "preprocessing.json")
    (tmp_path / "inputs.json").write_text(json.dumps(settings))
    load_inputs(tmp_path / "inputs.json")


@pytest.mark.parametrize("key", ["not_a_setting", "mode"])
def test_a_setting_pisces_lite_does_not_declare_is_refused(key: str, tmp_path: Path) -> None:
    settings = {**json.loads(COMMITTED_INPUTS[0].read_text()), key: 1.0}
    with pytest.raises(ValueError):
        processing_config(settings, "preprocessing.json")
    (tmp_path / "inputs.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        load_inputs(tmp_path / "inputs.json")

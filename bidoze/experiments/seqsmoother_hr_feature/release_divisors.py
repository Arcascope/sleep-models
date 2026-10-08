"""The adult release's stage divisors: one per SleepAccel fold model, and the bundle's.

Needs the adult code (``arcasleep_bidoze.adult``) on PYTHONPATH, i.e. the ``adult-formulation``
checkout's ``core/src`` and ``bidoze/src``. The divisor per fold is computed as
``card_metrics.py`` computes it: from the fold model's training nights and BIDSleep.

Usage: python release_divisors.py <adult-devices run> <arm> <adult cache root> <bundle> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from arcasleep.data.cache import discover_cached_subjects
from arcasleep_bidoze.adult import adult_stage_divisor, stage_counts

CACHE_NAME = "12Hz_log_PSD_aligned"
BETA = 0.999


def night_counts(paths: list[Path]) -> np.ndarray:
    return np.array([stage_counts(np.load(p)["y"]) for p in paths])


def main(run: Path, arm: str, adult_root: Path, bundle: Path, out: Path) -> None:
    bidsleep = night_counts([s.path for s in discover_cached_subjects(["bidsleep"], adult_root / CACHE_NAME)["bidsleep"]])
    cache = run / "setup/cache" / arm / CACHE_NAME
    folds = {}
    for k in range(5):
        cv = json.loads((run / f"setup/cv/{arm}/cv_fold{k}.json").read_text())
        paths = [p for d in cv["train_sets"] for p in sorted(cache.glob(f"{d}_subject_*"))]
        folds[k] = adult_stage_divisor(night_counts(paths), bidsleep, BETA).tolist()
    spec = json.loads((bundle / "composite.json").read_text())
    final = [1.0] + [spec["stage_divisor"][n] for n in ("light", "deep", "rem")]
    out.write_text(json.dumps({"folds": folds, "final": final, "arm": arm, "bundle": str(bundle)}, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))

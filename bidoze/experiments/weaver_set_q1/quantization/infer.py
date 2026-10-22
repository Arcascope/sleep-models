"""Frozen-model probabilities for the four accelerometer conditions.

For each arm, draw and seed, loads the Q1 checkpoint (``<run>/training/<arm>/draw<d>/model_outputs``) and
scores the draw's test adults of both cohorts under the four conditions of ``build_features.py``
(one batch of 4 inputs per recording). Saves softmax probabilities, ``(4, epochs, 5)``, to
``<out>/predictions/<arm>/draw<d>/<cohort>_<subject>_seed<s>.npz``; existing files are skipped.
The native condition is checked against the checkpoint's saved prediction CSV
(``predictions/draw<d>_seed_<s>/test<Cohort>_<subject>.csv``): max probability difference must be < 0.025.

Run on the GPU only through the shared lock, e.g.
  flock ~/.cache/arcasleep-gpu.lock python infer.py ... ; or on CPU with JAX_PLATFORMS=cpu.

Usage: python infer.py <run dir> <feature dir> <arm> [<arm> ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adult_devices"))
from arcasleep_bidoze.model import BiDozeNetModel  # noqa: E402
from build_features import DRAW_KEY  # noqa: E402
from score import load_subject  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = [0, 1, 2]
SAVED_PREFIX = {"SleepAccel": "testSleepAccel", "SleepAccel-Clinical": "testClinical"}


def main(run: Path, features: Path, arms: list[str]) -> None:
    draws = json.loads((HERE.parent / "draws.json").read_text())
    for arm in arms:
        for d in range(5):
            fit = run / f"training/{arm}/draw{d}"
            for seed in SEEDS:
                checkpoint = fit / f"model_outputs/model_draw{d}_seed_{seed}.pisces"
                jobs = []
                for cohort, key in DRAW_KEY.items():
                    for sid in draws[f"draw{d}"][key]:
                        dest = run / f"quantized/predictions/{arm}/draw{d}/{cohort}_{sid}_seed{seed}.npz"
                        if (features / cohort / f"{sid}.npz").exists() and not dest.exists():
                            jobs.append((cohort, sid, dest))
                if not jobs:
                    continue
                model = BiDozeNetModel.loadz(checkpoint)
                assert model.config.gap_from_row
                start = time.time()
                for cohort, sid, dest in jobs:
                    with np.load(features / cohort / f"{sid}.npz") as z:
                        x, n = z["inputs"], len(z["y"])
                    p = softmax(np.asarray(model.infer(x)), axis=-1)[:, :n]
                    saved = fit / f"predictions/draw{d}_seed_{seed}/{SAVED_PREFIX[cohort]}_{sid}.csv"
                    _, saved_p = load_subject(saved)
                    diff = float(np.max(np.abs(p[0] - saved_p)))
                    assert diff < 0.025, (arm, d, seed, sid, diff)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dest, probabilities=p, saved_max_difference=diff)
                    print("INFER", arm, d, seed, cohort, sid, "saved_max_diff", round(diff, 5),
                          "elapsed", round(time.time() - start), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3:])

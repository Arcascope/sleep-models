"""Does ``fused.frames`` reproduce the release pipeline's frames in each jerk mode?

Per night and mode: the release's ``features`` under ``modes.jerk_mode`` against
``fused.frames`` with ``fused.plan(..., jerk=mode)``, on the same cleaned samples and origin.
Reports whether the padding frames match, and the largest and 99.9th-percentile absolute
log-PSD difference of the jerk channel over non-padding frames.

Usage: python layer_check.py <data root> <adult bundle> <out csv> <cohort/night> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

import fused
from arcasleep.data.prepare import clean_accelerometer_rows
from arcasleep.predict import load_predictor
from modes import MODES, jerk_mode, load_night


def main(root: Path, bundle: Path, out: Path, names: list[str]) -> None:
    features = load_predictor(bundle).ensemble.features
    rows = []
    for name in names:
        cohort, stem = name.split("/")
        accel, _, origin = load_night(root, cohort, stem)
        clean = clean_accelerometer_rows(accel)
        for mode in MODES:
            with jerk_mode(mode):
                ref, origin_s = features(accel, origin)
            p = fused.plan(clean[:, 0], origin_s, jerk=mode)
            got = np.asarray(fused.frames(*fused.device_plan(p), jnp.asarray(clean[:, 1:], jnp.float32)))
            ref = np.asarray(ref, np.float64)
            got = got[: ref.shape[0]]
            pad = (ref == -1).all(axis=(1, 2))
            diff = np.abs(ref[~pad, :, 1] - got[~pad, :, 1])
            rows.append({"night": name, "mode": mode,
                         "same_padding": bool(np.array_equal(pad, (got == -1).all(axis=(1, 2)))),
                         "max_abs_diff": float(diff.max()), "p999_abs_diff": float(np.percentile(diff, 99.9))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])

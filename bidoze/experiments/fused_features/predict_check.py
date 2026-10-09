"""Do the release's calls change when its frames come from ``fused.frames``?

Per night, ``Predictor.predict`` twice: as released, and with the ensemble's ``features`` replaced
by ``fused.frames`` on the same cleaned samples and origin. Reports the share of epochs with the
same call (``stage_raw``, before the GGIR filter), and the largest change in any stage probability.

Usage: python predict_check.py <data root> <adult bundle> <out csv> <cohort/night> [...]
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

COLUMNS = ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"]


def main(root: Path, bundle: Path, out: Path, names: list[str]) -> None:
    predictor = load_predictor(bundle)
    released = predictor.ensemble.features
    rows = []
    for name in names:
        cohort, night = name.split("/")
        accel = pd.read_csv(root / cohort / "cleaned_accelerometer" / f"{night}.csv", usecols=COLUMNS,
                            dtype="float64")[COLUMNS].to_numpy()
        origin = float(pd.read_csv(root / cohort / "cleaned_psg" / f"{night}.csv").TIMESTAMP.iloc[0] % 30.0)
        def fused_features(acc, org, _released=released):
            ref, origin_s = _released(acc, org)
            clean = clean_accelerometer_rows(np.asarray(acc, np.float64))
            got = fused.frames(*fused.device_plan(fused.plan(clean[:, 0], origin_s)),
                               jnp.asarray(clean[:, 1:], jnp.float32))
            return np.asarray(got, ref.dtype)[: ref.shape[0]], origin_s

        base = predictor.predict(accel, origin)
        predictor.ensemble.features = fused_features
        new = predictor.predict(accel, origin)
        predictor.ensemble.features = released
        p0, p1 = np.asarray(base.probabilities), np.asarray(new.probabilities)
        s0, s1 = np.asarray(base.stage_raw), np.asarray(new.stage_raw)
        rows.append({"night": name, "epochs": len(s0), "same_call": float(np.mean(s0 == s1)),
                     "changed_epochs": int(np.sum(s0 != s1)), "max_abs_dp": float(np.abs(p0 - p1).max()),
                     "mean_abs_dp": float(np.abs(p0 - p1).mean())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])

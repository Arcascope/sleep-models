"""Does ``fused.frames`` reproduce the release's frames, how fast is it, and do gradients flow?

Per night: the release's ``Predictor.features`` against ``fused.frames`` on the same cleaned
samples and origin (the PSG epoch grid). Reports whether the padding frames (all bins −1) match,
the largest absolute difference in log-PSD over finite bins, and its 99.9th percentile.

Timing on the default JAX device: the host plan, the first call (compile) and the median of 5
later calls, with ``block_until_ready``.

Gradient: d/dg of the mean frame value with white noise of RMS g added to every axis.

Usage: python check.py <data root> <adult bundle> <out csv> <cohort/night> [...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import fused
from arcasleep.data.prepare import clean_accelerometer_rows
from arcasleep.predict import load_predictor

COLUMNS = ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"]


def main(root: Path, bundle: Path, out: Path, names: list[str]) -> None:
    predictor = load_predictor(bundle)
    rows = []
    for name in names:
        cohort, night = name.split("/")
        accel = pd.read_csv(root / cohort / "cleaned_accelerometer" / f"{night}.csv", usecols=COLUMNS,
                            dtype="float64")[COLUMNS].to_numpy()
        psg = pd.read_csv(root / cohort / "cleaned_psg" / f"{night}.csv")
        origin = float(psg.TIMESTAMP.iloc[0] % 30.0)
        start = time.perf_counter()
        ref, origin_s = predictor.ensemble.features(accel, origin)
        t_ref = time.perf_counter() - start

        clean = clean_accelerometer_rows(accel)
        start = time.perf_counter()
        p = fused.plan(clean[:, 0], origin_s)
        t_plan = time.perf_counter() - start
        args = fused.device_plan(p)
        xyz = jnp.asarray(clean[:, 1:], jnp.float32)
        start = time.perf_counter()
        got = fused.frames(*args, xyz).block_until_ready()
        t_first = time.perf_counter() - start
        times = []
        for _ in range(5):
            start = time.perf_counter()
            fused.frames(*args, xyz).block_until_ready()
            times.append(time.perf_counter() - start)
        got = np.asarray(got)[: ref.shape[0]]
        ref = np.asarray(ref, np.float64)
        same_nan = bool(np.array_equal((ref == -1).all(axis=(1, 2)), (got == -1).all(axis=(1, 2))))
        both = np.isfinite(ref) & np.isfinite(got) & ~(ref == -1).all(axis=(1, 2))[:, None, None]
        diff = np.abs(ref[both] - got[both])

        noise = jax.random.normal(jax.random.PRNGKey(0), xyz.shape)

        def loss(g):
            f = fused.frames(*args, xyz + g * noise)
            return jnp.mean(f)

        grad = float(jax.grad(loss)(jnp.float32(0.003)))
        rows.append({"night": name, "samples": len(clean), "frames_ref": ref.shape[0],
                     "frames_fused": int(np.asarray(fused.frames(*args, xyz)).shape[0]), "same_padding": same_nan,
                     "max_abs_diff": float(diff.max()), "p999_abs_diff": float(np.percentile(diff, 99.9)),
                     "release_s": t_ref, "plan_s": t_plan, "first_call_s": t_first,
                     "call_s": float(np.median(times)), "grad_at_3mg": grad})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])

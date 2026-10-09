"""Does ``jax_generator`` draw the same noise as ``generator``?

On one night, ``DRAWS`` draws from each: the RMS of the added noise vector (mg) over the night,
and its RMS in 2–6 Hz and in 0.1–0.5 Hz (FFT of the added noise, whole night), plus the time per
draw. The two use different random streams, so the comparison is of the distributions (quantiles).

Writes ``<out>/generator_check.csv`` (one row per draw) and prints quantiles.

Usage: python jax_generator_check.py <data root> <cohort/night> <out>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import generator
import jax_generator
from arcasleep.data.prepare import clean_accelerometer_rows
from survey import load_accel

DRAWS = 40


def summary(noise: np.ndarray, fs: float) -> dict:
    spectrum = np.fft.rfft(noise, axis=0)
    f = np.fft.rfftfreq(noise.shape[0], 1.0 / fs)
    power = (np.abs(spectrum) ** 2).sum(axis=1) * 2 / noise.shape[0] ** 2

    def band(lo, hi):
        return float(np.sqrt(power[(f >= lo) & (f <= hi)].sum()) * 1000)

    return {"rms_mg": float(np.sqrt((noise**2).sum(axis=1).mean()) * 1000),
            "band_2_6_mg": band(2.0, 6.0), "band_01_05_mg": band(0.1, 0.5)}


def main(root: Path, name: str, out: Path) -> None:
    cohort, night = name.split("/")
    accel = clean_accelerometer_rows(load_accel(root, cohort, night))
    t = accel[:, 0] - accel[0, 0]
    fs = 1.0 / float(np.median(np.diff(t)))
    rows = []
    for d in range(DRAWS):
        start = time.perf_counter()
        rng = np.random.default_rng(d)
        noisy = generator.render(generator.sample_spec(rng, fs / 2), accel, rng)
        rows.append({"generator": "numpy", "draw": d, "seconds": time.perf_counter() - start}
                    | summary(noisy[:, 1:] - accel[:, 1:], fs))

    @jax.jit
    def draw(key, t, xyz, n, fs):
        k1, k2 = jax.random.split(key)
        return jax_generator.render(k2, jax_generator.sample_spec(k1, fs / 2), t, xyz, n, fs)

    args = (jnp.asarray(t, jnp.float32), jnp.asarray(accel[:, 1:], jnp.float32), jnp.int32(len(t)), jnp.float32(fs))
    jax.block_until_ready(draw(jax.random.PRNGKey(10_000), *args))
    for d in range(DRAWS):
        start = time.perf_counter()
        noisy = np.asarray(jax.block_until_ready(draw(jax.random.PRNGKey(d), *args)), np.float64)
        rows.append({"generator": "jax", "draw": d, "seconds": time.perf_counter() - start}
                    | summary(noisy - np.asarray(args[1], np.float64), fs))
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "generator_check.csv", index=False, float_format="%.5g")
    print(table.groupby("generator")[["rms_mg", "band_2_6_mg", "band_01_05_mg", "seconds"]]
          .quantile([0.1, 0.5, 0.9]).round(3).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))

"""Checks of the jerk low-pass (``fused.lowpass_taps``, ``fused.channels(xyz, taps)``).

Per night: the filter's gain at a few frequencies; the largest difference between the JAX jerk
and the same FIR (``np.convolve``) followed by ``senpy.compute_jerk``; the largest change in the
``|a|`` frames (should be 0); and, per frame, the change in the jerk channel's median 2–6 Hz
log-PSD, reported as the 10th, 50th and 90th percentiles over frames.

Writes ``<out>/lowpass_check.csv``.

Usage: python lowpass_check.py <data root> <out> <cohort/night> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import senpy

import fused
from arcasleep.data.prepare import clean_accelerometer_rows
from survey import load_accel

GAINS_HZ = (1.0, 4.0, 5.0, 6.0, 7.0, 10.0)


def main(root: Path, out: Path, names: list[str]) -> None:
    rows = []
    for name in names:
        cohort, night = name.split("/")
        a = clean_accelerometer_rows(load_accel(root, cohort, night))
        p = fused.plan(a[:, 0], 0.0)
        taps = fused.lowpass_taps(p.median_fs)
        w, h = np.fft.rfftfreq(1 << 14, 1 / p.median_fs), np.abs(np.fft.rfft(taps, 1 << 14))
        gains = {f"gain_db_{f:g}hz": float(20 * np.log10(np.interp(f, w, h) + 1e-12)) for f in GAINS_HZ if f < p.median_fs / 2}
        xyz = jnp.asarray(a[:, 1:], jnp.float32)
        jerk = np.asarray(fused.channels(xyz, jnp.asarray(taps)))[:, 1]
        smooth = [np.convolve(a[:, k], taps.astype(np.float64), mode="same") for k in (1, 2, 3)]
        ref = np.asarray(senpy.compute_jerk(a[:, 0], *smooth, use_diff=True).jerk)
        plain = np.asarray(fused.frames(*fused.device_plan(p), xyz))
        low = np.asarray(fused.frames(*fused.device_plan(p), xyz, jnp.asarray(taps)))
        valid = p.row_valid
        band = (fused.BINS / fused.WINDOW_S >= 2.0) & (fused.BINS / fused.WINDOW_S <= 6.0)
        drop = np.median(low[valid][:, band, 1], axis=1) - np.median(plain[valid][:, band, 1], axis=1)
        rows.append({"night": name, "fs": p.median_fs, "taps": int((taps != 0).sum())} | gains
                    | {"jerk_max_abs_diff": float(np.abs(jerk - ref).max()), "jerk_rms": float(np.sqrt(np.mean(ref**2))),
                       "abs_frames_max_change": float(np.abs(low[..., 0] - plain[..., 0]).max()),
                       "jerk_band_change_db_p10": float(10 / np.log(10) * np.quantile(drop, 0.1)),
                       "jerk_band_change_db_p50": float(10 / np.log(10) * np.quantile(drop, 0.5)),
                       "jerk_band_change_db_p90": float(10 / np.log(10) * np.quantile(drop, 0.9))})
        print(rows[-1], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "lowpass_check.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])

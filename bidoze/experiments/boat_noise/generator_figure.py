"""The model's own input for one land night, clean and with several random generator draws.

Rows: clean, then one row per seed. Columns: the |a| and jerk inputs of the adult release
(0.1–6 Hz, natural-log PSD shown in dB) over one hour starting at ``start_h``. Every 2-s frame is
its own pixel column (no resampling across frames). The row title lists the draw's components.

Usage: python generator_figure.py <data root> <adult bundle> <night> <out png> <start_h> <seed> [seed ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arcasleep.predict import load_predictor
from generator import render, sample_spec

COLUMNS = ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"]
COHORT = "SleepAccel-Clinical"


def describe(spec) -> str:
    parts = []
    for c in spec:
        p = c.params
        what = {"floor": f"floor {'iso' if c.direction is None else 'dir'} slope {p.get('slope', 0):.1f}",
                "line": f"line {p.get('f0', 0):.2f} Hz", "comb": f"comb {p.get('f0', 0):.2f} Hz",
                "pulses": f"pulses {p.get('rate', 0):.2f}/s", "hull": f"hull {p.get('centre', 0):.2f} Hz"}[c.kind]
        parts.append(f"{what} {c.rms_mg:.1f} mg")
    return "; ".join(parts) or "nothing"


def main(root: Path, bundle: Path, night: str, out: Path, start_h: float, seeds: list[int]) -> None:
    ensemble = load_predictor(bundle).ensemble
    accel = pd.read_csv(root / COHORT / "cleaned_accelerometer" / f"{night}.csv", usecols=COLUMNS,
                        dtype="float64")[COLUMNS].to_numpy()
    psg = pd.read_csv(root / COHORT / "cleaned_psg" / f"{night}.csv")
    origin = float(psg.TIMESTAMP.iloc[0] % 30.0)
    rows = [("clean", accel)]
    for s in seeds:
        rng = np.random.default_rng(s)
        spec = sample_spec(rng, nyquist=0.5 / np.median(np.diff(accel[:, 0])))
        rows.append((f"seed {s}: {describe(spec)}", render(spec, accel)))
    first = int(start_h * 1800)
    fig, axes = plt.subplots(len(rows), 2, figsize=(26, 2.6 * len(rows)), squeeze=False)
    for r, (title, a) in enumerate(rows):
        frames, _ = ensemble.features(a, origin)
        frames = 10 * np.log10(np.e) * np.asarray(frames, float)[first:first + 1800]
        for c, name in enumerate(("|a|", "jerk")):
            ax = axes[r, c]
            ax.imshow(frames[:, :, c].T, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
                      extent=(0, frames.shape[0] * 2 / 60, 0.1, 6.0), vmin=-90, vmax=-30)
            ax.set_ylabel("Hz")
            ax.set_title(f"{name} — {title}", fontsize=9, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(f"minutes after hour {start_h:g}")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), float(sys.argv[5]),
         [int(s) for s in sys.argv[6:]])

"""How much do the release's input frames change when the jerk is divided by the time step?

Per night, the release's ``features`` (pisces-lite and senpy, float64) under each jerk mode, on
the same samples and origin. Frames that are padding in every mode (all bins −1) are left out.
The release's JAX NUFFT is not bit-deterministic, so ``release`` is computed twice: the
difference between the two runs is the noise floor (``rerun_*``). The ``|a|`` channel is checked
to change by less than 1e-3 between modes; for the jerk channel, per frame and bin:

- ``median_dt − release``: should equal ``−2·log(median Δt)`` everywhere; reports the constant
  and the largest deviation from it.
- ``dt − release`` and ``dt − median_dt`` (timestamp irregularity alone): the mean signed
  difference, quantiles of the absolute difference (log-PSD, natural log) over all frame×bin
  cells, and the share of cells changed by more than 0.1 and 1.0.

Writes ``<out>/nights.csv`` (one row per night, with ``modes.timing``), and with ``--frames
<cohort/night> ...`` also ``<out>/frames/<cohort>_<night>.npz`` holding each mode's jerk frames.

Usage: python spectra.py <data root> <adult bundle> <out> <cohort> [...] [--frames <cohort/night> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.data.prepare import clean_accelerometer_rows
from arcasleep.predict import load_predictor
from modes import MODES, jerk_mode, load_night, nights, steps_s, timing

QUANTILES = [50, 95, 99, 99.9]


def summarize(d: np.ndarray, prefix: str) -> dict:
    a = np.abs(d)
    return ({f"{prefix}_mean": float(d.mean())}
            | {f"{prefix}_abs_p{q:g}": float(np.percentile(a, q)) for q in QUANTILES}
            | {f"{prefix}_abs_max": float(a.max()), f"{prefix}_over_0.1": float(np.mean(a > 0.1)),
               f"{prefix}_over_1": float(np.mean(a > 1.0))})


def main(root: Path, bundle: Path, out: Path, night_list: list[tuple[str, str]], keep: set[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    features = load_predictor(bundle).ensemble.features
    rows = []
    for cohort, stem in night_list:
        accel, _, origin = load_night(root, cohort, stem)
        frames = {}
        for mode in MODES:
            with jerk_mode(mode):
                frames[mode], _ = features(accel, origin)
        rerun = np.asarray(features(accel, origin)[0], np.float64)
        f = {m: np.asarray(v, np.float64) for m, v in frames.items()}
        padding = np.all([(v == -1).all(axis=(1, 2)) for v in f.values()], axis=0)
        mag_change = float(np.abs(f["dt"][~padding, :, 0] - f["release"][~padding, :, 0]).max())
        if mag_change > 1e-3:
            raise AssertionError(f"{cohort}/{stem}: the |a| channel changed by {mag_change:.3g}")
        jerk = {m: v[~padding, :, 1] for m, v in f.items()}
        t = clean_accelerometer_rows(accel)[:, 0]
        constant = -2.0 * np.log(np.median(steps_s(t)))
        rows.append({"cohort": cohort, "night": stem, "frames": int((~padding).sum()),
                     "frames_padding_differs": int(np.sum(np.any([(v == -1).all(axis=(1, 2)) for v in f.values()],
                                                                 axis=0) & ~padding)),
                     "offset_expected": constant,
                     "offset_max_deviation": float(np.abs(jerk["median_dt"] - jerk["release"] - constant).max())}
                    | timing(t)
                    | {"mag_change_max": mag_change}
                    | summarize(rerun[~padding, :, 1] - jerk["release"], "rerun")
                    | summarize(jerk["dt"] - jerk["release"], "dt_vs_release")
                    | summarize(jerk["dt"] - jerk["median_dt"], "dt_vs_median_dt"))
        if f"{cohort}/{stem}" in keep:
            (out / "frames").mkdir(exist_ok=True)
            np.savez_compressed(out / "frames" / f"{cohort}_{stem}.npz", **{m: v[..., 1] for m, v in f.items()})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "nights.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    args = sys.argv[1:]
    keep = set(args[args.index("--frames") + 1:]) if "--frames" in args else set()
    args = args[: args.index("--frames")] if "--frames" in args else args
    main(Path(args[0]), Path(args[1]), Path(args[2]), nights(Path(args[0]), args[3:]), keep)

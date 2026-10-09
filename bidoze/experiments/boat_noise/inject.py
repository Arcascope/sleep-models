"""Inject synthetic boat-noise components into clean land nights and score the released adult model.

Each component is a signal ``n(t)`` in g, added to the raw accelerometer of every SleepAccel-Clinical
night before the release's own processing, so leakage, the jerk and the 12-Hz input see it as
they would see a real boat:

- along gravity (``a + n·â``, with ``â = a/|a|`` per sample) the noise shows in |a|;
- perpendicular to gravity (``a + n·û``, ``û ⟂ â``) it shows in the jerk and barely in |a|.

Components (zoology type in brackets) are scaled to an RMS amplitude in mg and are on for the
whole night:

- ``line_0.88``, ``line_1.4`` [3, 8]: sinusoids along gravity;
- ``wander_3.85`` [2]: a sinusoid whose frequency wanders over 3.6–4.1 Hz on 1–10 min scales;
- ``wobble_1.8`` [9]: 1.8 Hz with ±0.15 Hz sinusoidal wobble, 20-s period;
- ``comb_wobble`` [5]: harmonics 1–10 of a 1.2 Hz fundamental (amplitude 1/k) wobbling ±0.1 Hz;
- ``knock_1.05`` [4]: 20-ms pulses every 1.05 s, half along and half perpendicular to gravity;
- ``jerk_lines`` [10]: 1.08 and 2.15 Hz sinusoids perpendicular to gravity;
- ``hull_0.4`` [1]: a narrow band at 0.3–0.5 Hz (40 random-phase sinusoids) along gravity;
- ``line_8.6``, ``line_24`` [6, 7]: lines above the model's 6-Hz band, along gravity.

Per condition and night the release's prediction (default GGIR filter, ``origin`` on the PSG's
epoch grid) is written to ``<out>/predictions/<condition>/<night>.csv``; per night, against PSG:
sleep sensitivity, wake specificity, κ (4-stage and sleep/wake), deep %, mean P(wake) on PSG
wake and on PSG sleep. Also the injected line's height in the model's own input frames: per
frame, the largest bin within ±0.3 Hz of the component's main frequency above the median of
the 7 bins centred on it (dB), as the 10th/50th/90th percentile over frames. For a line above
6 Hz the height is read where it would alias on a 12-Hz grid (8.6 → 3.4 Hz, 24 → 0 Hz, which is
outside the band, so the 0.1–0.3 Hz bins are read).

Writes ``<out>/nights.csv``. A condition and night already scored are skipped.

Usage: python inject.py <data root> <adult bundle> <out> [condition ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from arcasleep.predict import load_predictor

COLUMNS = ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"]
PSG_TO_STAGE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
STAGES = ["wake", "light", "deep", "rem"]
GAINS_MG = [0.3, 1.0, 3.0, 10.0]
COHORT = "SleepAccel-Clinical"


def fm_phase(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    return 2 * np.pi * np.cumsum(f * dt)


def slow_wander(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A smooth path in [−1, 1] varying on 1–10 min scales."""
    periods = rng.uniform(60, 600, 8)
    w = sum(np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi)) for p in periods)
    return w / np.abs(w).max()


def rms(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2))


def component(name: str, t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """``(along gravity, perpendicular)`` unit-RMS signals and the main frequency (Hz)."""
    zero = np.zeros_like(t)
    if name.startswith("line_"):
        f0 = float(name.split("_")[1])
        return rms(np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))), zero, f0
    if name == "wander_3.85":
        return rms(np.sin(fm_phase(t, 3.85 + 0.25 * slow_wander(t, rng)))), zero, 3.85
    if name == "wobble_1.8":
        return rms(np.sin(fm_phase(t, 1.8 + 0.15 * np.sin(2 * np.pi * t / 20.0)))), zero, 1.8
    if name == "comb_wobble":
        phase = fm_phase(t, 1.2 + 0.1 * np.sin(2 * np.pi * t / 20.0))
        return rms(sum(np.sin(k * phase) / k for k in range(1, 11))), zero, 1.2
    if name == "knock_1.05":
        phase = (t % 1.05) - 0.525
        pulses = rms(np.exp(-0.5 * (phase / 0.02) ** 2))
        return pulses / np.sqrt(2), pulses / np.sqrt(2), 1 / 1.05
    if name == "jerk_lines":
        x = np.sin(2 * np.pi * 1.08 * t + rng.uniform(0, 6.3)) + np.sin(2 * np.pi * 2.15 * t + rng.uniform(0, 6.3))
        return zero, rms(x), 1.08
    if name == "hull_0.4":
        freqs = rng.uniform(0.3, 0.5, 40)
        x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in freqs)
        return rms(x), zero, 0.4
    raise ValueError(name)


COMPONENTS = ["line_0.88", "line_1.4", "wander_3.85", "wobble_1.8", "comb_wobble", "knock_1.05", "jerk_lines",
              "hull_0.4", "line_8.6", "line_24"]


def conditions() -> list[str]:
    return ["clean"] + [f"{c}@{g:g}mg" for c in COMPONENTS for g in GAINS_MG]


def inject(accel: np.ndarray, condition: str, seed: int) -> tuple[np.ndarray, float | None]:
    if condition == "clean":
        return accel, None
    name, gain = condition.split("@")
    gain_g = float(gain.removesuffix("mg")) / 1000.0
    t, a = accel[:, 0], accel[:, 1:]
    along, perp, f0 = component(name, t, np.random.default_rng(seed))
    up = a / np.linalg.norm(a, axis=1, keepdims=True)
    side = np.cross(up, [1.0, 0.0, 0.0])
    near = np.linalg.norm(side, axis=1) < 0.3
    side[near] = np.cross(up[near], [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    out = accel.copy()
    out[:, 1:] = a + gain_g * (along[:, None] * up + perp[:, None] * side)
    return out, f0


def line_height(frames: np.ndarray, f0: float | None, freqs: np.ndarray) -> dict:
    """10th/50th/90th percentile over frames of the injected line's height (dB) in each channel."""
    out = {}
    for c, name in enumerate(("mag", "jerk")):
        db = 10 * np.log10(np.exp(frames[:, :, c].astype(np.float64)))
        prominence = db - median_filter(db, size=(1, 7), mode="nearest")
        f = 1.0 if f0 is None else abs(f0 - 12.0 * round(f0 / 12.0))
        near = np.abs(freqs - f) <= 0.3
        h = prominence[:, near].max(axis=1)
        h = h[np.isfinite(h)]
        for q in (10, 50, 90):
            out[f"height_{name}_p{q}"] = float(np.percentile(h, q)) if h.size else np.nan
    return out


def kappa(y: np.ndarray, p: np.ndarray) -> float:
    cats = np.union1d(y, p)
    cm = np.zeros((cats.size, cats.size))
    np.add.at(cm, (np.searchsorted(cats, y), np.searchsorted(cats, p)), 1)
    n = cm.sum()
    po, pe = np.trace(cm) / n, (cm.sum(0) * cm.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def score(psg: pd.DataFrame, pred: pd.DataFrame) -> dict:
    p = pred.set_index(np.rint(pred.start_s.to_numpy()).astype(int))
    t = np.rint(psg.TIMESTAMP.to_numpy()).astype(int)
    y = np.array([PSG_TO_STAGE.get(int(v), -1) for v in psg.PSG.to_numpy()])
    keep = (y >= 0) & np.isin(t, p.index)
    y, t = y[keep], t[keep]
    names = {s: i for i, s in enumerate(STAGES)} | {"gap": 4}
    call = p.loc[t, "stage"].map(names).to_numpy()
    p_wake = p.loc[t, "wake"].to_numpy()
    sleep = y > 0
    sw = np.where(call == 4, 2, (call > 0).astype(int))
    return {
        "epochs": int(y.size),
        "sleep_sensitivity": 100 * np.mean(sw[sleep] == 1),
        "wake_specificity": 100 * np.mean(sw[~sleep] == 0) if (~sleep).any() else np.nan,
        "kappa_sleep_wake": kappa(sleep.astype(int), sw),
        "kappa_4stage": kappa(y, call),
        "deep_pct": 100 * np.mean(call == 2),
        "p_wake_on_wake": float(np.mean(p_wake[~sleep])) if (~sleep).any() else np.nan,
        "p_wake_on_sleep": float(np.mean(p_wake[sleep])),
    }


def main(data: Path, bundle: Path, out: Path, only: list[str]) -> None:
    predictor = load_predictor(bundle)
    features = predictor.ensemble
    freqs = np.round(np.arange(1, 61) * 0.1, 3)
    table_path = out / "nights.csv"
    done = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame(columns=["condition", "night"])
    seen = set(zip(done.condition, done.night))
    rows = done.to_dict("records")
    nights = sorted(p.stem for p in (data / COHORT / "cleaned_psg").glob("*.csv")
                    if (data / COHORT / "cleaned_accelerometer" / p.name).exists())
    for k, night in enumerate(nights):
        accel = pd.read_csv(data / COHORT / "cleaned_accelerometer" / f"{night}.csv", usecols=COLUMNS,
                            dtype="float64")[COLUMNS].to_numpy()
        psg = pd.read_csv(data / COHORT / "cleaned_psg" / f"{night}.csv")
        origin = float(psg.TIMESTAMP.iloc[0] % 30.0)
        for condition in only or conditions():
            if (condition, night) in seen:
                continue
            start = time.time()
            noisy, f0 = inject(accel, condition, seed=k)
            pred = predictor.predict(noisy, origin, subject_id=night)
            target = out / "predictions" / condition / f"{night}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(target)
            frames, _ = features.features(noisy, origin)
            rows.append({"condition": condition, "night": night}
                        | score(psg, pd.read_csv(target)) | line_height(frames, f0, freqs))
            print(f"{night} {condition}: {time.time() - start:.1f} s", flush=True)
        pd.DataFrame(rows).to_csv(table_path, index=False, float_format="%.4f")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])

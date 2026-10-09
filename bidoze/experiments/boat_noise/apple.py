"""SURF004's Apple Watch and Galaxy Watch on the same nights, scored by the adult release.

The Apple Watch files hold CoreMotion's ``userAcceleration`` (gravity estimate subtracted), which
the release cannot read as is: ``|a|`` would be the size of the user acceleration, not
``1 + `` its component along gravity. A constant 1 g is added along one watch axis
(``apple+g<axis>``) so ``|a| ≈ 1 + u_axis``. The true gravity direction is unknown and changes
when the wrist turns, so this is an approximation; the three axes are scored separately to show
how much it matters. The jerk is barely affected by the choice.

Conditions per night, against the same PSG:

- ``galaxy``: the Galaxy Watch recording;
- ``apple+gx``, ``apple+gy``, ``apple+gz``: the Apple Watch with gravity along x, y or z;
- ``apple+gz+white<mg>``: ``apple+gz`` with independent white noise of that RMS on each axis,
  to put the Galaxy Watch's floor on the Apple Watch (log: −62.9 dB trace floor, about 2 mg
  per axis at 50 Hz).

Also per condition, the 10th percentile over PSG-sleep epochs of the quietest frame's 2–6 Hz
trace level (``devices.py``), to check the floors.

Writes ``<out>/apple.csv``.

Usage: python apple.py <data root> <adult bundle> <out>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.predict import load_predictor
from devices import trace_db
from inject import score
from survey import load_accel

WHITE_MG = [1.0, 2.0, 3.0]


def floor_db(a: np.ndarray, psg: pd.DataFrame) -> float:
    times, f, db = trace_db(a[:, 0], a[:, 1:].T)
    level = np.median(db[:, (f >= 2) & (f <= 6)], axis=1)
    quietest = []
    for e, stage in zip(psg.TIMESTAMP, psg.PSG):
        inside = (times >= e) & (times < e + 30)
        if stage in (1, 2, 3, 4, 5) and inside.sum() >= 14 and np.isfinite(level[inside]).all():
            quietest.append(level[inside].min())
    return float(np.percentile(quietest, 10))


def main(root: Path, bundle: Path, out: Path) -> None:
    predictor = load_predictor(bundle)
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (root / "SURF" / "cleaned_accelerometer").glob("SURF004_*_AppleWatch.csv"))
    rows = []
    for k, apple_stem in enumerate(stems):
        galaxy_stem = apple_stem.replace("AppleWatch", "GalaxyWatch")
        psg = pd.read_csv(root / "SURF" / "cleaned_psg" / f"{galaxy_stem}.csv")
        origin = float(psg.TIMESTAMP.iloc[0] % 30.0)
        apple = load_accel(root, "SURF", apple_stem)
        conditions = {"galaxy": load_accel(root, "SURF", galaxy_stem)}
        for axis, name in enumerate("xyz", start=1):
            a = apple.copy()
            a[:, axis] += 1.0
            conditions[f"apple+g{name}"] = a
        rng = np.random.default_rng(k)
        for mg in WHITE_MG:
            a = conditions["apple+gz"].copy()
            a[:, 1:] += mg / 1000.0 * rng.standard_normal(a[:, 1:].shape)
            conditions[f"apple+gz+white{mg:g}mg"] = a
        for name, a in conditions.items():
            pred = predictor.predict(a, origin, subject_id=apple_stem)
            target = out / "predictions" / name / f"{apple_stem}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(target)
            rows.append({"night": apple_stem, "condition": name} | score(psg, pd.read_csv(target))
                        | {"floor_db": floor_db(a, psg)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "apple.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

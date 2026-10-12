"""Descriptive label-shift sensitivity on saved frozen-ensemble outputs.

Convention: shift k pairs the PSG label of epoch i with the prediction at epoch
i + k (shifted_truth[j] = truth[j - k]). Positive k therefore means the labels
are moved later in time relative to the accelerometer. Prediction-side arrays
(still, common_missing, gap, GGIR) stay on the prediction grid; epochs whose
shifted label falls outside the recording are unscored (-1).
Decoding and scoring follow boat_observation/rescore.py, 'native' variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.ggir import evaluate_epoch_grid
from arcasleep_bidoze.adult import hybrid_calls
from boat_evidence.score import DIVISOR, metrics

SHIFTS = (-2, -1, 0, 1, 2)
METRICS = ["sleep_sensitivity", "wake_specificity", "still_wake_specificity", "kappa_sleep_wake",
           "kappa_4stage", "auroc", "descriptive_spec95", "epochs"]


def shift_labels(truth: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(truth, -1)
    if k >= 0:
        out[k:] = truth[: len(truth) - k]
    else:
        out[:k] = truth[-k:]
    return out


def main(a: argparse.Namespace) -> None:
    meta = pd.read_csv(a.data / "inventory.csv").set_index("name").to_dict("index")
    rows = []
    for folder in sorted((a.obs / "nights").iterdir()):
        cohort = folder.name.split("_", 1)[0]
        if cohort not in ("Somerset", "SURF") or not (folder / "complete.json").exists():
            continue
        r = meta[folder.name]
        with np.load(a.data / "nights" / f"{folder.name}.npz") as src:
            ggir = evaluate_epoch_grid(src["anglez"][: int(src["n"])])
        with np.load(folder / "probabilities.npz") as d:
            p, truth, still, common = d["native"], d["truth"], d["still"], d["common_missing"]
        with np.load(folder / "native_axis_psd.npz") as v:
            gap = v["gap"]
        raw = hybrid_calls(p[:, :4], p[:, :4], DIVISOR)
        raw[gap] = 4
        filtered = raw.copy()
        filtered[(raw > 0) & (raw < 4) & ~ggir.is_sib & ~ggir.is_spt] = 0
        for k in SHIFTS:
            y = shift_labels(truth, k)
            for mode, call in (("raw", raw), ("ggir", filtered)):
                rows.append(dict(name=folder.name, cohort=cohort, participant=str(r["participant"]),
                                 group=r["group"], shift=k, mode=mode)
                            | metrics(y, p, call, still, common, True))
    nights = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    nights.to_csv(a.out / "shift_nights.csv", index=False)
    keys = ["cohort", "participant", "shift", "mode"]
    subjects = nights.groupby(keys)[METRICS].mean().reset_index()
    subjects.to_csv(a.out / "shift_participants.csv", index=False)
    summary = subjects.groupby(["cohort", "mode", "shift"])[METRICS].mean()
    summary.insert(0, "participants", subjects.groupby(["cohort", "mode", "shift"]).size())
    summary.insert(1, "nights", nights.groupby(["cohort", "mode", "shift"]).size())
    summary.to_csv(a.out / "shift_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(4))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("obs", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    main(p.parse_args())

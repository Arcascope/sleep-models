"""Score the free-living arms and the release on Foundry, SleepAccel-Clinical and SENSE.

Every model is scored the same way: the mean of its three seeds' five output probabilities
(wake, light, deep, REM, gap) per epoch, then the argmax, with no stage divisor and no GGIR
filter. A scored epoch called gap counts as not sleep and as a miss in 4-stage κ. Prediction
CSVs hold the four stage columns rescaled to sum to 1 and ``gap_head`` as the model gave it;
the stage columns are multiplied by ``1 - gap_head`` to recover the model's five
probabilities before averaging.

- Foundry windows are scored by the phase that held their person out (``random/final`` for
  ``random``); SleepAccel-Clinical and SENSE by each arm's ``final`` phase.
- Sleep recall and precision are over every scored epoch of the recording (24 h for Foundry
  and SENSE).
- Foundry only: ``outside_sleep_min`` is the minutes called sleep outside Apple's first-to-last
  sleep epoch; ``outside_lead2h_sleep_min`` also leaves out the 2 h before Apple's first sleep
  epoch. Foundry recordings have no naps, so every such minute is a false sleep call unless
  Apple missed early sleep.

Layout read under the run directory:
- ``<arm>/<phase>/predictions/<phase>_seed_<k>/<Dataset>_<subject>.csv`` (training outputs)
- ``release/seed<k>/<Dataset>_<subject>.csv``, the release's stage-source seeds scored by
  ``arcasleep bidoze evaluate``

Writes ``analysis/per_recording.csv``, ``analysis/summary.csv`` (mean and SD over
recordings) and ``analysis/paired.csv`` (each arm against the release; paired Wilcoxon over
recordings).

Usage: python score.py <run dir>
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion

ARMS = ["random", "foundry", "foundry-start"]
STAGES = ["wake", "light", "deep", "rem"]
EPOCH_MIN = 0.5
LEAD_EPOCHS = 240
METRICS = ["kappa_4", "kappa_sw", "sleep_recall", "sleep_precision", "wake_specificity",
           "tst_error_min", "sleep_called_gap", "outside_sleep_min", "outside_lead2h_sleep_min"]


def recording_of(path: Path) -> tuple[str, str]:
    """``(cohort, recording)`` from a prediction file name."""
    dataset, subject = path.stem.split("_", 1)
    if dataset.startswith("Foundry"):
        # FoundryApple_<window> from evaluate; Foundry<Person>_<Person>_<window> from training.
        return "Foundry", subject if dataset == "FoundryApple" else subject.split("_", 1)[1]
    return dataset, subject


def release_files(run: Path) -> dict[tuple[str, str], list[Path]]:
    files = defaultdict(list)
    for path in sorted(run.glob("release/seed*/*.csv")):
        files[recording_of(path)].append(path)
    return files


def arm_files(run: Path, arm: str) -> dict[tuple[str, str], list[Path]]:
    files = defaultdict(list)
    for phase in sorted(p for p in (run / arm).iterdir() if p.is_dir()):
        for path in sorted(phase.glob(f"predictions/{phase.name}_seed_*/*.csv")):
            key = recording_of(path)
            if key[0] == "Foundry" or phase.name == "final":
                files[key].append(path)
    return files


def kappa(truth: np.ndarray, call: np.ndarray, classes: int) -> float:
    cm = np.zeros((classes, classes + 1), dtype=np.int64)
    np.add.at(cm, (truth, np.minimum(call, classes)), 1)
    return cohen_kappa_from_confusion(cm)


def softmax_of(frame: pd.DataFrame) -> np.ndarray:
    """The model's five probabilities from a prediction CSV's rescaled stage columns."""
    gap = frame["gap_head"].to_numpy()[:, None]
    return np.hstack([frame[STAGES].to_numpy() * (1 - gap), gap])


def score(paths: list[Path]) -> dict:
    frames = [pd.read_csv(p).set_index("epoch") for p in paths]
    proba = np.mean([softmax_of(f) for f in frames], axis=0)
    psg = frames[0]["psg"]
    call = proba.argmax(axis=1)
    truth = psg.map({s: i for i, s in enumerate(STAGES)}).to_numpy()
    scored = ~np.isnan(truth)
    t, c = truth[scored].astype(int), call[scored]
    t_sleep, c_sleep = t > 0, (c > 0) & (c < 4)
    tp = int((t_sleep & c_sleep).sum())
    row = {
        "seeds": len(paths),
        "scored_epochs": int(scored.sum()),
        "kappa_4": kappa(t, c, 4),
        "kappa_sw": kappa(t_sleep.astype(int), c_sleep.astype(int), 2),
        "sleep_recall": tp / t_sleep.sum() if t_sleep.any() else np.nan,
        "sleep_precision": tp / c_sleep.sum() if c_sleep.any() else np.nan,
        "wake_specificity": (~c_sleep[~t_sleep]).mean() if (~t_sleep).any() else np.nan,
        "tst_error_min": (int(c_sleep.sum()) - int(t_sleep.sum())) * EPOCH_MIN,
        "sleep_called_gap": (c[t_sleep] == 4).mean() if t_sleep.any() else np.nan,
    }
    epochs = psg.index.to_numpy()[scored]
    sleep_epochs = epochs[t_sleep]
    if sleep_epochs.size:
        first, last = sleep_epochs.min(), sleep_epochs.max()
        outside = (epochs < first) | (epochs > last)
        outside_lead = (epochs < first - LEAD_EPOCHS) | (epochs > last)
        row["outside_sleep_min"] = float((c_sleep & outside).sum() * EPOCH_MIN)
        row["outside_lead2h_sleep_min"] = float((c_sleep & outside_lead).sum() * EPOCH_MIN)
    return row


def main(run: Path) -> None:
    models = {"release": release_files(run)}
    models.update({arm: arm_files(run, arm) for arm in ARMS if (run / arm).is_dir()})
    rows = []
    for model, files in models.items():
        for (cohort, recording), paths in sorted(files.items()):
            row = {"model": model, "cohort": cohort, "recording": recording, **score(paths)}
            if cohort != "Foundry":
                row["outside_sleep_min"] = row["outside_lead2h_sleep_min"] = np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_recording.csv", index=False, float_format="%.4f")

    summary = df.groupby(["cohort", "model"])[METRICS].agg(["mean", "std"])
    summary.insert(0, ("recordings", "n"), df.groupby(["cohort", "model"]).size())
    summary.to_csv(out / "summary.csv", float_format="%.4f")

    paired = []
    base = df[df.model == "release"].set_index(["cohort", "recording"])
    for model in models:
        if model == "release":
            continue
        arm = df[df.model == model].set_index(["cohort", "recording"])
        for cohort in arm.index.get_level_values(0).unique():
            both = arm.loc[cohort].join(base.loc[cohort], rsuffix="_release", how="inner")
            for metric in METRICS:
                pair = both[[metric, f"{metric}_release"]].dropna()
                if len(pair) < 2:
                    continue
                diff = pair[metric] - pair[f"{metric}_release"]
                p = wilcoxon(diff).pvalue if diff.abs().sum() > 0 else 1.0
                paired.append({"cohort": cohort, "model": model, "metric": metric, "n": len(pair),
                               "release": pair[f"{metric}_release"].mean(), "arm": pair[metric].mean(),
                               "difference": diff.mean(), "p": p})
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.groupby(["cohort", "model"])[METRICS].mean().round(3))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

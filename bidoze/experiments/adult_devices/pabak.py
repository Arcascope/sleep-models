"""PABAK, beside Cohen's κ, for arms scored by ``score.py`` with their own argmax (``alone``).

PABAK (prevalence-adjusted, bias-adjusted κ) depends only on the observed agreement ``p_o``:
``(k · p_o − 1) / (k − 1)`` for ``k`` categories, so ``2 · p_o − 1`` for sleep/wake. Unlike Cohen's
κ, it does not fall when one category (here, sleep) fills most of the night.

Scored epochs are those with a PSG stage. The call is the argmax of the 5-way output (the 4 stages
and gap); a predicted gap agrees with no stage, so it counts as a miss. Subjects are combined as in
``score.py``: SleepAccel by the fold model that held them out, SleepAccel-Clinical by the mean of
the 5 fold models; ``ensemble`` is the mean output of the 3 seeds. Cohen's κ over the same
epochs, with a predicted gap as its own category, is reported beside PABAK as a check against
``score.py``.

Writes ``<run>/analysis/pabak_subjects.csv``, ``pabak_runs.csv`` (means per arm, model and
cohort) and ``pabak_paired.csv`` (each arm against ``matched``, paired Wilcoxon over subjects).

Usage: python pabak.py <run dir> <arm> [<arm> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from arcasleep.data.splits import read_plan
from score import BIDOZE, COHORTS, FOLDS, SEEDS, combine_folds, load_predictions, mean_over_seeds, seed_predictions

GAP = 4
METRICS = ["pabak_sleep_wake", "pabak_4stage", "kappa_sleep_wake", "kappa_4stage"]


def kappa(y: np.ndarray, p: np.ndarray) -> float:
    cats = np.union1d(y, p)
    cm = np.zeros((cats.size, cats.size))
    np.add.at(cm, (np.searchsorted(cats, y), np.searchsorted(cats, p)), 1)
    n = cm.sum()
    po, pe = np.trace(cm) / n, (cm.sum(0) * cm.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def subject_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    scored = (y >= 0) & (y < GAP)
    y, call = y[scored], proba[scored].argmax(axis=1)
    y_sw = (y > 0).astype(int)
    call_sw = np.where(call == GAP, 2, (call > 0).astype(int))
    return {"pabak_sleep_wake": 2 * np.mean(call_sw == y_sw) - 1,
            "pabak_4stage": (4 * np.mean(call == y) - 1) / 3,
            "kappa_sleep_wake": kappa(y_sw, call_sw),
            "kappa_4stage": kappa(y, call)}


def main(run: Path, arms: list[str]) -> None:
    plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")
    rows = []
    for arm in arms:
        per_seed = [combine_folds([load_predictions(seed_predictions(run / arm, k, s), plan.fold_names[k])
                                   for k in FOLDS]) for s in SEEDS]
        models = {f"seed{s}": p for s, p in zip(SEEDS, per_seed)} | {"ensemble": mean_over_seeds(per_seed)}
        for model, combined in models.items():
            for cohort in COHORTS:
                for sid, (y, proba) in combined[cohort].items():
                    rows.append({"arm": arm, "model": model, "cohort": cohort, "subject_id": sid}
                                | subject_metrics(y, proba))
        print(f"{arm}: scored", flush=True)
    subjects = pd.DataFrame(rows)
    out = run / "analysis"
    subjects.to_csv(out / "pabak_subjects.csv", index=False)
    runs = subjects.groupby(["arm", "model", "cohort"], sort=False)[METRICS].mean()
    runs.to_csv(out / "pabak_runs.csv")
    paired = []
    for cohort in COHORTS:
        for kind in ["seed mean", "ensemble"]:
            sel = subjects[(subjects.cohort == cohort)
                           & ((subjects.model == "ensemble") if kind == "ensemble" else subjects.model.str.startswith("seed"))]
            by = sel.groupby(["subject_id", "arm"])[METRICS].mean().unstack("arm")
            for arm in arms:
                if arm == "matched":
                    continue
                for metric in METRICS:
                    d = (by[(metric, arm)] - by[(metric, "matched")]).dropna()
                    paired.append({"cohort": cohort, "kind": kind, "a": arm, "b": "matched", "metric": metric,
                                   "n": len(d), "mean_a": by[(metric, arm)].mean(),
                                   "mean_b": by[(metric, "matched")].mean(), "diff": d.mean(),
                                   "p": wilcoxon(d).pvalue if d.any() else 1.0})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "pabak_paired.csv", index=False)
    print(runs.round(4).to_string())
    print(paired.round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2:])

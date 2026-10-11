"""Score ensembles assembled from individual seeds of different runs, with the released decoding.

An ensemble is a list of members ``(run subdirectory, seed)``; ``matched/split1`` is the
``matched`` arm early-stopped on holdout draw 1. Each ensemble is scored like the release
(``card_metrics.py``): the members' mean output decides wake or sleep, and light/deep/REM are
divided by the divisor of the training nights.

- SleepAccel (cross-validation): each subject is scored by the members' fold models that held it
  out, with that fold's divisor.
- SleepAccel-Clinical: the members' ``final`` models, with the divisor of all training nights.

Writes ``<out>/subjects.csv`` and ``<out>/summary.csv`` (mean per cohort, ensemble and metric;
paired difference, its 95% CI and Wilcoxon p against ``matched``), and prints the summary.

Usage: python members.py <run dir> <adult cache root> <out dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import arcasleep.evaluation  # noqa: F401  (registers the metrics)
from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import read_plan
from arcasleep.training.config import load_metrics_config
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls
from card_metrics import paired, subject_metrics
from pisces_lite.metrics.logger import MetricsLogger
from score import BETA, BIDOZE, CACHE_NAME, FOLDS, load_predictions, load_subject, night_counts, wake_specificity_at

ARM = "matched"
BASELINE = "matched"
ENSEMBLES = {
    "matched": [("matched", 0), ("matched", 1), ("matched", 2)],
    "matched, holdout draw 1": [("matched/split1", 0), ("matched/split1", 1), ("matched/split1", 2)],
    "one holdout draw per seed": [("matched", 0), ("matched/split1", 1), ("matched/split2", 2)],
    "seed 0, draw 0": [("matched", 0)],
    "seed 1, draw 1": [("matched/split1", 1)],
    "seed 2, draw 2": [("matched/split2", 2)],
    "warm student, lr × 0.1": [("distill-warm", 0)],
    "warm student, full lr": [("distill-warm-full", 0)],
}
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity", "accuracy_4stage",
           "deep%"]


def member_predictions(run: Path, member: str, phase: str, seed: int) -> Path | None:
    """One seed's prediction directory: in the phase directory, or its own seed<n> phase directory."""
    for d in (run / member / phase / "predictions", run / member / phase / f"seed{seed}" / "predictions"):
        if (d / f"{phase}_seed_{seed}").is_dir():
            return d / f"{phase}_seed_{seed}"
    return None


def main(run: Path, adult_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")
    bidsleep = night_counts([s.path for s in discover_cached_subjects(["bidsleep"], adult_root / CACHE_NAME)["bidsleep"]])
    cache = run / "setup/cache" / ARM / CACHE_NAME

    def divisor(cv_name: str) -> np.ndarray:
        cv = json.loads((run / f"setup/cv/{ARM}/{cv_name}.json").read_text())
        paths = [p for d in cv["train_sets"] for p in sorted(cache.glob(f"{d}_subject_*"))]
        return adult_stage_divisor(night_counts(paths), bidsleep, BETA)

    fold_divisors = {k: divisor(f"cv_fold{k}") for k in FOLDS}
    final_divisor = divisor("cv_final")

    rows, pooled = [], {}
    for name, members in ENSEMBLES.items():
        scored: dict[str, dict[str, tuple]] = {"SleepAccel": {}, "SleepAccel-Clinical": {}}
        missing = [f"{m}/fold{k}/seed{s}" for m, s in members for k in FOLDS
                   if member_predictions(run, m, f"fold{k}", s) is None]
        if not missing:
            for k in FOLDS:
                preds = [load_predictions(member_predictions(run, m, f"fold{k}", s), plan.fold_names[k])["SleepAccel"]
                         for m, s in members]
                for sid in preds[0]:
                    mean = np.mean([p[sid][1] for p in preds], axis=0)
                    scored["SleepAccel"][sid] = (preds[0][sid][0], mean, hybrid_calls(mean, mean, fold_divisors[k]))
        finals = [member_predictions(run, m, "final", s) for m, s in members]
        if all(finals):
            preds = [{f.stem.split("_", 1)[1]: load_subject(f) for f in d.glob("SleepAccel-Clinical_*.csv")}
                     for d in finals]
            for sid in preds[0]:
                mean = np.mean([p[sid][1] for p in preds], axis=0)
                scored["SleepAccel-Clinical"][sid] = (preds[0][sid][0], mean, hybrid_calls(mean, mean, final_divisor))
        for cohort, subjects in scored.items():
            if not subjects:
                print(f"{name}, {cohort}: not scored (missing {', '.join(missing) or 'final models'})")
                continue
            pooled[name, cohort] = {sid: (y, p) for sid, (y, p, _) in subjects.items()}
            for sid, (y, p, calls) in subjects.items():
                rows.append({"ensemble": name, "cohort": cohort, "subject_id": sid,
                             **subject_metrics(logger, y, p, calls)})
    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(out / "subjects.csv", index=False)

    summary = []
    for (name, cohort), g in per_subject.groupby(["ensemble", "cohort"], sort=False):
        base = per_subject[(per_subject.ensemble == BASELINE) & (per_subject.cohort == cohort)]
        for metric in METRICS:
            row = {"ensemble": name, "cohort": cohort, "metric": metric, "n": len(g), "mean": g[metric].mean()}
            if name != BASELINE:
                row |= dict(zip(["diff", "diff_low", "diff_high", "p"], paired(g, base, metric)))
            summary.append(row)
        summary.append({"ensemble": name, "cohort": cohort, "metric": "wake_spec_at_95", "n": len(g),
                        "mean": wake_specificity_at(pooled[name, cohort], 0.95)})
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())

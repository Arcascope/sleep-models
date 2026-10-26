"""Metrics for the adult model's card, scored the way the released bundle scores recordings.

The released decoding: the 3-seed mean output decides wake or sleep; on its sleep epochs,
light/deep/REM are divided by the adult stage divisor (``hybrid_calls(mean, mean, divisor)``).

- SleepAccel (cross-validation): each subject is scored by the fold models of ``<arm>`` that
  held it out, with that fold's divisor.
- SleepAccel-Clinical: the released model (``<arm>/final``), with the bundle's divisor.
- The pediatric release is scored alongside, by itself (argmax), and with the adult rule and a
  divisor computed from its own training nights (the arm's Weaver nights) and BIDSleep.

Per night, 4-stage accuracy is the share of PSG-scored epochs with PSG's stage, and balanced
4-stage accuracy the mean recall of the stages present in that night's PSG; a predicted gap
counts as a miss. Stage-time errors are |model − PSG| minutes per night; the error metrics also
get 95% limits of agreement (mean ± 1.96 SD over nights).

Writes ``<out>/subjects.csv`` (per subject) and ``<out>/summary.csv`` (mean and 95% CI of the
mean per cohort, model and metric; for the adult model, the paired difference with its 95% CI and
Wilcoxon p against the pediatric release, and against the release with the divisor), and prints
the summary.

Usage: python card_metrics.py <run dir> <arm> <adult bundle> <adult cache root> <out dir>
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import arcasleep.evaluation  # noqa: F401  (registers the metrics)
from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import read_plan
from arcasleep.training.config import load_metrics_config
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls
from pisces_lite.metrics.logger import MetricsLogger
from score import (BETA, BIDOZE, CACHE_NAME, FOLDS, SEEDS, load_predictions, load_subject, night_counts,
                   seed_predictions, wake_specificity_at)

METRICS = ["kappa_sleep_wake", "kappa_4stage", "sleep_sensitivity", "wake_specificity",
           "tst_error_minutes", "waso_error_minutes", "light_mape", "deep_mape", "rem_mape"]
STAGES = ["wake", "light", "deep", "rem"]
EPOCH_MINUTES = 0.5
ERRORS = ["tst_error_minutes", "waso_error_minutes", "light_error_minutes", "deep_error_minutes",
          "rem_error_minutes"]
RELEASE = "pediatric release"
RELEASE_DIVIDED = "pediatric release + divisor"
RELEASE_PREDICTIONS = "bidoze-bidsleep-seeds-0/reference/pediatric-adults/predictions"


def subject_metrics(logger: MetricsLogger, y: np.ndarray, proba: np.ndarray, calls: np.ndarray | None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        recorded = logger.record(Path(tmp) / "m.csv", y_true=y, y_pred_proba=proba, y_pred=calls, run_id="",
                                 fold="", epoch=0, subject_id="", dataset="", training_dataset="")
    r = {m["metric"]: m["value"] for m in recorded}
    pred = np.asarray(proba).argmax(axis=1) if calls is None else np.asarray(calls)
    scored = (y >= 0) & (y < len(STAGES))
    recalls = [np.mean(pred[scored & (y == c)] == c) for c in range(len(STAGES)) if np.any(scored & (y == c))]
    total = sum(r[f"pred_{s}_count"] for s in STAGES)
    psg_total = sum(r[f"true_{s}_count"] for s in STAGES[1:])
    return {**{m: r[m] for m in METRICS},
            "balanced_accuracy_4stage": float(np.mean(recalls)),
            "accuracy_4stage": float(np.mean(pred[scored] == y[scored])),
            "deep%": 100 * r["pred_deep_count"] / total,
            "psg_deep%": 100 * r["true_deep_count"] / max(psg_total + r["true_wake_count"], 1),
            **{f"{s}_error_minutes": EPOCH_MINUTES * abs(r[f"pred_{s}_count"] - r[f"true_{s}_count"])
               for s in STAGES[1:]}}


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> tuple[float, float, float, float]:
    """Mean paired difference ``a − b`` over subjects, its 95% t-interval, and the Wilcoxon p."""
    d = (a.set_index("subject_id")[metric] - b.set_index("subject_id").loc[a.subject_id, metric]).to_numpy(float)
    d = d[np.isfinite(d)]
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    return d.mean(), d.mean() - half, d.mean() + half, p


def main(run: Path, arm: str, bundle: Path, adult_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")
    bidsleep = night_counts([s.path for s in discover_cached_subjects(["bidsleep"], adult_root / CACHE_NAME)["bidsleep"]])
    cache = run / "setup/cache" / arm / CACHE_NAME

    # SleepAccel: fold models, 3-seed mean, each fold's own divisor.
    adult: dict[str, dict[str, tuple]] = {"SleepAccel": {}, "SleepAccel-Clinical": {}}
    for k in FOLDS:
        cv = json.loads((run / f"setup/cv/{arm}/cv_fold{k}.json").read_text())
        paths = [p for d in cv["train_sets"] for p in sorted(cache.glob(f"{d}_subject_*"))]
        divisor = adult_stage_divisor(night_counts(paths), bidsleep, BETA)
        seeds = [load_predictions(seed_predictions(run / arm, k, s), plan.fold_names[k])["SleepAccel"] for s in SEEDS]
        for sid in seeds[0]:
            mean = np.mean([s[sid][1] for s in seeds], axis=0)
            adult["SleepAccel"][sid] = (seeds[0][sid][0], mean, hybrid_calls(mean, mean, divisor))

    # SleepAccel-Clinical: the released model and the bundle's divisor.
    spec = json.loads((bundle / "config.json").read_text())
    divisor = np.array([1.0] + [spec["stage_divisor"][n] for n in ("light", "deep", "rem")])
    final = run / arm / "final/predictions"
    seeds = [{f.stem.split("_", 1)[1]: load_subject(f) for f in (final / f"final_seed_{s}").glob("SleepAccel-Clinical_*.csv")}
             for s in SEEDS]
    for sid in seeds[0]:
        mean = np.mean([s[sid][1] for s in seeds], axis=0)
        adult["SleepAccel-Clinical"][sid] = (seeds[0][sid][0], mean, hybrid_calls(mean, mean, divisor))

    # The pediatric release: argmax, and the adult rule with a divisor from its own training nights.
    weaver = [p for d in json.loads((run / f"setup/cv/{arm}/cv_fold0.json").read_text())["train_sets"]
              if not d.startswith("BiDozeSleepAccel") for p in sorted(cache.glob(f"{d}_subject_*"))]
    release_divisor = adult_stage_divisor(night_counts(weaver), bidsleep, BETA)
    release: dict[str, dict[str, tuple]] = {c: {} for c in adult}
    release_divided: dict[str, dict[str, tuple]] = {c: {} for c in adult}
    for f in sorted((run.parent / RELEASE_PREDICTIONS).glob("*.csv")):
        for c in adult:
            if f.stem.startswith(c + "_") and f.stem[len(c) + 1:] in adult[c]:
                y, p = load_subject(f)
                release[c][f.stem[len(c) + 1:]] = (y, p, None)
                release_divided[c][f.stem[len(c) + 1:]] = (y, p, hybrid_calls(p, p, release_divisor))

    models = {"adult": adult, RELEASE_DIVIDED: release_divided, RELEASE: release}
    rows = []
    for model, data in models.items():
        for cohort, subjects in data.items():
            for sid, (y, p, calls) in subjects.items():
                rows.append({"model": model, "cohort": cohort, "subject_id": sid,
                             **subject_metrics(logger, y, p, calls)})
    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(out / "subjects.csv", index=False)

    summary = []
    for (model, cohort), g in per_subject.groupby(["model", "cohort"]):
        pooled = {sid: (y, p) for sid, (y, p, _) in models[model][cohort].items()}
        for metric in [*METRICS, *ERRORS[2:], "balanced_accuracy_4stage", "accuracy_4stage", "deep%", "psg_deep%"]:
            x = g[metric].to_numpy(float)
            x = x[np.isfinite(x)]
            sd = x.std(ddof=1)
            half = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
            row = {"model": model, "cohort": cohort, "metric": metric, "n": len(x), "mean": x.mean(),
                   "ci_low": x.mean() - half, "ci_high": x.mean() + half, "median": np.median(x)}
            if metric in ERRORS:
                row |= {"loa_low": x.mean() - 1.96 * sd, "loa_high": x.mean() + 1.96 * sd}
            if model == "adult" and metric != "psg_deep%":
                for name, other in (("release", RELEASE), ("divided", RELEASE_DIVIDED)):
                    base = per_subject[(per_subject.model == other) & (per_subject.cohort == cohort)]
                    row |= dict(zip([f"diff_{name}", f"diff_{name}_low", f"diff_{name}_high", f"p_{name}"],
                                    paired(g, base, metric)))
            summary.append(row)
        summary.append({"model": model, "cohort": cohort, "metric": "wake_spec_at_95", "n": len(g),
                        "mean": wake_specificity_at(pooled, 0.95)})
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve(),
         Path(sys.argv[5]).resolve())

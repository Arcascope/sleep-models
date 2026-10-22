"""Score the pediatric release trained on all revised-Weaver recordings against the one trained on
the earlier Weaver cache.

Models (each a run directory of phases ``fold0`` to ``fold4`` and ``final``):
- ``matched``: the earlier ``weaver`` cache (202 recordings), ``config/splits/weaver_5fold.json``;
- ``all``: every revised-Weaver recording (634), ``config/splits/weaver_revised_pediatric_5fold.json``.

The plan of ``all`` keeps each child of the earlier plan in its earlier fold, so for each fold k
both models' fold k held out the same children. Both are scored on the revised-Weaver recordings
of the fold-k children (every device and night of them), by ``arcasleep bidoze evaluate``, argmax
decoding. A child whose earlier nights are in two folds (``split_earlier_children`` in the plan)
is left out of the fold scoring: ``matched`` trained on one of its nights.

Per recording: 4-stage κ, sleep/wake κ, sleep sensitivity, wake specificity (the recall of wake),
recall of light, deep and REM, and total sleep time (TST) bias in minutes (predicted − PSG). A fold's
value is the mean over its recordings. The comparison is ``all`` − ``matched`` per fold, then the
mean and range of that difference over the 5 folds; no threshold is applied.

The ``final`` phases are compared on SleepAccel (28 subjects) and SleepAccel-Clinical (32), from
the predictions of each ``final`` phase; p is a paired Wilcoxon over subjects.

Usage:
  python score.py predict <run dir> <cache root with weaver_revised>
  python score.py score <run dir>

``<run dir>`` holds ``matched/`` and ``all/``, each the training directory of a run copied from
its results (``aws s3 cp --recursive``); ``predict`` needs each fold's ``config/`` and
``model_outputs/`` and writes ``<run dir>/<model>/revised/fold<k>/predictions/``; ``score`` also
needs ``final/predictions/`` and writes ``<run dir>/analysis/``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import arcasleep.evaluation  # noqa: F401  (registers the metrics)
from arcasleep.data.splits import read_plan
from arcasleep.training.config import load_metrics_config
from pisces_lite.metrics.logger import MetricsLogger

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
PLAN = BIDOZE / "config/splits/weaver_revised_pediatric_5fold.json"
MODELS = ("matched", "all")
FOLDS = range(5)
STAGES = ["wake", "light", "deep", "rem"]
ADULT_COHORTS = {"SleepAccel": "SleepAccel_", "SleepAccel-Clinical": "SleepAccel-Clinical_"}
METRICS = {
    "kappa_4stage": "4-stage κ",
    "kappa_sleep_wake": "sleep/wake κ",
    "sleep_sensitivity": "sleep sensitivity",
    "wake_specificity": "wake specificity",
    "recall_light": "light recall",
    "recall_deep": "deep recall",
    "recall_rem": "REM recall",
    "tst_error_minutes": "TST bias (min)",
}


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(y_true, proba)``: the PSG stage per epoch (−1 unscored) and the 5-way output."""
    d = pd.read_csv(path, dtype={"psg": str})
    stage = d[STAGES].to_numpy(float) * (1.0 - d["gap_head"].to_numpy(float))[:, None]
    proba = np.column_stack([stage, d["gap_head"].to_numpy(float)])
    y = d["psg"].map({s: i for i, s in enumerate(STAGES)}).fillna(-1).astype(int).to_numpy()
    return y, proba


def stage_recalls(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Fraction of each stage's PSG epochs that the argmax call gets right; NaN without any."""
    called = proba.argmax(axis=1)
    return {f"recall_{name}": float(np.mean(called[y == i] == i)) if np.any(y == i) else np.nan
            for i, name in enumerate(STAGES)}


def recording_metrics(logger: MetricsLogger, y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        recorded = logger.record(Path(tmp) / "m.csv", y_true=y, y_pred_proba=proba, y_pred=None,
                                 run_id="", fold="", epoch=0, subject_id="", dataset="",
                                 training_dataset="")
    values = {m["metric"]: m["value"] for m in recorded}
    values.update(stage_recalls(y, proba))
    return {k: values[k] for k in METRICS}


def score_directory(logger: MetricsLogger, directory: Path, prefix: str = "") -> pd.DataFrame:
    rows = [{"subject_id": f.stem[len(prefix):], **recording_metrics(logger, *load_subject(f))}
            for f in sorted(directory.glob(f"{prefix}*.csv"))
            if not (prefix == "SleepAccel_" and f.stem.startswith("SleepAccel-Clinical_"))]
    return pd.DataFrame(rows)


def fold_recordings(plan_path: Path = PLAN) -> dict[int, list[str]]:
    """The revised-Weaver recordings of each fold, less the split earlier children."""
    plan = read_plan(plan_path)
    split = set(plan.provenance.get("split_earlier_children", []))
    return {k: [s for s in plan.folds[name] if s.split("_", 1)[0] not in split]
            for k, name in enumerate(plan.fold_names)}


def predict(run: Path, cache_root: Path) -> None:
    """Score each model's fold k on the revised recordings of the fold-k children."""
    cache = cache_root / CACHE_NAME
    for k, recordings in fold_recordings().items():
        inputs = [cache / f"weaver_revised_subject_{sid}_features.npz" for sid in recordings]
        for model in MODELS:
            out = run / model / "revised" / f"fold{k}"
            if (out / "predictions").is_dir():
                continue
            subprocess.run(
                ["arcasleep", "bidoze", "evaluate", "-t", str(run / model / f"fold{k}"),
                 "-i", *map(str, inputs), "-o", str(out), "--overwrite"], check=True)


def paired(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """``all`` − ``matched`` per metric from rows of ``model``, ``by`` and the metrics."""
    wide = frame.pivot(index=by, columns="model", values=list(METRICS))
    rows = []
    for metric, label in METRICS.items():
        diff = (wide[(metric, "all")] - wide[(metric, "matched")]).dropna()
        rows.append({"metric": label, "matched": wide[(metric, "matched")].mean(),
                     "all": wide[(metric, "all")].mean(), "difference": diff.mean(),
                     "min": diff.min(), "max": diff.max(), "n": len(diff)})
    return pd.DataFrame(rows)


def paired_p(difference: pd.Series) -> float:
    """Paired Wilcoxon p of the per-subject differences; NaN when there is none to test."""
    difference = difference.dropna()
    return float(wilcoxon(difference).pvalue) if (difference != 0).any() else np.nan


def main(run: Path) -> None:
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    recordings, folds = [], []
    for model in MODELS:
        for k in FOLDS:
            scored = score_directory(logger, run / model / "revised" / f"fold{k}/predictions",
                                     "weaver_revised_")
            recordings.append(scored.assign(model=model, fold=k))
            folds.append({"model": model, "fold": k, "recordings": len(scored),
                          **scored.drop(columns="subject_id").mean().to_dict()})
    pd.concat(recordings, ignore_index=True).to_csv(out / "recordings.csv", index=False)
    folds = pd.DataFrame(folds)
    folds.to_csv(out / "folds.csv", index=False)
    by_fold = paired(folds, "fold")
    by_fold.to_csv(out / "paired_folds.csv", index=False)

    adult = []
    for cohort, prefix in ADULT_COHORTS.items():
        both = pd.concat([
            score_directory(logger, run / model / "final/predictions/final_seed_0", prefix)
            .assign(model=model) for model in MODELS], ignore_index=True)
        table = paired(both, "subject_id")
        wide = both.pivot(index="subject_id", columns="model", values=list(METRICS))
        table["p"] = [paired_p(wide[(m, "all")] - wide[(m, "matched")]) for m in METRICS]
        adult.append(table.assign(cohort=cohort))
    adult = pd.concat(adult, ignore_index=True)
    adult.to_csv(out / "final_adult.csv", index=False)

    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("Per fold, mean over recordings (revised Weaver, fold-k children):")
    print(folds.rename(columns=METRICS).to_string(index=False))
    print("\nPaired over the 5 folds (difference = all − matched; min and max over folds):")
    print(by_fold.to_string(index=False))
    for cohort in ADULT_COHORTS:
        print(f"\nFinal phases on {cohort} (mean over subjects; p: paired Wilcoxon):")
        print(adult[adult.cohort == cohort].drop(columns="cohort").to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "predict":
        predict(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        main(Path(sys.argv[2]).resolve())
    else:
        raise SystemExit(__doc__.split("Usage:")[1])

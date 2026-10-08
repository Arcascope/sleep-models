"""Score the device ablation: each arm's stage source alone and in the adult formulation.

Models per arm (seeds 0-2):
- seed s: SleepAccel subjects scored by the fold model that held them out (28 subjects);
  SleepAccel-Clinical by the mean output of the 5 fold models.
- ensemble: the mean output of the 3 seeds, per fold model; then as above.

Arms: every arm in ``setup/arms.json`` (paired comparisons against ``matched`` and ``all`` as
well as the reference), plus ``weaver-2026-10-06``, the October 6 §2 mix runs
(`bidoze-weaver-mix-0/mix{k}-seed{s}`, 202 Weaver nights + 4/5 SleepAccel), seeds 0-2.

Scorings:
- alone:  the stage source's argmax (sleep/wake and stages).
- adult:  `arcasleep_bidoze.adult.hybrid_calls`: the pediatric release's wake and gap calls; on
          its sleep epochs, the stage source's light/deep/REM divided by the adult stage divisor.
          The divisor is computed per fold model from its own training nights; adult reference:
          BIDsleep. SleepAccel-Clinical, scored by the mean output of the 5 fold models, uses the
          geometric mean of their divisors.

Stalled fold models (best eval κ < 0.3; healthy ones reach ≥ 0.48) are kept in ``<arm>`` and
left out of ``<arm> (stalled out)``, which is added for each arm that has one: a stalled seed's
held-out SleepAccel subjects are dropped from that seed, and that fold's ensemble and the
SleepAccel-Clinical mean use the other fold models.

Statistics: mean over subjects; p: paired Wilcoxon over subjects. Seed arms are compared on the
per-subject mean over seeds.

Writes to ``<run>/analysis/``: runs.csv, subjects.csv, paired.csv.

Usage: python score.py <run dir> <SleepAccel cache root>
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import arcasleep.evaluation  # noqa: F401  (registers the metrics)
from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import read_plan
from arcasleep.training.config import load_metrics_config
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls, stage_counts
from pisces_lite.metrics.logger import MetricsLogger

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
COHORTS = ["SleepAccel", "SleepAccel-Clinical"]
SEEDS = [0, 1, 2]
STALLED_KAPPA = 0.3
FOLDS = range(5)
REFERENCE = "weaver-2026-10-06"
STAGES = ["wake", "light", "deep", "rem"]
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity"]
BETA = 0.999


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(y_true, proba)``; proba is the 5-way output (4 stages scaled by 1 − gap, then gap)."""
    d = pd.read_csv(path, dtype={"psg": str})
    stage = d[STAGES].to_numpy(float) * (1.0 - d["gap_head"].to_numpy(float))[:, None]
    proba = np.column_stack([stage, d["gap_head"].to_numpy(float)])
    y = d["psg"].map({s: i for i, s in enumerate(STAGES)}).fillna(-1).astype(int).to_numpy()
    return y, proba


def load_predictions(pred_dir: Path, held_out: str) -> dict[str, dict[str, tuple]]:
    out: dict[str, dict[str, tuple]] = {c: {} for c in COHORTS}
    for f in sorted(pred_dir.glob("*.csv")):
        if f.stem.startswith(held_out + "_"):
            out["SleepAccel"][f.stem[len(held_out) + 1:]] = load_subject(f)
        elif f.stem.startswith("SleepAccel-Clinical_"):
            out["SleepAccel-Clinical"][f.stem[len("SleepAccel-Clinical_"):]] = load_subject(f)
    return out


def combine_folds(per_fold: list[dict | None]) -> dict[str, dict[str, tuple]]:
    """SleepAccel: each subject from its held-out fold; SleepAccel-Clinical: mean over folds.

    A fold left out (None) contributes no SleepAccel subjects and no SleepAccel-Clinical output.
    """
    per_fold = [f for f in per_fold if f is not None]
    clinical = per_fold[0]["SleepAccel-Clinical"]
    return {
        "SleepAccel": {sid: v for f in per_fold for sid, v in f["SleepAccel"].items()},
        "SleepAccel-Clinical": {
            sid: (y, np.mean([f["SleepAccel-Clinical"][sid][1] for f in per_fold], axis=0))
            for sid, (y, _) in clinical.items()
        },
    }


def mean_over_seeds(per_seed: list[dict | None]) -> dict:
    per_seed = [s for s in per_seed if s is not None]
    return {c: {sid: (y, np.mean([s[c][sid][1] for s in per_seed], axis=0))
                for sid, (y, _) in per_seed[0][c].items()} for c in COHORTS}


def best_eval_kappas(log: Path) -> list[float]:
    """Best eval κ per seed, in the order the seeds trained."""
    return [float(line.split()[3]) for line in log.read_text().splitlines()
            if line.strip().startswith("best eval kappa")]


def night_counts(paths: list[Path]) -> np.ndarray:
    return np.array([stage_counts(np.load(p)["y"]) for p in paths])


def score(logger: MetricsLogger, subjects: dict, calls: dict | None) -> pd.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for sid, (y, p) in subjects.items():
            recorded = logger.record(Path(tmp) / "m.csv", y_true=y, y_pred_proba=p,
                                     y_pred=None if calls is None else calls[sid], run_id="", fold="",
                                     epoch=0, subject_id=sid, dataset="", training_dataset="")
            r = {m["metric"]: m["value"] for m in recorded}
            total = sum(r[f"pred_{s}_count"] for s in STAGES)
            rows.append({"subject_id": sid, **{m: r[m] for m in METRICS},
                         "deep%": 100 * r["pred_deep_count"] / total})
    return pd.DataFrame(rows)


def auroc(subjects: dict) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([s[0] for s in subjects.values()])
    p = np.concatenate([s[1][:, 1:4].sum(axis=1) for s in subjects.values()])
    keep = (y >= 0) & (y < 4)
    return float(roc_auc_score(y[keep] > 0, p[keep]))


def main(run: Path, adult_root: Path) -> None:
    results = run.parent
    out_dir = run / "analysis"
    out_dir.mkdir(exist_ok=True)
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")
    adult_cache = adult_root / CACHE_NAME
    cached = discover_cached_subjects(["bidsleep", "weaver", "SleepAccel"], adult_cache)
    bidsleep = night_counts([s.path for s in cached["bidsleep"]])
    sleepaccel = {s.subject_id: s.path for s in cached["SleepAccel"]}

    def sleepaccel_paths(k: int) -> list[Path]:
        return [sleepaccel[sid] for i, f in enumerate(plan.fold_names) if i != k for sid in plan.folds[f]]

    ped = {c: {} for c in COHORTS}
    for f in sorted((results / "bidoze-bidsleep-seeds-0/reference/pediatric-adults/predictions").glob("*.csv")):
        for c in COHORTS:
            if f.stem.startswith(c + "_"):
                ped[c][f.stem[len(c) + 1:]] = load_subject(f)

    # arm -> (per seed: per fold predictions, per fold training-night paths)
    arms: dict[str, tuple[dict[int, list[dict]], list[list[Path]]]] = {}
    for arm in json.loads((run / "setup/arms.json").read_text()):
        cache = run / "setup/cache" / arm / CACHE_NAME
        if not all((run / arm / f"fold{k}/predictions/fold{k}_seed_{s}").is_dir() for k in FOLDS for s in SEEDS):
            print(f"{arm}: incomplete, skipped")
            continue
        preds = {s: [load_predictions(run / arm / f"fold{k}/predictions/fold{k}_seed_{s}", plan.fold_names[k])
                     for k in FOLDS] for s in SEEDS}
        train = []
        for k in FOLDS:
            cv = json.loads((run / f"setup/cv/{arm}/cv_fold{k}.json").read_text())
            train.append([p for d in cv["train_sets"] for p in sorted(cache.glob(f"{d}_subject_*"))])
        arms[arm] = (preds, train)
        stalled = [(s, k) for k in FOLDS
                   for s, kappa in zip(SEEDS, best_eval_kappas(run / arm / f"fold{k}/training.log"))
                   if kappa < STALLED_KAPPA]
        if stalled:
            print(f"{arm}: stalled fold models (seed, fold): {stalled}")
            kept = {s: [None if (s, k) in stalled else preds[s][k] for k in FOLDS] for s in SEEDS}
            arms[f"{arm} (stalled out)"] = (kept, train)
    mix = results / "bidoze-weaver-mix-0"
    arms[REFERENCE] = (
        {s: [load_predictions(mix / f"mix{k}-seed{s}/predictions/mix{k}_seed_{s}", plan.fold_names[k])
             for k in FOLDS] for s in SEEDS},
        [[s.path for s in cached["weaver"]] + sleepaccel_paths(k) for k in FOLDS],
    )

    rows, subjects = [], []

    def record(arm: str, model: str, scoring: str, cohort: str, data: dict, calls: dict | None) -> None:
        m = score(logger, data, calls)
        subjects.append(m.assign(arm=arm, model=model, scoring=scoring, cohort=cohort))
        rows.append({"arm": arm, "model": model, "scoring": scoring, "cohort": cohort, "n": len(m),
                     **m.drop(columns="subject_id").mean().to_dict(),
                     "auroc": auroc(data) if scoring == "alone" else np.nan})

    for c in COHORTS:
        record("pediatric release", "release", "alone", c, ped[c], None)

    for arm, (preds, train) in arms.items():
        divisors = [adult_stage_divisor(night_counts(paths), bidsleep, BETA) for paths in train]
        models = {f"seed{s}": preds[s] for s in SEEDS}
        models["ensemble"] = [mean_over_seeds([preds[s][k] for s in SEEDS]) for k in FOLDS]
        for model, per_fold in models.items():
            combined = combine_folds(per_fold)
            # Each subject's adult calls use the divisor of the fold model(s) that scored it.
            adult_calls: dict[str, dict] = {c: {} for c in COHORTS}
            for k, fold in enumerate(per_fold):
                if fold is None:
                    continue
                for sid, (_, p) in fold["SleepAccel"].items():
                    adult_calls["SleepAccel"][sid] = hybrid_calls(ped["SleepAccel"][sid][1], p, divisors[k])
            used = [d for d, fold in zip(divisors, per_fold) if fold is not None]
            mean_divisor = np.exp(np.mean(np.log(used), axis=0))
            for sid, (_, p) in combined["SleepAccel-Clinical"].items():
                adult_calls["SleepAccel-Clinical"][sid] = hybrid_calls(ped["SleepAccel-Clinical"][sid][1], p,
                                                                      mean_divisor)
            for c in COHORTS:
                record(arm, model, "alone", c, combined[c], None)
                wake_source = {sid: ped[c][sid] for sid in combined[c]}
                record(arm, model, "adult", c, wake_source, adult_calls[c])
        print(f"{arm}: scored", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "runs.csv", index=False)
    per_subject = pd.concat(subjects, ignore_index=True)
    per_subject.to_csv(out_dir / "subjects.csv", index=False)

    def arm_scores(arm: str, kind: str, scoring: str, cohort: str) -> pd.DataFrame:
        sel = per_subject[(per_subject.arm == arm) & (per_subject.scoring == scoring) & (per_subject.cohort == cohort)]
        if arm == "pediatric release":
            return sel.set_index("subject_id")[[*METRICS, "deep%"]]
        sel = sel[sel.model == "ensemble"] if kind == "ensemble" else sel[sel.model.str.startswith("seed")]
        return sel.groupby("subject_id")[[*METRICS, "deep%"]].mean()

    paired = []
    names = [a for a in arms if a != REFERENCE]
    for cohort in COHORTS:
        for scoring in ["alone", "adult"]:
            for kind in ["seed mean", "ensemble"]:
                for a in names + [REFERENCE]:
                    for b in [REFERENCE, "matched", "all", "pediatric release"]:
                        if a == b or (b in ("all", "matched") and (b not in arms or a == REFERENCE)):
                            continue
                        x, y = arm_scores(a, kind, scoring, cohort).align(
                            arm_scores(b, kind, "alone" if b == "pediatric release" else scoring, cohort),
                            join="inner")
                        for metric in [*METRICS, "deep%"]:
                            d = x[metric] - y[metric]
                            paired.append({"cohort": cohort, "scoring": scoring, "kind": kind, "a": a, "b": b,
                                           "metric": metric, "n": len(d), "mean_a": x[metric].mean(),
                                           "mean_b": y[metric].mean(), "diff": d.mean(),
                                           "p": wilcoxon(d).pvalue if d.any() else 1.0})
    pd.DataFrame(paired).to_csv(out_dir / "paired.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())

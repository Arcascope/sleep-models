"""Score the gap-rule retrains' cross-validation folds against the v0.1.0 release's folds.

Every model is scored per recording from its held-out fold predictions, with the release's
decoding:

- Adult: the mean output of the 3 seeds of the fold model that held the recording out decides
  wake, light/deep/REM and gap; on its sleep epochs, light/deep/REM are divided by the adult
  stage divisor (``hybrid_calls(mean, mean, divisor)``). The divisor is computed per fold model
  from its own training nights and BIDSleep.
- Pediatric: the argmax of the 5-way output.

Cohorts:

- ``SleepAccel``: the 28 recordings held out by the 5 SleepAccel folds.
- ``SleepAccel-Clinical``: scored by every fold model; the mean output of the 5 fold models is
  decoded with the geometric mean of their divisors.
- ``Weaver``: the 202 recordings held out by the 5 Weaver folds (``weaver_5fold.json``).

Models:

- ``v0.1.0 adult``: ``bidoze-adult-devices-0/matched``, the release's training arm.
- ``adult all (pre-gap)``: ``bidoze-adult-devices-0/all``, the same recipe as ``adult-gap-all``
  without the gap rule.
- ``adult-gap-matched``, ``adult-gap-all``: the gap-rule retrains.
- ``v0.1.0 pediatric``: ``bidoze-pediatric-v0.1.0/training/fold<k>``. The four metrics are those
  of the phases' own ``cv_results_4_stages.csv``. The phases saved no predictions, so the fold
  models were re-run with ``arcasleep bidoze evaluate -ft`` on the feature caches available
  (``cv-scores/v0.1.0-pediatric-predictions``); REM recall comes from the recordings whose re-run
  reproduces the phase's four metrics.
- ``v0.1.0 pediatric (re-run)``: every re-run recording scored from its own predictions.
- ``pediatric-gap``: the gap-rule retrain; the four metrics are its phases' own, REM recall comes
  from its predictions.

Metrics per recording: 4-stage κ, sleep/wake κ, sleep sensitivity, wake specificity (the
metrics of ``config/metrics.json``) and REM recall, the share of PSG REM epochs called REM (a
predicted gap counts as a miss). A recording with no PSG REM has no REM recall and is left out
of that metric's pairs.

Comparison: paired by recording on the recordings both models scored; the difference is
retrain − baseline, its 95% interval a percentile bootstrap over recordings, and p a paired
Wilcoxon test.

Writes ``<results>/gap-retrain-2026-10-21/cv-scores/``: ``per_recording.csv``, ``paired.csv``,
``tables.md``.

Usage: python score.py <bidoze results dir>
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
from arcasleep.data.splits import read_plan
from arcasleep.training.config import load_metrics_config
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls, stage_counts
from pisces_lite.metrics.logger import MetricsLogger

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
STAGES = ["wake", "light", "deep", "rem"]
REM = 3
BETA = 0.999
FOLDS = range(5)
BOOTSTRAPS = 10000
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity", "rem_recall"]
LABELS = {"kappa_4stage": "4-stage κ", "kappa_sleep_wake": "sleep/wake κ",
          "sleep_sensitivity": "sleep sensitivity", "wake_specificity": "wake specificity",
          "rem_recall": "REM recall"}
RUN = "gap-retrain-2026-10-21"
RERUN = "v0.1.0 pediatric (re-run)"
REPRODUCED = 1e-9

Subjects = dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(y_true, proba)``; proba is the 5-way output (4 stages scaled by 1 − gap, then gap)."""
    d = pd.read_csv(path, dtype={"psg": str})
    stage = d[STAGES].to_numpy(float) * (1.0 - d["gap_head"].to_numpy(float))[:, None]
    proba = np.column_stack([stage, d["gap_head"].to_numpy(float)])
    y = d["psg"].map({s: i for i, s in enumerate(STAGES)}).fillna(-1).astype(int).to_numpy()
    return y, proba


def night_counts(paths: list[Path]) -> np.ndarray:
    return np.array([stage_counts(np.load(p)["y"]) for p in paths])


def seed_outputs(predictions: Path, k: int) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Per seed of fold ``k``: recording file stem -> ``(y_true, proba)``.

    Seeds are the ``fold<k>_seed_<s>`` directories of ``predictions``, or the CSVs of ``predictions``
    itself for a single model.
    """
    dirs = sorted(predictions.glob(f"fold{k}_seed_*")) or [predictions]
    return [{f.stem: load_subject(f) for f in d.glob("*.csv")} for d in dirs]


def fold_divisor(fold_dir: Path, cache: Path, bidsleep: np.ndarray) -> tuple[np.ndarray, list[str]]:
    train_sets = json.loads((fold_dir / "config/cv.json").read_text())["train_sets"]
    paths = [p for d in train_sets for p in sorted(cache.glob(f"{d}_subject_*"))]
    return adult_stage_divisor(night_counts(paths), bidsleep, BETA), train_sets


def adult_model(root: Path, cache: Path, bidsleep: np.ndarray, plan) -> tuple[dict[str, Subjects], dict]:
    """SleepAccel and SleepAccel-Clinical recordings of an arm's five fold models, decoded with the divisor."""
    sleepaccel: Subjects = {}
    clinical: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    divisors, held_out, train_sets = [], {}, {}
    for k in FOLDS:
        fold_dir = root / f"fold{k}"
        divisor, train_sets[k] = fold_divisor(fold_dir, cache, bidsleep)
        divisors.append(divisor)
        seeds = seed_outputs(fold_dir / "predictions", k)
        held_out[k] = []
        for name in seeds[0]:
            y = seeds[0][name][0]
            mean = np.mean([s[name][1] for s in seeds], axis=0)
            if name.startswith(plan.fold_names[k] + "_"):
                sid = name[len(plan.fold_names[k]) + 1:]
                held_out[k].append(sid)
                sleepaccel[sid] = (y, mean, hybrid_calls(mean, mean, divisor))
            elif name.startswith("SleepAccel-Clinical_"):
                clinical.setdefault(name[len("SleepAccel-Clinical_"):], []).append((y, mean))
    mean_divisor = np.exp(np.mean(np.log(divisors), axis=0))
    clinical_out: Subjects = {}
    for sid, per_fold in clinical.items():
        mean = np.mean([p for _, p in per_fold], axis=0)
        clinical_out[sid] = (per_fold[0][0], mean, hybrid_calls(mean, mean, mean_divisor))
    return ({"SleepAccel": sleepaccel, "SleepAccel-Clinical": clinical_out},
            {"held_out": held_out, "train_sets": train_sets})


def pediatric_model(prediction_dirs: dict[int, Path]) -> tuple[dict[str, Subjects], dict]:
    """Weaver recordings of a pediatric run's five fold models, decoded by argmax of the seed-mean output."""
    weaver: Subjects = {}
    held_out = {}
    for k in FOLDS:
        seeds = seed_outputs(prediction_dirs[k], k)
        held_out[k] = []
        for name in seeds[0]:
            sid = name.split("_", 1)[1]
            mean = np.mean([s[name][1] for s in seeds], axis=0)
            weaver[sid] = (seeds[0][name][0], mean, mean.argmax(axis=1))
            held_out[k].append(sid)
    return {"Weaver": weaver}, {"held_out": held_out}


def recording_metrics(logger: MetricsLogger, y: np.ndarray, proba: np.ndarray, calls: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        recorded = logger.record(Path(tmp) / "m.csv", y_true=y, y_pred_proba=proba, y_pred=calls, run_id="",
                                 fold="", epoch=0, subject_id="", dataset="", training_dataset="")
    r = {m["metric"]: m["value"] for m in recorded}
    rem = y == REM
    return {**{m: r[m] for m in METRICS[:4]},
            "rem_recall": float(np.mean(calls[rem] == REM)) if rem.any() else np.nan}


def score_model(logger: MetricsLogger, name: str, data: dict[str, Subjects]) -> pd.DataFrame:
    rows = [{"model": name, "cohort": cohort, "subject_id": sid, "source": "predictions",
             **recording_metrics(logger, y, p, calls)}
            for cohort, subjects in data.items() for sid, (y, p, calls) in subjects.items()]
    return pd.DataFrame(rows)


def training_phase_metrics(root: Path) -> pd.DataFrame:
    """Per-recording metrics that a pediatric run's training phases recorded for their held-out folds."""
    rows = pd.concat([pd.read_csv(root / f"fold{k}/cv_results_4_stages.csv") for k in FOLDS])
    wide = rows[rows.metric.isin(METRICS[:4])].pivot(index="subject_id", columns="metric", values="value")
    return wide[METRICS[:4]]


def paired_difference(a: pd.Series, b: pd.Series, rng: np.random.Generator) -> dict:
    """Retrain − baseline over the recordings both scored: mean, bootstrap 95% interval, Wilcoxon p."""
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    d = (both.a - both.b).to_numpy(float)
    boot = d[rng.integers(len(d), size=(BOOTSTRAPS, len(d)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    return {"n": len(d), "baseline_mean": both.b.mean(), "retrain_mean": both.a.mean(), "diff": d.mean(),
            "ci_low": low, "ci_high": high, "p": p}


def fmt(x: float, digits: int = 3, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "n/a"
    x = 0.0 if round(x, digits) == 0 else x
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace("-", "−")


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|:---|" + "---:|" * (len(header) - 1)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def split_report(name: str, info: dict, other_name: str, other: dict) -> list[str]:
    """One line per fold in which two models held out different recordings."""
    lines = []
    for k in FOLDS:
        a, b = set(info["held_out"][k]), set(other["held_out"][k])
        if a != b:
            lines.append(f"  - fold {k}: {name} holds out {len(a)}, {other_name} {len(b)}; only {name}: "
                         f"{len(a - b)}, only {other_name}: {len(b - a)}")
    return lines


def main(results: Path) -> None:
    run = results / RUN
    out = run / "cv-scores"
    out.mkdir(exist_ok=True)
    devices = results / "bidoze-adult-devices-0"
    labels = devices / "setup/labels" / CACHE_NAME
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    sleepaccel_plan = read_plan(BIDOZE / "config/splits/sleepaccel_5fold.json")
    bidsleep = night_counts(sorted(labels.glob("bidsleep_subject_*")))

    adult_inputs = {
        "v0.1.0 adult": (devices / "matched", devices / "setup/cache/matched" / CACHE_NAME),
        "adult all (pre-gap)": (devices / "all", devices / "setup/cache/all" / CACHE_NAME),
        "adult-gap-matched": (run / "adult-gap-matched", devices / "setup/cache/matched" / CACHE_NAME),
        "adult-gap-all": (run / "adult-gap-all", devices / "setup/cache/all" / CACHE_NAME),
    }
    models, infos = {}, {}
    for name, (root, cache) in adult_inputs.items():
        models[name], infos[name] = adult_model(root, cache, bidsleep, sleepaccel_plan)

    v010_ped = results / "bidoze-pediatric-v0.1.0"
    rerun = run / "cv-scores/v0.1.0-pediatric-predictions"
    models[RERUN], _ = pediatric_model({k: rerun / f"fold{k}/predictions" for k in FOLDS})
    models["pediatric-gap"], infos["pediatric-gap"] = pediatric_model(
        {k: run / f"pediatric-gap/fold{k}/predictions" for k in FOLDS})
    frames = {name: score_model(logger, name, data) for name, data in models.items()}

    # Pediatric: the four metrics are the training phases' own (every held-out recording, both models);
    # REM recall comes from predictions. The v0.1.0 re-run reproduces the phase's metrics for only some
    # recordings, so its REM recall is kept for those.
    phase = training_phase_metrics(v010_ped / "training")
    gap_phase = training_phase_metrics(run / "pediatric-gap")
    infos["v0.1.0 pediatric"] = {"held_out": {
        k: sorted(pd.read_csv(v010_ped / f"training/fold{k}/cv_results_4_stages.csv").subject_id.unique())
        for k in FOLDS}}
    rerun_frame = frames[RERUN].set_index("subject_id")
    difference = (rerun_frame[METRICS[:4]] - phase.loc[rerun_frame.index]).abs()
    reproduced = difference.index[(difference < REPRODUCED).all(axis=1)]
    v010 = phase.assign(model="v0.1.0 pediatric", cohort="Weaver", source="training phase",
                        rem_recall=rerun_frame.rem_recall.where(rerun_frame.index.isin(reproduced)))
    gap = frames["pediatric-gap"].set_index("subject_id")
    gap_difference = float((gap[METRICS[:4]] - gap_phase.loc[gap.index]).abs().to_numpy().max())
    gap_frame = gap_phase.assign(model="pediatric-gap", cohort="Weaver", source="training phase",
                                 rem_recall=gap.rem_recall)
    frames["v0.1.0 pediatric"] = v010.rename_axis("subject_id").reset_index()
    frames["pediatric-gap"] = gap_frame.rename_axis("subject_id").reset_index()

    per_recording = pd.concat(frames.values(), ignore_index=True)
    per_recording.to_csv(out / "per_recording.csv", index=False)

    comparisons = [
        ("SleepAccel", "adult-gap-matched", "v0.1.0 adult"),
        ("SleepAccel-Clinical", "adult-gap-matched", "v0.1.0 adult"),
        ("SleepAccel", "adult-gap-all", "adult all (pre-gap)"),
        ("SleepAccel-Clinical", "adult-gap-all", "adult all (pre-gap)"),
        ("SleepAccel", "adult-gap-all", "v0.1.0 adult"),
        ("SleepAccel-Clinical", "adult-gap-all", "v0.1.0 adult"),
        ("Weaver", "pediatric-gap", "v0.1.0 pediatric"),
        ("Weaver", "pediatric-gap", RERUN),
    ]
    rng = np.random.default_rng(0)
    paired_rows = []
    for cohort, retrain, baseline in comparisons:
        a = per_recording[(per_recording.model == retrain) & (per_recording.cohort == cohort)].set_index("subject_id")
        b = per_recording[(per_recording.model == baseline) & (per_recording.cohort == cohort)].set_index("subject_id")
        for metric in METRICS:
            paired_rows.append({"cohort": cohort, "retrain": retrain, "baseline": baseline, "metric": metric,
                                **paired_difference(a[metric], b[metric], rng)})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired.csv", index=False)

    pairs = [("adult-gap-matched", "v0.1.0 adult"), ("adult-gap-all", "adult all (pre-gap)"),
             ("adult-gap-all", "v0.1.0 adult"), ("pediatric-gap", "v0.1.0 pediatric")]
    lines = ["# Gap-rule retrains against the v0.1.0 release, cross-validation folds", "",
             "Mean over recordings. Adult: seed-mean output, stage divisor; pediatric: argmax. "
             "Differences are retrain − baseline over recordings both scored, with a bootstrap 95% "
             "interval over recordings and a paired Wilcoxon p.", "", "## Split check", ""]
    for retrain, baseline in pairs:
        differing = split_report(retrain, infos[retrain], baseline, infos[baseline])
        lines.append(f"- {retrain} vs {baseline}: held-out recordings "
                     + ("identical in every fold." if not differing else "differ:"))
        lines += differing
        if "train_sets" in infos[retrain]:
            same = all(infos[retrain]["train_sets"][k] == infos[baseline]["train_sets"][k] for k in FOLDS)
            lines.append(f"- {retrain} vs {baseline}: training sets " + ("identical in every fold." if same else "differ."))
    lines += [f"- v0.1.0 pediatric: the four metrics are the training phases' own for all {len(phase)} held-out "
              f"recordings. Predictions were re-run from the feature cache for {len(rerun_frame)} of them; the "
              f"re-run reproduces the phase's four metrics (to {REPRODUCED:g}) for {len(reproduced)}, and only "
              f"those {len(reproduced)} have a REM recall. `{RERUN}` scores every re-run recording from its own "
              f"predictions ({len(rerun_frame)} recordings; its metrics differ from the phase's by up to "
              f"{float(difference.to_numpy().max()):.2f}).",
              f"- pediatric-gap: the four metrics are its training phases' own; its predictions reproduce them to "
              f"{gap_difference:.1e}.", ""]

    lines += ["## Means", ""]
    rows = []
    for name, frame in frames.items():
        for cohort, g in frame.groupby("cohort"):
            rows.append([name, cohort, str(len(g)), *[fmt(g[m].mean()) for m in METRICS]])
    lines += [markdown_table(["model", "cohort", "n", *[LABELS[m] for m in METRICS]], rows), ""]

    lines += ["## Paired differences", ""]
    for cohort, retrain, baseline in comparisons:
        lines += [f"### {cohort}: {retrain} − {baseline}", ""]
        sub = paired[(paired.cohort == cohort) & (paired.retrain == retrain) & (paired.baseline == baseline)]
        lines += [markdown_table(
            ["metric", "n", baseline, retrain, "difference", "95% interval", "p"],
            [[LABELS[r.metric], str(r.n), fmt(r.baseline_mean), fmt(r.retrain_mean), fmt(r.diff, signed=True),
              f"[{fmt(r.ci_low, signed=True)}, {fmt(r.ci_high, signed=True)}]", f"{r.p:.3f}"]
             for r in sub.itertuples()]), ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

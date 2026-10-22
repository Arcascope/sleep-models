"""Score the Weaver-set experiment and compare the arms, paired over draws.

Models: per arm (matched, all, all-adult-share) and draw 0-4, the 3-seed ensemble (mean of
probabilities) of ``<run>/training/<arm>/draw<d>/predictions/draw<d>_seed_<s>``, decoded as the adult
release is: the ensemble decides wake, gap or sleep; on sleep epochs, light/deep/REM are divided by the
adult stage divisor. The divisor is the draw's original one, the same for every arm: the stage
counts of the draw's TriDoze training nights (101 Weaver children + its training adults, from the
October 15 joint-ensemble run's ``setup/cv`` and ``setup/cache``) against the BIDsleep nights
(``adult_stage_divisor``, beta 0.999).

Per draw, arm and cohort (SleepAccel, SleepAccel-Clinical; the draw's test adults): the mean over test
subjects of 4-stage kappa, sleep/wake kappa, sleep sensitivity, wake specificity and the recall of each
stage (a gap call counts as a miss; subjects without that stage in their PSG are left out of its mean).

Comparisons ``matched - all`` and ``all-adult-share - all`` per draw, then the mean, SD (ddof 1) over
draws and the paired 95% t-interval of the mean (4 degrees of freedom for 5 draws).

Writes to ``<run>/analysis/``: draws.csv, subjects.csv, paired.csv, summary.md.

Usage: python summarize.py [--run RUN] [--pull] [--baseline-run RUN0] [--labels-root ROOT] [--draws 0 1 2 3 4]
``--pull`` first copies the run's outputs from
gs://arcascope-sleep-models-data/training_outputs/<run id>/ into ``<run>``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[2]
sys.path.insert(0, str(BIDOZE / "experiments/adult_devices"))

import arcasleep.evaluation  # noqa: E402,F401  (registers the metrics)
from arcasleep.data.cache import discover_cached_subjects  # noqa: E402
from arcasleep.training.config import load_metrics_config  # noqa: E402
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls  # noqa: E402
from card_metrics import subject_metrics  # noqa: E402
from pisces_lite.metrics.logger import MetricsLogger  # noqa: E402
from score import BETA, CACHE_NAME, STAGES, load_subject, night_counts  # noqa: E402

RESULTS = BIDOZE / "results"
RUN = RESULTS / "weaver-set-q1-2026-10-21"
BUCKET = "gs://arcascope-sleep-models-data/training_outputs"
BASELINE = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15")
LABELS = BASELINE.parent / "bidoze-adult-devices-0/setup/labels"
ARMS = ["matched", "all", "all-adult-share"]
COMPARISONS = [("matched", "all"), ("all-adult-share", "all")]
SEEDS = [0, 1, 2]
COHORTS = {"testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical"}
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity",
           "recall_wake", "recall_light", "recall_deep", "recall_rem"]
HEADLINE = ["kappa_4stage", "kappa_sleep_wake"]
MARGIN = 0.02


def stage_recalls(y: np.ndarray, calls: np.ndarray) -> dict[str, float]:
    return {f"recall_{name}": float(np.mean(calls[y == k] == k)) if np.any(y == k) else np.nan
            for k, name in enumerate(STAGES)}


def draw_divisor(baseline: Path, draw: int, bidsleep: np.ndarray) -> np.ndarray:
    name = f"draw{draw}"
    cv = json.loads((baseline / f"setup/cv/{name}.json").read_text())
    cache = baseline / f"setup/cache/{name}" / CACHE_NAME
    paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
    return adult_stage_divisor(night_counts(paths), bidsleep, BETA)


def score_run(run: Path, baseline: Path, labels_root: Path, draws: list[int]) -> pd.DataFrame:
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    bidsleep = night_counts(
        [s.path for s in discover_cached_subjects(["bidsleep"], labels_root / CACHE_NAME)["bidsleep"]])
    split = json.loads((HERE.parent / "draws.json").read_text())
    rows = []
    for d in draws:
        divisor = draw_divisor(baseline, d, bidsleep)
        for arm in ARMS:
            dirs = [run / f"training/{arm}/draw{d}/predictions/draw{d}_seed_{s}" for s in SEEDS]
            missing = [p for p in dirs if not (p.is_dir() and any(p.glob("*.csv")))]
            if missing:
                print(f"skip draw {d} {arm}: no predictions in {[p.name for p in missing]}")
                continue
            for dataset, cohort in COHORTS.items():
                for sid in split[f"draw{d}"][dataset]:
                    loaded = [load_subject(p / f"{dataset}_{sid}.csv") for p in dirs]
                    y, proba = loaded[0][0], np.mean([q for _, q in loaded], axis=0)
                    calls = hybrid_calls(proba, proba, divisor)
                    m = subject_metrics(logger, y, proba, calls)
                    scored = (y >= 0) & (y < len(STAGES))
                    rows.append({"draw": d, "arm": arm, "cohort": cohort, "subject_id": sid,
                                 **{k: m[k] for k in METRICS[:4]}, **stage_recalls(y[scored], calls[scored])})
    return pd.DataFrame(rows)


def per_draw(subjects: pd.DataFrame) -> pd.DataFrame:
    return subjects.groupby(["draw", "arm", "cohort"])[METRICS].mean().reset_index()


def paired(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in COMPARISONS:
        for cohort, part in draws.groupby("cohort"):
            wide = part.pivot(index="draw", columns="arm", values=METRICS)
            for metric in METRICS:
                if a not in wide[metric] or b not in wide[metric]:
                    continue
                delta = (wide[metric][a] - wide[metric][b]).dropna().to_numpy(float)
                n = len(delta)
                if n < 2:
                    continue
                sd = delta.std(ddof=1)
                half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
                rows.append({"comparison": f"{a} - {b}", "cohort": cohort, "metric": metric, "draws": n,
                             "mean": delta.mean(), "sd": sd, "ci_low": delta.mean() - half,
                             "ci_high": delta.mean() + half, "draws_higher": int(np.sum(delta > 0))})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> str:
    cells = df.astype(str)
    return "\n".join(["| " + " | ".join(cells.columns) + " |", "|" + ":---|" * len(cells.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells.to_numpy()])


def fmt(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")


def report(draws: pd.DataFrame, comparison: pd.DataFrame) -> str:
    out = ["# Weaver-set experiment (October 21 §6b, Q1)", ""]
    out += ["## Per arm: mean over draws (each draw: mean over its test subjects)", ""]
    mean = draws.groupby(["cohort", "arm"])[METRICS].mean().reset_index()
    out += [table(mean.assign(**{m: mean[m].map(fmt) for m in METRICS})), ""]
    for cohort in COHORTS.values():
        out += [f"## {cohort} per draw", ""]
        part = draws[draws.cohort == cohort].drop(columns="cohort").sort_values(["arm", "draw"])
        out += [table(part.assign(**{m: part[m].map(fmt) for m in METRICS})), ""]
    out += ["## Paired differences over draws", "",
            "Mean ± SD of the per-draw difference; the interval is the paired 95% t-interval of the mean.", ""]
    shown = comparison.assign(
        value=[f"{fmt(r['mean'])} ± {fmt(r['sd'])}" for _, r in comparison.iterrows()],
        interval=[f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" for _, r in comparison.iterrows()])
    out += [table(shown[["comparison", "cohort", "metric", "draws", "value", "interval", "draws_higher"]]), ""]
    out += ["## Decision rule (log §6b)", ""]
    for challenger in ("matched", "all-adult-share"):
        sub = comparison[(comparison.comparison == f"{challenger} - all") & (comparison.cohort == "SleepAccel")
                         & comparison.metric.isin(HEADLINE)]
        for _, r in sub.iterrows():
            passes = r["mean"] >= MARGIN and r["ci_low"] > 0
            out.append(f"- {challenger} − all, SleepAccel {r['metric']}: {fmt(r['mean'])} "
                       f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]; ≥ {MARGIN} with the interval above 0: "
                       f"{'yes' if passes else 'no'}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--pull", action="store_true", help="copy the run's outputs from GCS first")
    ap.add_argument("--baseline-run", type=Path, default=BASELINE)
    ap.add_argument("--labels-root", type=Path, default=LABELS)
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    if args.pull:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", f"{BUCKET}/{args.run.name}/", str(args.run)],
                       check=True)
    out = args.run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    subjects = score_run(args.run, args.baseline_run, args.labels_root, args.draws)
    if subjects.empty:
        raise SystemExit("no predictions found")
    draws = per_draw(subjects)
    comparison = paired(draws)
    subjects.to_csv(out / "subjects.csv", index=False)
    draws.to_csv(out / "draws.csv", index=False)
    comparison.to_csv(out / "paired.csv", index=False)
    text = report(draws, comparison)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

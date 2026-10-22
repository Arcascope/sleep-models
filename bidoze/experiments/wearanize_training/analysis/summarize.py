"""Score the Wearanize+ training experiment and compare the arms, paired over draws.

Models: per arm (g-base, g-wearanize, f2-base, f2-wearanize) and draw 0-4, the 3-seed ensemble (mean
of probabilities) of ``<run>/training/<arm>/draw<d>/predictions/draw<d>_seed_<s>``. Two decodings:
``hybrid`` as the adult release decodes (the ensemble decides wake, gap or sleep; on sleep epochs,
light/deep/REM are divided by the adult stage divisor) and ``argmax`` of the ensemble's probabilities.
The divisor is the draw's original one, the same for every arm: the stage counts of the draw's TriDoze
training nights (from the October 15 joint-ensemble run's ``setup/cv`` and ``setup/cache``) against the
BIDsleep nights (``adult_stage_divisor``, beta 0.999).

g-base is the Weaver-set Q1 ``all`` arm: ``task.sh g-base/draw<d>`` scores its checkpoints on this
experiment's test sets (``WEARANIZE_BASE_RUN``), writing the predictions under the same layout.

Cohorts, from the test sets in the predictions: SleepAccel (``testSleepAccel``, the draw's test adults),
SleepAccel-Clinical (``testClinical``, reported separately), Wearanize (``testWearanize``, the draw's
held-out fold), AAUWSS (``testAAUWSS``, all 13 nights in every draw). EESM19 (``testEESM19``) is scored
when its predictions exist. Per draw, arm, cohort and decoding: the mean over test subjects of 4-stage
kappa, sleep/wake kappa, sleep sensitivity, wake specificity and the recall of each stage (a gap call
counts as a miss; subjects without that stage in their PSG are left out of its mean).

Comparisons per cohort: ``g-wearanize - g-base`` and ``f2-wearanize - f2-base`` (the effect of
Wearanize+ per architecture), ``f2-base - g-base`` and ``f2-wearanize - g-wearanize`` (F2 against G
per data arm). Per draw the difference of the draw means, then the mean, SD (ddof 1) and the paired 95%
t-interval of the mean over draws (4 degrees of freedom for 5 draws). The draws of AAUWSS score the same
13 nights and the draws of SleepAccel share test adults, so those intervals are narrower than
independent draws would give; ``subject_paired.csv`` repeats the comparison per subject (each subject's
mean over the draws that score it, paired t over subjects).

Writes to ``<run>/analysis/``: subjects.csv, draws.csv, paired.csv, subject_paired.csv, summary.md.

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
RUN = RESULTS / "wearanize-training-2026-10-22"
BUCKET = "gs://arcascope-sleep-models-data/training_outputs"
BASELINE = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15")
LABELS = BASELINE.parent / "bidoze-adult-devices-0/setup/labels"
ARMS = ["g-base", "g-wearanize", "f2-base", "f2-wearanize"]
COMPARISONS = [("g-wearanize", "g-base"), ("f2-wearanize", "f2-base"),
               ("f2-base", "g-base"), ("f2-wearanize", "g-wearanize")]
SEEDS = [0, 1, 2]
#: Test set name -> cohort label; a test set with no predictions is left out (EESM19 until built).
COHORTS = {"testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical",
           "testWearanize": "Wearanize", "testAAUWSS": "AAUWSS", "testEESM19": "EESM19"}
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity",
           "recall_wake", "recall_light", "recall_deep", "recall_rem"]


def stage_recalls(y: np.ndarray, calls: np.ndarray) -> dict[str, float]:
    return {f"recall_{name}": float(np.mean(calls[y == k] == k)) if np.any(y == k) else np.nan
            for k, name in enumerate(STAGES)}


def draw_divisor(baseline: Path, draw: int, bidsleep: np.ndarray) -> np.ndarray:
    name = f"draw{draw}"
    cv = json.loads((baseline / f"setup/cv/{name}.json").read_text())
    cache = baseline / f"setup/cache/{name}" / CACHE_NAME
    paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
    return adult_stage_divisor(night_counts(paths), bidsleep, BETA)


def subject_ids(first_seed_dir: Path, dataset: str) -> list[str]:
    return sorted(f.stem[len(dataset) + 1:] for f in first_seed_dir.glob(f"{dataset}_*.csv"))


def score_run(run: Path, baseline: Path, labels_root: Path, draws: list[int]) -> pd.DataFrame:
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    bidsleep = night_counts(
        [s.path for s in discover_cached_subjects(["bidsleep"], labels_root / CACHE_NAME)["bidsleep"]])
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
                for sid in subject_ids(dirs[0], dataset):
                    loaded = [load_subject(p / f"{dataset}_{sid}.csv") for p in dirs]
                    y, proba = loaded[0][0], np.mean([q for _, q in loaded], axis=0)
                    scored = (y >= 0) & (y < len(STAGES))
                    for decoding, calls in (("hybrid", hybrid_calls(proba, proba, divisor)),
                                            ("argmax", np.argmax(proba, axis=-1))):
                        m = subject_metrics(logger, y, proba, calls)
                        rows.append({"draw": d, "arm": arm, "cohort": cohort, "decoding": decoding,
                                     "subject_id": sid, **{k: m[k] for k in METRICS[:4]},
                                     **stage_recalls(y[scored], calls[scored])})
    return pd.DataFrame(rows)


def per_draw(subjects: pd.DataFrame) -> pd.DataFrame:
    return subjects.groupby(["draw", "arm", "cohort", "decoding"])[METRICS].mean().reset_index()


def interval(delta: np.ndarray) -> dict[str, float]:
    n = len(delta)
    sd = delta.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"n": n, "mean": delta.mean(), "sd": sd, "ci_low": delta.mean() - half,
            "ci_high": delta.mean() + half, "higher": int(np.sum(delta > 0))}


def paired(frame: pd.DataFrame, index: str) -> pd.DataFrame:
    """Per cohort and decoding, each comparison's difference per ``index`` value, then its interval.

    A value of ``index`` with several rows of an arm (a subject scored in several draws) is averaged.
    """
    rows = []
    for (cohort, decoding), part in frame.groupby(["cohort", "decoding"]):
        wide = part.pivot_table(index=index, columns="arm", values=METRICS)
        for a, b in COMPARISONS:
            for metric in METRICS:
                if a not in wide[metric] or b not in wide[metric]:
                    continue
                delta = (wide[metric][a] - wide[metric][b]).dropna().to_numpy(float)
                if len(delta) >= 2:
                    rows.append({"comparison": f"{a} - {b}", "cohort": cohort, "decoding": decoding,
                                 "metric": metric, **interval(delta)})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> str:
    cells = df.astype(str)
    return "\n".join(["| " + " | ".join(cells.columns) + " |", "|" + ":---|" * len(cells.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells.to_numpy()])


def fmt(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")


def comparison_table(df: pd.DataFrame, unit: str) -> str:
    shown = df.assign(
        value=[f"{fmt(r['mean'])} ± {fmt(r['sd'])}" for _, r in df.iterrows()],
        interval=[f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" for _, r in df.iterrows()])
    shown = shown.rename(columns={"n": unit, "higher": f"{unit}_higher"})
    return table(shown[["comparison", "cohort", "decoding", "metric", unit, "value", "interval", f"{unit}_higher"]])


def report(draws: pd.DataFrame, comparison: pd.DataFrame, by_subject: pd.DataFrame) -> str:
    out = ["# Wearanize+ training experiment", ""]
    out += ["## Per arm: mean over draws (each draw: mean over its test subjects)", ""]
    mean = draws.groupby(["cohort", "decoding", "arm"])[METRICS].mean().reset_index()
    out += [table(mean.assign(**{m: mean[m].map(fmt) for m in METRICS})), ""]
    out += ["## Paired differences over draws", "",
            "Mean ± SD of the per-draw difference; the interval is the paired 95% t-interval of the mean. "
            "SleepAccel-Clinical is its own cohort.", ""]
    if comparison.empty:
        return "\n".join(out + ["Fewer than 2 draws have both arms scored.", ""]) + "\n"
    out += [comparison_table(comparison, "draws"), ""]
    out += ["## Paired differences over subjects", "",
            "Each subject's mean over the draws that score it, then the paired t-interval over subjects.", ""]
    out += [comparison_table(by_subject, "subjects") if not by_subject.empty else "No subjects.", ""]
    for cohort in COHORTS.values():
        part = draws[(draws.cohort == cohort) & (draws.decoding == "hybrid")]
        if part.empty:
            continue
        part = part.drop(columns=["cohort", "decoding"]).sort_values(["arm", "draw"])
        out += [f"## {cohort} per draw (hybrid decoding)", "",
                table(part.assign(**{m: part[m].map(fmt) for m in METRICS})), ""]
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
    comparison = paired(draws, "draw")
    by_subject = paired(subjects, "subject_id")
    subjects.to_csv(out / "subjects.csv", index=False)
    draws.to_csv(out / "draws.csv", index=False)
    comparison.to_csv(out / "paired.csv", index=False)
    by_subject.to_csv(out / "subject_paired.csv", index=False)
    text = report(draws, comparison, by_subject)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

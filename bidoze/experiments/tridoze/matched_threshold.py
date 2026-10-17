"""Compare the ensembles at a matched sleep sensitivity, with each model's sleep/wake threshold set on the eval share.

Each model's per-epoch sleep score is P(sleep | not gap) = (P(light) + P(deep) + P(REM)) / (1 − P(gap)).
An epoch whose 5-way argmax is gap keeps its gap call (neither sleep nor wake); every other epoch is
called sleep when its score is ≥ the model's threshold, else wake. Sleep epochs are staged as the
release stages them: light/deep/REM divided by the draw's adult stage divisor, then argmax.

Thresholds, per draw and model, are set on the draw's eval share: the training recordings that
every arm holds out of its fit (train_predictions.py). The target is the pooled sleep sensitivity
of independent ×3 at its decoded calls (5-way argmax) on the same recordings, so independent ×3
stays close to its decoded operating point and every other model is moved to that sensitivity.
Sensitivity counts a gap call on a PSG sleep epoch as a miss. Two threshold sets:
- ``eval share``: all of the draw's eval-share recordings (about 30 children and 9 adults);
- ``eval adults``: the eval share's SleepAccel and SleepAccel-Clinical training nights only.

Test subjects are the draw's held-out adults (``testSleepAccel``, ``testClinical``). Operating points:
``decoded`` (the release's calls, as evaluate.py), ``eval share`` and ``eval adults``.

Writes to ``<run>/analysis/matched-threshold/``:
- thresholds.csv per draw, threshold set and model: target sensitivity, threshold, eval-share
                 sensitivity and wake specificity reached, recordings and epochs;
- subjects.csv   per operating point, draw, model, cohort and test subject: κ sleep/wake, sleep
                 sensitivity, wake specificity, κ 4-stage, TST and WASO by PSG and by the model;
- draws.csv      the mean over each draw's test subjects;
- summary.csv    per operating point, model, cohort and metric: mean over draws ± 95% t half-width,
                 and the difference to independent ×3 at the same operating point (mean ± 95% t
                 half-width, draws higher);
- matched-threshold.png  ``--figure-model`` (default BiDoze-G ×3) minus independent ×3 per draw at
                 each operating point, by cohort.

Usage: python matched_threshold.py [--run RUN] [--draws 0 1 2 3 4] [--ensemble label ...] [--figure-model LABEL]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate  # noqa: E402
import insight  # noqa: E402
from stack import train_npz  # noqa: E402

GAP = 4
BASE = "independent ×3"
SETS = ["eval share", "eval adults"]
METRICS = ["kappa_sleep_wake", "sleep_sensitivity", "wake_specificity", "kappa_4stage",
           "tst_bias_minutes", "tst_abs_error_minutes", "waso_bias_minutes", "waso_abs_error_minutes"]


def sleep_score(p5: np.ndarray) -> np.ndarray:
    return p5[:, 1:4].sum(axis=1) / np.clip(1.0 - p5[:, GAP], 1e-12, None)


def calls_at(p5: np.ndarray, threshold: float, divisor: np.ndarray) -> np.ndarray:
    staged = 1 + (p5[:, 1:4] / divisor[1:4]).argmax(axis=1)
    calls = np.where(sleep_score(p5) >= threshold, staged, 0)
    return np.where(p5.argmax(axis=1) == GAP, GAP, calls)


def pooled(recordings: list[tuple[np.ndarray, np.ndarray]]):
    """``(y, p5)`` over the PSG-scored epochs of the recordings."""
    y = np.concatenate([y for y, _ in recordings])
    p = np.concatenate([p for _, p in recordings])
    keep = (y >= 0) & (y < GAP)
    return y[keep], p[keep]


def decoded_sensitivity(y: np.ndarray, p5: np.ndarray) -> float:
    c = p5.argmax(axis=1)
    return float(np.mean((c[y > 0] > 0) & (c[y > 0] < GAP)))


def threshold_for(y: np.ndarray, p5: np.ndarray, target: float) -> float:
    """The score threshold at which this share of PSG sleep epochs is called sleep (gap calls are misses)."""
    score = np.where(p5.argmax(axis=1) == GAP, -np.inf, sleep_score(p5))
    return float(np.quantile(score[y > 0], 1.0 - target, method="higher"))


def rates(y: np.ndarray, p5: np.ndarray, threshold: float) -> tuple[float, float]:
    c = calls_at(p5, threshold, np.ones(5))
    sleep_call = (c > 0) & (c < GAP)
    return float(np.mean(sleep_call[y > 0])), float(np.mean(c[y == 0] == 0))


def eval_share(dirs: list[Path], run: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stores = [np.load(train_npz(run, p)) for p in dirs]
    return {k: (stores[0][f"__labels__/{k}"].astype(int), np.mean([s[k] for s in stores], axis=0))
            for k in stores[0]["__eval__"]}


def half_width(x: pd.Series) -> float:
    x = x.dropna()
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


SHOWN = {"kappa_sleep_wake": ("κ sleep/wake", 1), "sleep_sensitivity": ("Sleep sensitivity (pts)", 100),
         "wake_specificity": ("Wake specificity (pts)", 100), "tst_abs_error_minutes": ("TST MAE (min)", 1),
         "waso_abs_error_minutes": ("WASO MAE (min)", 1)}


def figure(draw_table: pd.DataFrame, model: str, path: Path) -> None:
    points = ["decoded", *SETS]
    cohorts = list(insight.COHORTS.values())
    fig, axes = plt.subplots(len(cohorts), len(SHOWN), figsize=(3.0 * len(SHOWN), 3.2 * len(cohorts)), squeeze=False)
    for r, cohort in enumerate(cohorts):
        for c, (metric, (title, scale)) in enumerate(SHOWN.items()):
            ax = axes[r, c]
            for x, point in enumerate(points):
                t = draw_table[(draw_table.point == point) & (draw_table.cohort == cohort)].set_index(["model", "draw"])[metric]
                diffs = scale * (t.loc[model] - t.loc[BASE]).dropna()
                ax.scatter(np.full(len(diffs), x) + np.linspace(-0.12, 0.12, len(diffs)), diffs, s=14, color="#4c72b0", alpha=0.7)
                ax.errorbar(x + 0.25, diffs.mean(), yerr=half_width(diffs), fmt="o", color="black", capsize=3)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks(range(len(points)), ["Decoded", "Eval share", "Eval adults"], fontsize=8)
            ax.set_title(title if r == 0 else "", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{cohort}\n{model} − {BASE}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=insight.RUN)
    ap.add_argument("--baseline-run", type=Path, default=insight.BASELINE)
    ap.add_argument("--labels-root", type=Path, default=evaluate.LABELS)
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ensemble", nargs="+", default=None)
    ap.add_argument("--figure-model", default="BiDoze-G ×3")
    args = ap.parse_args()
    out = args.run / "analysis/matched-threshold"
    out.mkdir(parents=True, exist_ok=True)

    logger = evaluate.MetricsLogger(evaluate.load_metrics_config(evaluate.BIDOZE / "config/metrics.json")[0])
    bidsleep = evaluate.night_counts([s.path for s in evaluate.discover_cached_subjects(
        ["bidsleep"], args.labels_root / evaluate.CACHE_NAME)["bidsleep"]])
    draws = json.loads((args.baseline_run / "setup/draws.json").read_text())
    defs = insight.ensembles(args.run, args.baseline_run)
    labels = args.ensemble or list(defs)

    threshold_rows, rows = [], []
    for d in args.draws:
        name = f"draw{d}"
        cv = json.loads((args.baseline_run / f"setup/cv/{name}.json").read_text())
        cache = args.baseline_run / f"setup/cache/{name}" / evaluate.CACHE_NAME
        train_paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
        divisor = np.asarray(evaluate.adult_stage_divisor(evaluate.night_counts(train_paths), bidsleep, evaluate.BETA))

        shares = {}
        for label in labels:
            dirs = defs[label](d)
            if not all(p.is_dir() for p in dirs) or not all(train_npz(args.run, p).exists() for p in dirs):
                print(f"skip {label} draw {d}: predictions missing")
                continue
            recs = eval_share(dirs, args.run)
            shares[label] = (dirs, {"eval share": pooled(list(recs.values())),
                                    "eval adults": pooled([v for k, v in recs.items()
                                                           if not k.startswith("weaverDraw/")])},
                             {"eval share": len(recs), "eval adults": sum(not k.startswith("weaverDraw/") for k in recs)})
        if BASE not in shares:
            print(f"skip draw {d}: {BASE} eval share missing")
            continue
        targets = {s: decoded_sensitivity(*shares[BASE][1][s]) for s in SETS}

        for label, (dirs, pooled_share, n_recs) in shares.items():
            thresholds = {}
            for s in SETS:
                y, p5 = pooled_share[s]
                thresholds[s] = threshold_for(y, p5, targets[s])
                sens, spec = rates(y, p5, thresholds[s])
                threshold_rows.append({"draw": d, "set": s, "model": label, "target_sensitivity": targets[s],
                                       "threshold": thresholds[s], "decoded_sensitivity": decoded_sensitivity(y, p5),
                                       "sensitivity": sens, "wake_specificity": spec,
                                       "recordings": n_recs[s], "epochs": len(y)})
            for dataset, cohort in insight.COHORTS.items():
                for sid in draws[name][dataset]:
                    loaded = [evaluate.load_subject(p / f"{dataset}_{sid}.csv") for p in dirs]
                    y, p5 = loaded[0][0], np.mean([q for _, q in loaded], axis=0)
                    scored = (y >= 0) & (y < GAP)
                    points = {"decoded": evaluate.hybrid_calls(p5, p5, divisor),
                              **{s: calls_at(p5, thresholds[s], divisor) for s in SETS}}
                    for point, calls in points.items():
                        m = evaluate.subject_metrics(logger, y, p5, calls)
                        mins = evaluate.sleep_minutes(y[scored], calls[scored])
                        tst = mins["pred_tst_minutes"] - mins["psg_tst_minutes"]
                        waso = mins["pred_waso_minutes"] - mins["psg_waso_minutes"]
                        rows.append({"point": point, "draw": d, "model": label, "cohort": cohort, "subject_id": sid,
                                     **{k: m[k] for k in METRICS[:4]}, **mins,
                                     "tst_bias_minutes": tst, "tst_abs_error_minutes": abs(tst),
                                     "waso_bias_minutes": waso, "waso_abs_error_minutes": abs(waso)})

    pd.DataFrame(threshold_rows).to_csv(out / "thresholds.csv", index=False)
    subjects = pd.DataFrame(rows)
    subjects.to_csv(out / "subjects.csv", index=False)
    draw_table = subjects.groupby(["point", "model", "cohort", "draw"], sort=False)[METRICS].mean().reset_index()
    draw_table.to_csv(out / "draws.csv", index=False)

    summary = []
    for (point, model, cohort), g in draw_table.groupby(["point", "model", "cohort"], sort=False):
        base = draw_table[(draw_table.point == point) & (draw_table.model == BASE)
                          & (draw_table.cohort == cohort)].set_index("draw")
        g = g.set_index("draw")
        for metric in METRICS:
            row = {"point": point, "model": model, "cohort": cohort, "metric": metric, "draws": len(g),
                   "mean": g[metric].mean(), "half_width": half_width(g[metric])}
            if model != BASE:
                diffs = (g[metric] - base[metric]).dropna()
                row |= {"diff": diffs.mean(), "diff_half_width": half_width(diffs),
                        "draws_higher": int((diffs > 0).sum()),
                        **{f"diff_draw{d}": v for d, v in diffs.items()}}
            summary.append(row)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    if args.figure_model in set(draw_table.model):
        figure(draw_table, args.figure_model, out / "matched-threshold.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

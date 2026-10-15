"""Score the joint-ensemble experiment with the released decoding, per draw and over draws.

Models per draw (see prepare.py for the arms):
- independent ×3: the mean output of independent seeds 0–2, as the release;
- independent ×2: seeds 0 and 1;
- independent seed s: one seed;
- joint ×3, joint ×2: the joint fit's mean output;
- joint ×3 member m, joint ×2 member m: one member of a joint fit by itself.

Every model is decoded like the release: its output decides wake or sleep; on its sleep epochs,
light/deep/REM are divided by the adult stage divisor of the draw's training nights (BIDsleep
reference). Test subjects are the draw's 30 held-out adults, reported by cohort.

Writes to ``<run>/analysis/``:
- fits.csv      best eval κ, best epoch, epochs run and training time of every fit;
- subjects.csv  per draw, model, cohort and subject;
- draws.csv     per draw, model, cohort and metric: the mean over subjects (wake specificity at
                95% sensitivity is pooled over the draw's subjects);
- summary.csv   per model, cohort and metric: the mean and SD over draws, and the difference to
                its comparison model (joint ×2 and its members against independent ×2; every
                other model against independent ×3):
                mean and SD over draws, its 95% t-interval over draws, and the Wilcoxon p over
                every (draw, subject) pair.

Usage: python score.py <run dir> <BIDsleep labels cache root>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adult_devices"))

import arcasleep.evaluation  # noqa: E402, F401  (registers the metrics)
from arcasleep.data.cache import discover_cached_subjects  # noqa: E402
from arcasleep.training.config import load_metrics_config  # noqa: E402
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls  # noqa: E402
from card_metrics import subject_metrics  # noqa: E402
from pisces_lite.metrics.logger import MetricsLogger  # noqa: E402
from score import BETA, BIDOZE, CACHE_NAME, load_subject, night_counts, wake_specificity_at  # noqa: E402

COHORTS = {"testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical"}
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity", "accuracy_4stage",
           "balanced_accuracy_4stage", "deep%", "tst_error_minutes", "waso_error_minutes"]
WAKE_SPEC_95 = "wake_spec_at_95_sens"


def models(run: Path, d: int) -> dict[str, list[Path]]:
    """Each model of draw ``d``: the prediction directories whose mean output it is."""
    def pred(arm: str, name: str) -> Path:
        return run / arm / f"draw{d}" / "predictions" / f"draw{d}_{name}"

    out = {
        "independent ×3": [pred("independent", f"seed_{s}") for s in range(3)],
        "independent ×2": [pred("independent", f"seed_{s}") for s in range(2)],
        **{f"independent seed {s}": [pred("independent", f"seed_{s}")] for s in range(3)},
    }
    for n in (3, 2):
        out[f"joint ×{n}"] = [pred(f"joint{n}", "seed_0")]
        out |= {f"joint ×{n} member {m}": [pred(f"joint{n}", f"seed_0_member{m}")] for m in range(n)}
    return out


def comparison(model: str) -> str | None:
    if model == "independent ×3":
        return None
    return "independent ×2" if model.startswith("joint ×2") else "independent ×3"


def fits(run: Path) -> pd.DataFrame:
    rows = []
    for log in sorted(run.glob("*/draw*/training.log")):
        arm, draw = log.parent.parent.name, log.parent.name
        text = log.read_text()
        for seed, block in re.findall(r"SEED \d+/\d+: (\d+)\n(.*?)(?=SEED \d+/\d+:|\Z)", text, re.S):
            best = re.search(r"best eval kappa ([\d.]+) at epoch (\d+)", block)
            trained = re.search(r"trained in ([\d.]+)s", block)
            epochs = re.findall(r"epoch +(\d+) +step (\d+)", block)
            rows.append({"arm": arm, "draw": draw, "seed": int(seed),
                         "best_eval_kappa": float(best.group(1)) if best else np.nan,
                         "best_epoch": int(best.group(2)) if best else np.nan,
                         "epochs": int(epochs[-1][0]) if epochs else 0,
                         "steps": int(epochs[-1][1]) if epochs else 0,
                         "train_seconds": float(trained.group(1)) if trained else np.nan})
    return pd.DataFrame(rows)


def main(run: Path, labels_root: Path) -> None:
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    bidsleep = night_counts([s.path for s in discover_cached_subjects(["bidsleep"], labels_root / CACHE_NAME)["bidsleep"]])
    draws = json.loads((run / "setup/draws.json").read_text())
    fit_table = fits(run)
    fit_table.to_csv(out / "fits.csv", index=False)
    print(fit_table.to_string(index=False))

    rows, per_draw = [], []
    for name in draws:
        d = int(name.removeprefix("draw"))
        cv = json.loads((run / f"setup/cv/{name}.json").read_text())
        cache = run / f"setup/cache/{name}" / CACHE_NAME
        train_paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
        divisor = adult_stage_divisor(night_counts(train_paths), bidsleep, BETA)
        for model, dirs in models(run, d).items():
            if not all(p.is_dir() for p in dirs):
                continue
            for dataset, cohort in COHORTS.items():
                subjects = {}
                for sid in draws[name][dataset]:
                    loaded = [load_subject(p / f"{dataset}_{sid}.csv") for p in dirs]
                    subjects[sid] = (loaded[0][0], np.mean([proba for _, proba in loaded], axis=0))
                for sid, (y, proba) in subjects.items():
                    rows.append({"draw": d, "model": model, "cohort": cohort, "subject_id": sid,
                                 **subject_metrics(logger, y, proba, hybrid_calls(proba, proba, divisor))})
                per_draw.append({"draw": d, "model": model, "cohort": cohort, "metric": WAKE_SPEC_95,
                                 "mean": 100 * wake_specificity_at(subjects, 0.95), "n": len(subjects)})
    subjects = pd.DataFrame(rows)
    subjects.to_csv(out / "subjects.csv", index=False)
    for (d, model, cohort), g in subjects.groupby(["draw", "model", "cohort"], sort=False):
        for metric in METRICS:
            per_draw.append({"draw": d, "model": model, "cohort": cohort, "metric": metric,
                             "mean": g[metric].mean(), "n": len(g)})
    draw_table = pd.DataFrame(per_draw).sort_values(["cohort", "metric", "model", "draw"])
    draw_table.to_csv(out / "draws.csv", index=False)

    summary = []
    for (model, cohort, metric), g in draw_table.groupby(["model", "cohort", "metric"], sort=False):
        row = {"model": model, "cohort": cohort, "metric": metric, "draws": len(g),
               "mean": g["mean"].mean(), "sd": g["mean"].std(ddof=1)}
        base_name = comparison(model)
        base = draw_table[(draw_table.model == base_name) & (draw_table.cohort == cohort) & (draw_table.metric == metric)]
        diffs = (g.set_index("draw")["mean"] - base.set_index("draw")["mean"]).dropna()
        if base_name is not None and len(diffs) > 1:
            half = stats.t.ppf(0.975, len(diffs) - 1) * diffs.std(ddof=1) / np.sqrt(len(diffs))
            row |= {"vs": base_name, "diff": diffs.mean(), "diff_sd": diffs.std(ddof=1),
                    "diff_low": diffs.mean() - half, "diff_high": diffs.mean() + half,
                    "draws_higher": int((diffs > 0).sum())}
            if metric in METRICS:
                a = subjects[(subjects.model == model) & (subjects.cohort == cohort)].set_index(["draw", "subject_id"])[metric]
                b = subjects[(subjects.model == base_name) & (subjects.cohort == cohort)].set_index(["draw", "subject_id"])[metric]
                paired = (a - b.reindex(a.index)).dropna()
                row |= {"subject_pairs": len(paired),
                        "p_subjects": float(stats.wilcoxon(paired).pvalue) if (paired != 0).any() else 1.0}
        summary.append(row)
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)
    show = ["kappa_4stage", "kappa_sleep_wake", "wake_specificity", WAKE_SPEC_95]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table[table.metric.isin(show)].round(4).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())

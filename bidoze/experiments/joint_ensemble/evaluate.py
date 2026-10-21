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
                every (draw, subject) pair;
- members.csv   what the members of each ensemble do by themselves, per draw, ensemble, cohort,
                member and PSG stage, over the cohort's PSG-scored epochs pooled over its test
                subjects: recall of the member's decoded calls, mean probability given to the PSG
                stage, share of the member's calls in the stage, share of the stage's epochs on
                which the member gives the PSG stage a higher probability than every other member
                (1/members without specialization), and the member's mean output entropy (bits)
                and maximum probability;
- diversity.csv per draw, ensemble and cohort: how differently the members score the same epochs
                (mean pairwise share of epochs with different calls; mean pairwise Jensen–Shannon
                divergence of the 5-way outputs, bits), the range over members of each stage's
                recall, the share of epochs that at least one member calls correctly against the
                ensemble's accuracy, and the ensemble's κ minus its members' mean κ.

The ensembles are joint ×3, joint ×2, independent ×3 (seeds 0–2) and independent ×2 (seeds 0–1).

Usage: python evaluate.py <run dir> <BIDsleep labels cache root>
"""
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
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
from score import BETA, BIDOZE, CACHE_NAME, STAGES, load_subject, night_counts, wake_specificity_at  # noqa: E402

COHORTS = {"testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical"}
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity", "accuracy_4stage",
           "balanced_accuracy_4stage", "deep%", "tst_error_minutes", "waso_error_minutes"]
WAKE_SPEC_95 = "wake_spec_at_95_sens"
ENSEMBLES = {"joint ×3": [f"joint ×3 member {m}" for m in range(3)],
             "joint ×2": [f"joint ×2 member {m}" for m in range(2)],
             "independent ×3": [f"independent seed {s}" for s in range(3)],
             "independent ×2": [f"independent seed {s}" for s in range(2)]}


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


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=1)


def jensen_shannon_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return entropy_bits((p + q) / 2) - (entropy_bits(p) + entropy_bits(q)) / 2


def members_of(d: int, ensemble: str, cohort: str, members: list[np.ndarray], y: np.ndarray,
               divisor: np.ndarray) -> tuple[list[dict], dict]:
    """members.csv rows and the diversity.csv row of one ensemble on one draw's cohort.

    ``members`` holds each member's (epochs, 5) output on the PSG-scored epochs ``y``.
    """
    calls = [hybrid_calls(p, p, divisor) for p in members]
    mean = np.mean(members, axis=0)
    ensemble_calls = hybrid_calls(mean, mean, divisor)
    true_proba = np.stack([p[np.arange(len(y)), y] for p in members])
    best = true_proba.argmax(axis=0)
    rows = []
    for m, (p, c) in enumerate(zip(members, calls)):
        common = {"draw": d, "ensemble": ensemble, "cohort": cohort, "member": m,
                  "entropy_bits": entropy_bits(p).mean(), "max_probability": p.max(axis=1).mean()}
        for k, stage in enumerate(STAGES):
            on = y == k
            rows.append({**common, "stage": stage, "recall": np.mean(c[on] == k),
                         "probability_of_psg_stage": true_proba[m, on].mean(),
                         "call_share": np.mean(c == k), "psg_share": np.mean(on),
                         "highest_of_members": np.mean(best[on] == m)})
    recalls = np.array([[np.mean(c[y == k] == k) for k in range(len(STAGES))] for c in calls])
    pairs = list(combinations(range(len(members)), 2))
    diversity = {"draw": d, "ensemble": ensemble, "cohort": cohort, "members": len(members),
                 "epochs": len(y),
                 "disagreement": np.mean([np.mean(calls[a] != calls[b]) for a, b in pairs]),
                 "jensen_shannon_bits": np.mean([jensen_shannon_bits(members[a], members[b]).mean()
                                                 for a, b in pairs]),
                 **{f"{stage}_recall_range": np.ptp(recalls[:, k]) for k, stage in enumerate(STAGES)},
                 "any_member_correct": np.mean(np.any([c == y for c in calls], axis=0)),
                 "ensemble_accuracy": np.mean(ensemble_calls == y),
                 "mean_member_accuracy": np.mean([np.mean(c == y) for c in calls])}
    return rows, diversity


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

    rows, per_draw, member_rows, diversity = [], [], [], []
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
        member_dirs = {m: dirs[0] for m, dirs in models(run, d).items() if len(dirs) == 1}
        for ensemble, names in ENSEMBLES.items():
            if not all(member_dirs[m].is_dir() for m in names):
                continue
            for dataset, cohort in COHORTS.items():
                outputs, labels = [[] for _ in names], []
                for sid in draws[name][dataset]:
                    for i, m in enumerate(names):
                        y, proba = load_subject(member_dirs[m] / f"{dataset}_{sid}.csv")
                        scored = (y >= 0) & (y < len(STAGES))
                        outputs[i].append(proba[scored])
                    labels.append(y[scored])
                r, div = members_of(d, ensemble, cohort, [np.concatenate(o) for o in outputs],
                                    np.concatenate(labels), divisor)
                member_rows += r
                diversity.append(div)
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

    diversity = pd.DataFrame(diversity)
    for ensemble, names in ENSEMBLES.items():
        for metric in ("kappa_4stage", "kappa_sleep_wake"):
            whole = draw_table[(draw_table.model == ensemble) & (draw_table.metric == metric)]
            parts = draw_table[draw_table.model.isin(names) & (draw_table.metric == metric)]
            gain = whole.set_index(["draw", "cohort"])["mean"] - parts.groupby(["draw", "cohort"])["mean"].mean()
            for (d, cohort), value in gain.dropna().items():
                diversity.loc[(diversity.draw == d) & (diversity.ensemble == ensemble)
                              & (diversity.cohort == cohort), f"{metric}_gain"] = value
    diversity.to_csv(out / "diversity.csv", index=False)
    members = pd.DataFrame(member_rows)
    members.to_csv(out / "members.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 500):
        print("\nMembers' diversity, mean over draws:")
        print(diversity.drop(columns=["draw", "epochs"]).groupby(["cohort", "ensemble"]).mean().round(3).T.to_string())
        print("\nEach member by itself, mean over draws (member numbers are not matched across draws):")
        print(members.groupby(["cohort", "ensemble", "stage", "member"])[
            ["recall", "probability_of_psg_stage", "call_share", "highest_of_members"]
        ].mean().round(3).unstack("member").to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())

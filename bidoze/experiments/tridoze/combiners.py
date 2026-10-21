"""Post-hoc ways of combining BiDoze-G members, scored on the draw's held-out adults.

Pools: the independent members (seeds 0–6), the bagged members (bags 0–6) and their union (up to 14). A
member whose test predictions or train_predictions are missing is skipped (printed). Combiners:
- ``mean``         mean of the members' 5-way probabilities (the ensemble_size.py ensemble);
- ``logit mean``   mean of log-probabilities, then softmax; this is the normalized geometric mean, so
                   ``geometric`` gives the same output (both rows are kept);
- ``median``       per-class median, renormalized;
- ``vote``         the vote share of member argmax calls plus 0.001 × the mean probability (so ties go to
                   the higher mean probability), decoded as every other output is;
- ``kappa weights``  mean weighted by each member's best eval κ from its training.log;
- ``caruana``      greedy selection with replacement (50 rounds, start from the best single member, select on
                   κ 4-stage over the eval share); the round with the best eval κ gives the chosen bag;
- ``temperature``  each member's probabilities raised to 1/T (T fitted on the eval share by NLL), then the mean;
- ``temperature per class``  the same with one T per class (wake, light, deep, REM, gap).

Methods ``kappa weights``, ``caruana`` and ``temperature`` use the eval share: the training recordings
held out of each fit (``train_predictions/*.npz``, keys ``__eval__``). The eval share also chose each
member's stopping epoch, so eval-share selection is slightly optimistic. Test metrics use test recordings only.

Metrics are ensemble_size.py's (``score_subjects``). Differences are to ``BiDoze-G ×3`` (seeds 0–2, mean) and to
the independent mean of the same pool size.

Writes to ``<run>/analysis/combiners/``: combiners.csv (pool, combiner, cohort, metrics, differences),
caruana.csv (bag composition) and combiners.png.

Usage: python combiners.py [--run RUN] [--tridoze-run RUN] [--draw 0]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ensemble_size as es  # noqa: E402
import evaluate  # noqa: E402
import insight  # noqa: E402

GAP = 4
EPS = 1e-9
ROUNDS = 50
COMBINERS = ["mean", "logit mean", "geometric", "median", "vote", "kappa weights", "caruana", "temperature",
             "temperature per class"]
REPORT = ["sleep_sensitivity", "wake_specificity", "kappa_sleep_wake", "kappa_4stage", es.WS95,
          "tst_bias_minutes", "tst_mae_minutes", "waso_bias_minutes", "waso_mae_minutes"]


def normalize(p: np.ndarray) -> np.ndarray:
    return p / np.clip(p.sum(axis=-1, keepdims=True), EPS, None)


def combine(P: np.ndarray, how: str, weights: np.ndarray | None = None) -> np.ndarray:
    """``P`` is (members, epochs, 5); returns (epochs, 5)."""
    if how == "mean":
        return P.mean(axis=0)
    if how in ("logit mean", "geometric"):
        z = np.log(np.clip(P, EPS, None)).mean(axis=0)
        return normalize(np.exp(z - z.max(axis=-1, keepdims=True)))
    if how == "median":
        return normalize(np.median(P, axis=0))
    if how == "vote":
        votes = np.eye(5)[P.argmax(axis=-1)].mean(axis=0)
        return votes + 1e-3 * P.mean(axis=0)
    if how == "weighted":
        return np.tensordot(weights / weights.sum(), P, axes=1)
    raise ValueError(how)


def temper(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(p, EPS, None)) / t
    return normalize(np.exp(z - z.max(axis=-1, keepdims=True)))


def fit_temperature(p: np.ndarray, y: np.ndarray, per_class: bool) -> np.ndarray:
    def nll(logt):
        q = temper(p, np.exp(logt) if per_class else np.exp(logt[0]))
        return -np.mean(np.log(np.clip(q[np.arange(len(y)), y], EPS, None)))
    r = minimize(nll, np.zeros(5 if per_class else 1), method="L-BFGS-B", bounds=[(-3, 3)] * (5 if per_class else 1))
    t = np.exp(r.x)
    return t if per_class else np.full(5, t[0])


def best_eval_kappa(log: Path) -> float:
    m = re.findall(r"best eval kappa ([0-9.]+)", log.read_text())
    return float(m[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=evaluate.RESULTS / "ensemble-size-2026-10-17")
    ap.add_argument("--tridoze-run", type=Path, default=evaluate.RUN)
    ap.add_argument("--baseline-run", type=Path, default=insight.BASELINE)
    ap.add_argument("--labels-root", type=Path, default=evaluate.LABELS)
    ap.add_argument("--draw", type=int, default=0)
    args = ap.parse_args()
    d = args.draw
    name = f"draw{d}"
    out = args.run / "analysis/combiners"
    out.mkdir(parents=True, exist_ok=True)

    logger = evaluate.MetricsLogger(evaluate.load_metrics_config(evaluate.BIDOZE / "config/metrics.json")[0])
    bidsleep = evaluate.night_counts([s.path for s in evaluate.discover_cached_subjects(
        ["bidsleep"], args.labels_root / evaluate.CACHE_NAME)["bidsleep"]])
    draws = json.loads((args.baseline_run / "setup/draws.json").read_text())
    cv = json.loads((args.baseline_run / f"setup/cv/{name}.json").read_text())
    cache = args.baseline_run / f"setup/cache/{name}" / evaluate.CACHE_NAME
    train_paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
    divisor = np.asarray(evaluate.adult_stage_divisor(evaluate.night_counts(train_paths), bidsleep, evaluate.BETA))

    # members: label -> (prediction dir, train_predictions npz, training.log)
    members = {}
    for fam, dirs in es.family_members(args.run, args.tridoze_run, d, 7).items():
        for k, p in dirs.items():
            arm_dir = p.parents[2]
            npz = arm_dir / f"draw{d}/train_predictions/draw{d}_seed_{k}.npz"
            log = arm_dir / f"draw{d}/training.log"
            ok = all((p / f"{ds}_{sid}.csv").exists() for ds in insight.COHORTS for sid in draws[name][ds])
            if ok and npz.exists() and log.exists():
                members[(fam, k)] = (p, npz, log)
            else:
                print(f"skip {fam} member {k}: predictions, train_predictions or log missing")

    test = {}  # (member) -> {ds: {sid: (y, p5)}}
    for key, (p, _, _) in members.items():
        test[key] = {ds: {sid: evaluate.load_subject(p / f"{ds}_{sid}.csv") for sid in draws[name][ds]} for ds in insight.COHORTS}
    evals, kappas = {}, {}
    for key, (_, npz, log) in members.items():
        z = np.load(npz)
        evals[key] = {k: z[k] for k in z["__eval__"]}
        if key == next(iter(members)):
            labels = {k: z[f"__labels__/{k}"].astype(int) for k in z["__eval__"]}
        else:
            assert set(evals[key]) == set(labels), "eval shares differ between members"
        kappas[key] = best_eval_kappa(log)
    recs = sorted(labels)

    def eval_P(keys):
        return {r: np.stack([evals[k][r] for k in keys]) for r in recs}

    def eval_kappa(p_by_rec: dict[str, np.ndarray]) -> float:
        vals = []
        for r in recs:
            p5 = p_by_rec[r]
            vals.append(evaluate.subject_metrics(logger, labels[r], p5, evaluate.hybrid_calls(p5, p5, divisor))["kappa_4stage"])
        return float(np.nanmean(vals))

    def test_metrics(p_for_subject) -> dict[str, dict[str, float]]:
        subjects = {ds: {sid: (test[next(iter(test))][ds][sid][0], p_for_subject(ds, sid)) for sid in draws[name][ds]}
                    for ds in insight.COHORTS}
        return es.score_subjects(logger, divisor, subjects)

    pools = {"independent": [k for k in members if k[0] == "independent"],
             "bagged": [k for k in members if k[0] == "bagged"],
             "union": list(members)}
    rows, caruana_rows = [], []

    def stack(keys, ds, sid):
        return np.stack([test[k][ds][sid][1] for k in keys])

    def add(pool, combiner, metrics, note=""):
        for cohort, m in metrics.items():
            rows.append({"pool": pool, "members": len(pools[pool]), "combiner": combiner, "cohort": cohort, "note": note,
                         **{k: m[k] for k in REPORT}})

    # reference: BiDoze-G x3, seeds 0-2 of the independent family
    ref_keys = [("independent", k) for k in range(3)]
    if all(k in members for k in ref_keys):
        pools["BiDoze-G ×3"] = ref_keys
        add("BiDoze-G ×3", "mean", test_metrics(lambda ds, sid: stack(ref_keys, ds, sid).mean(axis=0)))

    for pool, keys in pools.items():
        if pool == "BiDoze-G ×3" or len(keys) < 2:
            continue
        for how in ["mean", "logit mean", "geometric", "median", "vote"]:
            add(pool, how, test_metrics(lambda ds, sid, how=how: combine(stack(keys, ds, sid), how)))
        w = np.array([kappas[k] for k in keys])
        add(pool, "kappa weights", test_metrics(lambda ds, sid: combine(stack(keys, ds, sid), "weighted", w)),
            "weights " + " ".join(f"{k[0][:3]}{k[1]}:{v:.3f}" for k, v in zip(keys, w)))

        # Caruana: greedy selection with replacement on the eval share
        ev = {k: {r: evals[k][r] for r in recs} for k in keys}
        counts, history = np.zeros(len(keys), int), []
        total = {r: np.zeros_like(ev[keys[0]][r]) for r in recs}
        for rnd in range(ROUNDS):
            scores = []
            for i, k in enumerate(keys):
                n = counts.sum() + 1
                scores.append(eval_kappa({r: (total[r] + ev[k][r]) / n for r in recs}))
            best = int(np.argmax(scores))
            counts[best] += 1
            for r in recs:
                total[r] += ev[keys[best]][r]
            history.append((scores[best], counts.copy()))
        pick = int(np.argmax([h[0] for h in history]))
        chosen = history[pick][1]
        for i, k in enumerate(keys):
            caruana_rows.append({"pool": pool, "member": f"{k[0]} {k[1]}", "count": int(chosen[i]),
                                 "bag_size": int(chosen.sum()), "round": pick + 1, "eval_kappa_4stage": history[pick][0],
                                 "eval_kappa_4stage_round50": history[-1][0]})
        wts = chosen.astype(float)
        add(pool, "caruana", test_metrics(lambda ds, sid: combine(stack(keys, ds, sid), "weighted", wts)),
            f"bag of {int(chosen.sum())} from round {pick + 1}")

        # temperature per member, fitted on the eval share by NLL
        y_all = np.concatenate([labels[r][labels[r] >= 0] for r in recs])
        for per_class, label in ((False, "temperature"), (True, "temperature per class")):
            temps = {}
            for k in keys:
                p_all = np.concatenate([evals[k][r][labels[r] >= 0] for r in recs])
                temps[k] = fit_temperature(p_all, y_all, per_class)
            add(pool, label, test_metrics(lambda ds, sid: np.mean(
                [temper(test[k][ds][sid][1], temps[k]) for k in keys], axis=0)),
                "T " + " ".join(f"{k[0][:3]}{k[1]}:" + ("/".join(f"{x:.2f}" for x in temps[k]) if per_class else f"{temps[k][0]:.2f}")
                                for k in keys))
        print(f"{pool}: done ({len(keys)} members)")

    table = pd.DataFrame(rows)
    ref = table[table.pool == "BiDoze-G ×3"].set_index("cohort")
    for metric in ("kappa_4stage", "kappa_sleep_wake", es.WS95):
        table[f"{metric}_vs_g3"] = [r[metric] - ref.loc[r.cohort, metric] if len(ref) else np.nan for _, r in table.iterrows()]
        base = table[table.combiner == "mean"].set_index(["pool", "cohort"])[metric]
        table[f"{metric}_vs_mean"] = [r[metric] - base.get((r.pool, r.cohort), np.nan) for _, r in table.iterrows()]
    table.to_csv(out / "combiners.csv", index=False)
    pd.DataFrame(caruana_rows).to_csv(out / "caruana.csv", index=False)
    figure(table, out / "combiners.png")
    print(f"wrote {out}")


def figure(table: pd.DataFrame, path: Path) -> None:
    cohorts = list(insight.COHORTS.values())
    metrics = [("kappa_4stage", "κ 4-stage"), ("kappa_sleep_wake", "κ sleep/wake")]
    pools = [p for p in ("independent", "bagged", "union") if p in set(table.pool)]
    fig, axes = plt.subplots(len(cohorts), len(metrics), figsize=(10, 7), squeeze=False)
    colors = {"independent": "#4c72b0", "bagged": "#dd8452", "union": "#55a868"}
    methods = [c for c in COMBINERS if c != "geometric"]
    for r, cohort in enumerate(cohorts):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            for i, pool in enumerate(pools):
                g = table[(table.pool == pool) & (table.cohort == cohort)].drop_duplicates("combiner").set_index("combiner")
                ax.bar(np.arange(len(methods)) + (i - 1) * 0.27, [g[metric].get(m, np.nan) for m in methods], 0.27,
                       color=colors[pool], label=pool)
            ref = table[(table.pool == "BiDoze-G ×3") & (table.cohort == cohort)]
            if len(ref):
                ax.axhline(ref[metric].iloc[0], color="black", ls="--", lw=1, label="BiDoze-G ×3")
            vals = table[(table.cohort == cohort)][metric]
            ax.set_ylim(vals.min() - 0.03, vals.max() + 0.02)
            ax.set_xticks(range(len(methods)), methods, rotation=40, ha="right", fontsize=8)
            ax.set_title(f"{cohort}: {title}", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

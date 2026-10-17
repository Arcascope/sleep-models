"""How BiDoze-G ensembles scale with the number of members.

Two families of 7 members each, scored on the draw's held-out adults (``testSleepAccel``, ``testClinical``):
- independent: run seeds 0–6 (seeds 0–2 in the tridoze run, 3–6 in the ensemble-size run);
- bagged: run seeds 0–6, each fit on its own bootstrap resample of the training recordings.
Every K-subset of the members (K = 1, 2, 3, 5, 7) is averaged in 5-way probability and decoded as the
release decodes (``hybrid_calls`` with the draw's adult stage divisor). Seeds {0, 1, 2} are labelled
``BiDoze-G ×3`` and bags {0, 1, 2} ``Bagged G ×3``. A member whose predictions are missing is skipped, and
subsets use only the members present.

The split-batch arms (``split7_1x``, ``split7_3x`` under the run) are single 7-member ensembles: the output is
``draw<d>_seed_0`` and the members are ``draw<d>_seed_0_member<m>``. Each is scored as its own output
(``kind == "ensemble"``) and as K-subsets of its 7 members (``kind == "subset"``, family = the arm name).

Metrics are per-subject values averaged over the cohort: ``kappa_4stage``, ``kappa_sleep_wake``,
``sleep_sensitivity``, ``wake_specificity``, TST and WASO bias (model − PSG, minutes) and absolute error, and
``wake_spec_at_95_sens`` (percent, pooled over the cohort).

Writes to ``<run>/analysis/``:
- subsets.csv       every subset: draw, family, kind, K, label, members, cohort and the metrics;
- curve.csv         per draw, family, kind, K, cohort and metric: mean, SD, min, max over subsets and their number;
- vs_bidoze_g3.csv  difference to ``BiDoze-G ×3`` (seeds 0–2) per draw, and over draws (mean ± 95% t half-width, draws higher);
- ensemble-size.png κ 4-stage and κ sleep/wake against K by cohort.

Usage: python ensemble_size.py [--run RUN] [--tridoze-run RUN] [--draws 0] [--members 7]
"""
from __future__ import annotations

import argparse
import itertools
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

GAP = 4
KS = [1, 2, 3, 5, 7]
WS95 = evaluate.WAKE_SPEC_95
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity",
           "tst_bias_minutes", "tst_mae_minutes", "waso_bias_minutes", "waso_mae_minutes", WS95]
VS = ["kappa_4stage", "kappa_sleep_wake", WS95]
REFERENCE = "BiDoze-G ×3"
SPECIAL = {"independent": (0, 1, 2), "bagged": (0, 1, 2)}
LABEL = {"independent": REFERENCE, "bagged": "Bagged G ×3"}
SPLIT_ARMS = {"split7_1x": "1× params", "split7_3x": "3× params"}


def pred_dir(arm_dir: Path, d: int, name: str) -> Path:
    return arm_dir / f"draw{d}" / "predictions" / f"draw{d}_{name}"


def family_members(run: Path, tri: Path, d: int, n: int) -> dict[str, dict[int, Path]]:
    """family -> {member index: prediction dir} for the first ``n`` members."""
    seeds = {k: tri / ("bidoze_g" if k == 0 else f"bidoze_g_s{k}") if k < 3 else run / f"bidoze_g_s{k}" for k in range(n)}
    bags = {k: (tri if k < 3 else run) / f"bidoze_g_bag{k}" for k in range(n)}
    return {"independent": {k: pred_dir(a, d, f"seed_{k}") for k, a in seeds.items()},
            "bagged": {k: pred_dir(a, d, f"seed_{k}") for k, a in bags.items()}}


def half_width(x) -> float:
    x = pd.Series(x).dropna()
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=evaluate.RESULTS / "ensemble-size-2026-10-17")
    ap.add_argument("--tridoze-run", type=Path, default=evaluate.RUN)
    ap.add_argument("--baseline-run", type=Path, default=insight.BASELINE)
    ap.add_argument("--labels-root", type=Path, default=evaluate.LABELS)
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--members", type=int, default=7, help="use members 0..N-1 of each family")
    args = ap.parse_args()
    out = args.run / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    logger = evaluate.MetricsLogger(evaluate.load_metrics_config(evaluate.BIDOZE / "config/metrics.json")[0])
    bidsleep = evaluate.night_counts([s.path for s in evaluate.discover_cached_subjects(
        ["bidsleep"], args.labels_root / evaluate.CACHE_NAME)["bidsleep"]])
    draws = json.loads((args.baseline_run / "setup/draws.json").read_text())

    rows = []
    for d in args.draws:
        name = f"draw{d}"
        cv = json.loads((args.baseline_run / f"setup/cv/{name}.json").read_text())
        cache = args.baseline_run / f"setup/cache/{name}" / evaluate.CACHE_NAME
        train_paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
        divisor = np.asarray(evaluate.adult_stage_divisor(evaluate.night_counts(train_paths), bidsleep, evaluate.BETA))

        def complete(p: Path) -> bool:
            return all((p / f"{ds}_{sid}.csv").exists() for ds in insight.COHORTS for sid in draws[name][ds])

        groups = {}  # family -> (kind, members {index: dir}, ensemble dir or None)
        for fam, members in family_members(args.run, args.tridoze_run, d, args.members).items():
            have = {k: p for k, p in members.items() if complete(p)}
            for k in sorted(set(members) - set(have)):
                print(f"skip draw {d} {fam} member {k}: predictions missing")
            groups[fam] = (have, None)
        for arm in SPLIT_ARMS:
            arm_dir = args.run / arm
            members = {m: pred_dir(arm_dir, d, f"seed_0_member{m}") for m in range(7)}
            have = {m: p for m, p in members.items() if complete(p)}
            ens = pred_dir(arm_dir, d, "seed_0")
            if not complete(ens):
                print(f"skip draw {d} {arm} ensemble output: predictions missing")
                ens = None
            if len(have) < 7:
                print(f"skip draw {d} {arm}: members {sorted(set(members) - set(have))} missing")
            if have or ens:
                groups[arm] = (have, ens)

        # every member's per-subject probabilities, read once
        loaded = {}
        def get(p: Path, ds: str, sid: str):
            if (p, ds, sid) not in loaded:
                loaded[(p, ds, sid)] = evaluate.load_subject(p / f"{ds}_{sid}.csv")
            return loaded[(p, ds, sid)]

        def score(dirs: list[Path]) -> dict[str, dict[str, float]]:
            result = {}
            for ds, cohort in insight.COHORTS.items():
                per, subjects = [], {}
                for sid in draws[name][ds]:
                    ls = [get(p, ds, sid) for p in dirs]
                    y, p5 = ls[0][0], np.mean([q for _, q in ls], axis=0)
                    calls = evaluate.hybrid_calls(p5, p5, divisor)
                    m = evaluate.subject_metrics(logger, y, p5, calls)
                    scored = (y >= 0) & (y < GAP)
                    mins = evaluate.sleep_minutes(y[scored], calls[scored])
                    tst = mins["pred_tst_minutes"] - mins["psg_tst_minutes"]
                    waso = mins["pred_waso_minutes"] - mins["psg_waso_minutes"]
                    per.append({**{k: m[k] for k in METRICS[:4]}, "tst_bias_minutes": tst, "tst_mae_minutes": abs(tst),
                                "waso_bias_minutes": waso, "waso_mae_minutes": abs(waso)})
                    subjects[sid] = (y, p5)
                mean = pd.DataFrame(per).mean().to_dict()
                mean[WS95] = 100 * evaluate.wake_specificity_at(subjects, 0.95)
                result[cohort] = mean
            return result

        for fam, (have, ens) in groups.items():
            idx = sorted(have)
            for k in KS:
                for combo in itertools.combinations(idx, k):
                    label = LABEL[fam] if fam in LABEL and combo == SPECIAL[fam] else ""
                    for cohort, m in score([have[i] for i in combo]).items():
                        rows.append({"draw": d, "family": fam, "kind": "subset", "K": k, "label": label,
                                     "members": " ".join(map(str, combo)), "cohort": cohort, **m})
            if ens is not None:
                for cohort, m in score([ens]).items():
                    rows.append({"draw": d, "family": fam, "kind": "ensemble", "K": 7, "label": SPLIT_ARMS[fam],
                                 "members": "output", "cohort": cohort, **m})
        print(f"draw {d}: scored {sum(r['draw'] == d for r in rows)} rows")

    if not rows:
        print("no predictions found; nothing to write")
        return
    subsets = pd.DataFrame(rows)
    subsets.to_csv(out / "subsets.csv", index=False)

    long = subsets.melt(id_vars=["draw", "family", "kind", "K", "label", "members", "cohort"],
                        value_vars=METRICS, var_name="metric", value_name="value")
    keys = ["draw", "family", "kind", "K", "cohort", "metric"]
    curve = long.groupby(keys, sort=False)["value"].agg(mean="mean", sd=lambda x: x.std(ddof=1) if len(x) > 1 else np.nan,
                                                       min="min", max="max", subsets="count").reset_index()
    marked = long[long.label != ""].rename(columns={"value": "mean"})
    marked = marked.assign(family=marked.label, sd=np.nan, min=marked["mean"], max=marked["mean"], subsets=1)
    pd.concat([curve, marked[[*keys, "mean", "sd", "min", "max", "subsets"]]]).to_csv(out / "curve.csv", index=False)

    # difference to BiDoze-G ×3
    ref = long[(long.label == REFERENCE)].set_index(["draw", "cohort", "metric"])["value"]
    models = []
    for fam in ("independent", "bagged"):
        pretty = {"independent": "Independent", "bagged": "Bagged"}[fam]
        for k, combo in ((5, "0 1 2 3 4"), (7, "0 1 2 3 4 5 6")):
            g = long[(long.family == fam) & (long.kind == "subset") & (long.K == k)]
            models.append((f"{pretty} K={k}, mean over subsets", g.groupby(["draw", "cohort", "metric"])["value"].mean()))
            models.append((f"{pretty} K={k}, members {combo}", g[g.members == combo].set_index(["draw", "cohort", "metric"])["value"]))
    for arm in SPLIT_ARMS:
        g = long[(long.family == arm) & (long.kind == "ensemble")]
        models.append((f"{arm} output", g.set_index(["draw", "cohort", "metric"])["value"]))
    vs = []
    for model, s in models:
        for (d, cohort, metric), v in s.items():
            if metric in VS and (d, cohort, metric) in ref.index:
                vs.append({"model": model, "draw": d, "cohort": cohort, "metric": metric, "value": v,
                           "bidoze_g3": ref[(d, cohort, metric)], "diff": v - ref[(d, cohort, metric)]})
    vs = pd.DataFrame(vs, columns=["model", "draw", "cohort", "metric", "value", "bidoze_g3", "diff"])
    over = []
    if vs.draw.nunique() > 1:
        for (model, cohort, metric), g in vs.groupby(["model", "cohort", "metric"], sort=False):
            over.append({"model": model, "draw": "all", "cohort": cohort, "metric": metric, "value": g.value.mean(),
                         "bidoze_g3": g.bidoze_g3.mean(), "diff": g["diff"].mean(), "half_width": half_width(g["diff"]),
                         "draws_higher": int((g["diff"] > 0).sum()), "draws": len(g)})
    pd.concat([vs, pd.DataFrame(over)]).to_csv(out / "vs_bidoze_g3.csv", index=False)

    figure(curve, long, out / "ensemble-size.png")
    print(f"wrote {out}")


def figure(curve: pd.DataFrame, long: pd.DataFrame, path: Path) -> None:
    metrics = [("kappa_4stage", "κ 4-stage"), ("kappa_sleep_wake", "κ sleep/wake")]
    cohorts = list(insight.COHORTS.values())
    fig, axes = plt.subplots(len(cohorts), len(metrics), figsize=(9, 7), squeeze=False)
    style = {"independent": ("#4c72b0", "Independent seeds"), "bagged": ("#dd8452", "Bagged")}
    markers = {"split7_1x": ("s", "#55a868", "Split-batch, 1× params"), "split7_3x": ("D", "#c44e52", "Split-batch, 3× params")}
    handles = {}
    for r, cohort in enumerate(cohorts):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            for fam, (color, text) in style.items():
                g = curve[(curve.family == fam) & (curve.kind == "subset") & (curve.cohort == cohort) & (curve.metric == metric)]
                g = g.groupby("K")[["mean", "min", "max"]].mean()
                if g.empty:
                    continue
                ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.2, lw=0)
                handles[text] = ax.plot(g.index, g["mean"], "-o", color=color, ms=4, label=text)[0]
            for arm, (mk, color, text) in markers.items():
                g = long[(long.family == arm) & (long.kind == "ensemble") & (long.cohort == cohort) & (long.metric == metric)]
                if len(g):
                    handles[text] = ax.plot([7], [g.value.mean()], mk, color=color, ms=8, ls="", label=text)[0]
            g = long[(long.label == REFERENCE) & (long.cohort == cohort) & (long.metric == metric)]
            if len(g):
                handles[REFERENCE] = ax.axhline(g.value.mean(), color="black", ls="--", lw=1, label=REFERENCE)
            ax.set_xticks(KS)
            ax.set_xlabel("Members (K)")
            ax.set_title(f"{cohort}: {title}", fontsize=10)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

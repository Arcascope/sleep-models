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
- splits_draw<d>.csv  the split-batch arms against BiDoze-G ×3: mean output and members alone, parameters, train s, best eval κ;
- ensembles_draw<d>.csv  per ensemble and cohort: total parameters, training seconds, CPU inference seconds (``--time``), metrics;
- ensemble-size.png κ 4-stage, κ sleep/wake and WS95 against total parameters (log x) by cohort.

Usage: python ensemble_size.py [--run RUN] [--tridoze-run RUN] [--draws 0] [--members 7]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
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
SPLIT_ARMS = {"split7_1x": 7, "split7_1x_boot": 7, "split7_1x_half": 7, "split7_3x": 7, "split7_3x_s1": 7}  # arm -> members
SPLIT_SEED = {"split7_3x_s1": 1}  # run seed of an arm (default 0)
FAMILIES = {"independent": "Independent", "bagged": "Bagged", "subagged": "Subagged"}


def split_name(arm: str, member: int | None = None) -> str:
    """Prediction directory name of a split arm's output, or of its member."""
    base = f"seed_{SPLIT_SEED.get(arm, 0)}"
    return base if member is None else f"{base}_member{member}"


def arm_label(arm: str) -> str:
    m = re.match(r"split(\d+)_(\d+)x(?:_(\w+))?", arm)
    if not m:
        return arm
    extra = {None: "", "boot": ", own bootstraps", "half": ", own halves"}.get(m.group(3), f", {m.group(3)}")
    return f"Split-batch, {m.group(1)} members, {m.group(2)}× width{extra}"


def pred_dir(arm_dir: Path, d: int, name: str) -> Path:
    return arm_dir / f"draw{d}" / "predictions" / f"draw{d}_{name}"


def family_members(run: Path, tri: Path, d: int, n: int) -> dict[str, dict[int, Path]]:
    """family -> {member index: prediction dir} for the first ``n`` members."""
    seeds = {k: tri / ("bidoze_g" if k == 0 else f"bidoze_g_s{k}") if k < 3 else run / f"bidoze_g_s{k}" for k in range(n)}
    bags = {k: (tri if k < 3 else run) / f"bidoze_g_bag{k}" for k in range(n)}
    return {"independent": {k: pred_dir(a, d, f"seed_{k}") for k, a in seeds.items()},
            "bagged": {k: pred_dir(a, d, f"seed_{k}") for k, a in bags.items()},
            "subagged": {k: pred_dir(run / f"bidoze_g_sub{k}", d, f"seed_{k}") for k in range(n)}}


def half_width(x) -> float:
    x = pd.Series(x).dropna()
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def score_subjects(logger, divisor: np.ndarray, subjects: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> dict[str, dict[str, float]]:
    """Cohort means of the metrics for ``{test set: {subject id: (labels, 5-way probabilities)}}``, decoded as the release decodes."""
    result = {}
    for ds, cohort in insight.COHORTS.items():
        per = []
        for y, p5 in subjects[ds].values():
            calls = evaluate.hybrid_calls(p5, p5, divisor)
            m = evaluate.subject_metrics(logger, y, p5, calls)
            scored = (y >= 0) & (y < GAP)
            mins = evaluate.sleep_minutes(y[scored], calls[scored])
            tst = mins["pred_tst_minutes"] - mins["psg_tst_minutes"]
            waso = mins["pred_waso_minutes"] - mins["psg_waso_minutes"]
            per.append({**{k: m[k] for k in METRICS[:4]}, "tst_bias_minutes": tst, "tst_mae_minutes": abs(tst),
                        "waso_bias_minutes": waso, "waso_mae_minutes": abs(waso)})
        mean = pd.DataFrame(per).mean().to_dict()
        mean[WS95] = 100 * evaluate.wake_specificity_at(subjects[ds], 0.95)
        result[cohort] = mean
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=evaluate.RESULTS / "ensemble-size-2026-10-17")
    ap.add_argument("--tridoze-run", type=Path, default=evaluate.RUN)
    ap.add_argument("--baseline-run", type=Path, default=insight.BASELINE)
    ap.add_argument("--labels-root", type=Path, default=evaluate.LABELS)
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--members", type=int, default=7, help="use members 0..N-1 of each family")
    ap.add_argument("--split-arm", action="append", default=[], metavar="NAME:MEMBERS", help="another split-batch arm under --run")
    ap.add_argument("--time", action="store_true", help="measure CPU inference seconds per checkpoint (needs a quiet machine)")
    args = ap.parse_args()
    for spec in args.split_arm:
        SPLIT_ARMS[spec.split(":")[0]] = int(spec.split(":")[1])
        if spec.count(":") > 1:
            SPLIT_SEED[spec.split(":")[0]] = int(spec.split(":")[2])
    for arm in [a for a in SPLIT_ARMS if not (args.run / a / f"draw{args.draws[0]}/predictions").exists()]:
        print(f"skip split arm {arm}: no predictions")
        del SPLIT_ARMS[arm]
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
            members = {m: pred_dir(arm_dir, d, split_name(arm, m)) for m in range(SPLIT_ARMS[arm])}
            have = {m: p for m, p in members.items() if complete(p)}
            ens = pred_dir(arm_dir, d, split_name(arm))
            if not complete(ens):
                print(f"skip draw {d} {arm} ensemble output: predictions missing")
                ens = None
            if len(have) < SPLIT_ARMS[arm]:
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
            subjects = {ds: {} for ds in insight.COHORTS}
            for ds in insight.COHORTS:
                for sid in draws[name][ds]:
                    ls = [get(p, ds, sid) for p in dirs]
                    subjects[ds][sid] = (ls[0][0], np.mean([q for _, q in ls], axis=0))
            return score_subjects(logger, divisor, subjects)

        for fam, (have, ens) in groups.items():
            idx = sorted(have)
            for k in [k for k in KS if k <= len(idx)]:
                for combo in itertools.combinations(idx, k):
                    label = LABEL[fam] if fam in LABEL and combo == SPECIAL[fam] else ""
                    for cohort, m in score([have[i] for i in combo]).items():
                        rows.append({"draw": d, "family": fam, "kind": "subset", "K": k, "label": label,
                                     "members": " ".join(map(str, combo)), "cohort": cohort, **m})
            if ens is not None:
                for cohort, m in score([ens]).items():
                    rows.append({"draw": d, "family": fam, "kind": "ensemble", "K": SPLIT_ARMS[fam], "label": arm_label(fam),
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
    for fam, pretty in FAMILIES.items():
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

    for d in args.draws:
        if all((args.run / arm / f"draw{d}/model_outputs").exists() for arm in SPLIT_ARMS) and (args.run / f"bidoze_g_s3/draw{d}/model_outputs").exists():
            table = split_table(subsets, args.run, d)
            table.to_csv(out / f"splits_draw{d}.csv", index=False)
            infer = inference_seconds(args.run, d) if args.time else None
            ens = ensembles_table(subsets, args.run, args.tridoze_run, d, infer)
            ens.to_csv(out / f"ensembles_draw{d}.csv", index=False)
            if d == args.draws[0]:
                figure(ens, curve, out / "ensemble-size.png", d)
    print(f"wrote {out}")


def parameter_counts(run: Path, d: int) -> dict[str, dict[str, float]]:
    """Parameters per member and in total for BiDoze-G and each split arm, and the total in BiDoze-G models."""
    from arcasleep_bidoze.model import BiDozeNetModel

    def count(path: Path) -> int:
        return int(BiDozeNetModel.loadz(path).parameter_count())

    g = count(run / f"bidoze_g_s3/draw{d}/model_outputs/model_draw{d}_seed_3.pisces")
    out = {"bidoze_g": {"per_member": g, "total": g, "k_equivalent": 1.0}}
    for arm in SPLIT_ARMS:
        per = count(run / arm / f"draw{d}/model_outputs/model_draw{d}_{split_name(arm, 0)}.pisces")
        out[arm] = {"per_member": per, "total": SPLIT_ARMS[arm] * per, "k_equivalent": SPLIT_ARMS[arm] * per / g}
    return out


def training_summary(log: Path) -> dict[str, float]:
    text = log.read_text()
    best = re.findall(r"best eval kappa ([0-9.]+) at epoch (\d+)", text)[-1]
    return {"best_eval_kappa": float(best[0]), "best_epoch": int(best[1]),
            "train_seconds": float(re.findall(r"trained in ([0-9.]+)s", text)[-1])}


def split_table(subsets: pd.DataFrame, run: Path, d: int) -> pd.DataFrame:
    """Each split arm's mean output and its members alone, against BiDoze-G ×3 (draw ``d``)."""
    params = parameter_counts(run, d)
    sub = subsets[subsets.draw == d]
    ref = sub[sub.label == REFERENCE].set_index("cohort")
    rows = []
    for arm in SPLIT_ARMS:
        info = {**params[arm], **training_summary(run / arm / f"draw{d}/training.log")}
        groups = {"mean output": sub[(sub.family == arm) & (sub.kind == "ensemble")]}
        alone = sub[(sub.family == arm) & (sub.kind == "subset") & (sub.K == 1)]
        for cohort, g in alone.groupby("cohort"):
            for stat, f in (("member median", g[METRICS].median()), ("member min", g[METRICS].min()),
                            ("member max", g[METRICS].max())):
                rows.append({"arm": arm, "what": stat, "cohort": cohort, **f.to_dict(), **info})
        for what, g in groups.items():
            for _, r in g.iterrows():
                rows.append({"arm": arm, "what": what, "cohort": r.cohort, **r[METRICS].to_dict(), **info})
    table = pd.DataFrame(rows)
    for m in VS:
        table[f"{m}_vs_g3"] = [r[m] - ref.loc[r.cohort, m] for _, r in table.iterrows()]
    return table


def train_seconds(run: Path, tri: Path, d: int) -> dict[str, float]:
    """Training seconds of each G member (``independent k`` / ``bagged k``) and of each split arm."""
    out = {}
    for fam, dirs in family_members(run, tri, d, 7).items():
        for k, p in dirs.items():
            log = p.parents[2] / f"draw{d}/training.log"
            if log.exists():
                out[f"{fam} {k}"] = training_summary(log)["train_seconds"]
    for arm in SPLIT_ARMS:
        if (run / arm / f"draw{d}/training.log").exists():
            out[arm] = training_summary(run / arm / f"draw{d}/training.log")["train_seconds"]
    return out


def inference_seconds(run: Path, d: int) -> dict[str, float]:
    """CPU seconds to predict one 2 × 3000-epoch batch with one checkpoint (median of 3 after a warm-up)."""
    import time
    from arcasleep_bidoze.model import BiDozeNetModel
    paths = {"bidoze_g": run / f"bidoze_g_s3/draw{d}/model_outputs/model_draw{d}_seed_3.pisces",
             **{arm: run / arm / f"draw{d}/model_outputs/model_draw{d}_{split_name(arm, 0)}.pisces" for arm in SPLIT_ARMS}}
    x = np.zeros((2, 3000, 61, 2), dtype=np.float32)
    out = {}
    for name, path in paths.items():
        model = BiDozeNetModel.loadz(path)
        model.infer(x)
        times = []
        for _ in range(3):
            t = time.time()
            np.asarray(model.infer(x))
            times.append(time.time() - t)
        out[name] = float(np.median(times))
    return out


def ensembles_table(subsets: pd.DataFrame, run: Path, tri: Path, d: int, infer: dict[str, float] | None) -> pd.DataFrame:
    """One row per ensemble and cohort: total parameters, training and inference seconds, κ4, κsw, WS95."""
    params = parameter_counts(run, d)
    train = train_seconds(run, tri, d)
    sub = subsets[subsets.draw == d]
    rows = []
    for fam in FAMILIES:
        for k in KS:
            g = sub[(sub.family == fam) & (sub.kind == "subset") & (sub.K == k)]
            if g.empty:
                continue
            secs = [sum(train.get(f"{fam} {i}", np.nan) for i in map(int, m.split())) for m in g.members.unique()]
            for cohort, gc in g.groupby("cohort"):
                rows.append({"ensemble": f"{fam} K={k}", "family": fam, "K": k, "total_parameters": k * params["bidoze_g"]["total"],
                             "train_seconds": float(np.nanmean(secs)), "cpu_inference_seconds": k * infer["bidoze_g"] if infer else np.nan,
                             "subsets": len(gc), "cohort": cohort,
                             **{m: gc[m].mean() for m in (*VS, "sleep_sensitivity", "wake_specificity")}})
    for arm, n in SPLIT_ARMS.items():
        g = sub[(sub.family == arm) & (sub.kind == "ensemble")]
        for _, r in g.iterrows():
            rows.append({"ensemble": arm, "family": arm, "K": n, "total_parameters": params[arm]["total"],
                         "train_seconds": train.get(arm, np.nan), "cpu_inference_seconds": n * infer[arm] if infer else np.nan,
                         "subsets": 1, "cohort": r.cohort, **{m: r[m] for m in (*VS, "sleep_sensitivity", "wake_specificity")}})
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, curve: pd.DataFrame, path: Path, draw: int = 0) -> None:
    """κ 4-stage, κ sleep/wake and WS95 against total parameters (log x) by cohort. Colors are the first
    three validated categorical slots; the split-batch arms share one hue and differ in marker shape."""
    curve = curve[curve.draw == draw]
    metrics = [("kappa_4stage", "κ 4-stage"), ("kappa_sleep_wake", "κ sleep/wake"), (WS95, "Wake spec. at 95% sens. (%)")]
    cohorts = list(insight.COHORTS.values())
    ink, muted, grid, bg = "#1a1a19", "#6b6b66", "#e4e4e0", "#fcfcfb"
    style = {"independent": ("#2a78d6", "Independent seeds"), "bagged": ("#eb6834", "Bagged"), "subagged": ("#1baf7a", "Subagged (halves)")}
    shapes = ["s", "^", "v", "D", "P", "X", "*", "h"]
    fig, axes = plt.subplots(len(cohorts), len(metrics), figsize=(13, 7.2), squeeze=False, facecolor=bg)
    for r, cohort in enumerate(cohorts):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            ax.set_facecolor(bg)
            for fam, (color, text) in style.items():
                g = curve[(curve.family == fam) & (curve.kind == "subset") & (curve.cohort == cohort) & (curve.metric == metric)]
                g = g.groupby("K")[["mean", "min", "max"]].mean()
                if g.empty:
                    continue
                x = g.index * table[table.family == "independent"].total_parameters.iloc[0] / table[(table.family == "independent")].K.iloc[0]
                ax.fill_between(x, g["min"], g["max"], color=color, alpha=0.18, lw=0)
                ax.plot(x, g["mean"], "-o", color=color, lw=2, ms=6, mec=bg, mew=1.5, label=text)
            for i, arm in enumerate(SPLIT_ARMS):
                t = table[(table.family == arm) & (table.cohort == cohort)]
                if len(t):
                    ax.plot(t.total_parameters, t[metric], shapes[i % len(shapes)], color=ink, ms=9, mec=bg, mew=1.5,
                            ls="", label=arm_label(arm))
            ax.set_xscale("log")
            ticks = [k * table[table.family == "independent"].total_parameters.iloc[0] / table[table.family == "independent"].K.iloc[0] for k in KS]
            ax.set_xticks(ticks, [f"{t / 1e6:.1f}M" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("Total parameters", fontsize=8, color=muted)
            ax.set_title(f"{cohort}: {title}", fontsize=10, color=ink, loc="left")
            ax.grid(axis="y", color=grid, lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=muted, labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()

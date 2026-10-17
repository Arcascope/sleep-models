"""Score the TriDoze pilot arms against independent ×3 with the 2026-10-15 joint-ensemble decoding.

Every model is decoded as the adult release is: its output decides wake or sleep; on its sleep
epochs light/deep/REM are divided by the adult stage divisor of the draw's training nights
(BIDsleep reference). Test subjects are the draw's held-out adults (SleepAccel `testSleepAccel`,
SleepAccel-Clinical `testClinical`), reported by cohort. The decoding, cohorts, metrics and
statistics are those of ``joint_ensemble/evaluate.py``, which this imports.

Models per draw:
- ``independent ×3`` and ``independent seed s`` from the baseline run (2026-10-15, not retrained);
- each arm's own output (``predictions/draw<d>_seed_0``);
- each arm's members by themselves, where ``predictions/draw<d>_seed_0_member<m>`` exist;
- each ``--ensemble label=arm:seed,arm:seed,...``: the mean output of those arms' fits
  (``<arm>/draw<d>/predictions/draw<d>_seed_<seed>``), and each of them by itself as a member;
- ``--draw-dir arm:draw=directory`` takes one draw of an arm from another arm directory;
- ``<arm> W branch`` for the arms named in ``--w-branch``: the sleep/wake model that is the
  combined output's p(wake) alone (wake when p(wake) ≥ 0.5; sleep score 1 − p(wake)).

An arm or draw whose predictions are missing is skipped with a note.

Writes to ``<run>/analysis/``:
- draws.csv    per draw, model, cohort and metric: the mean over the draw's test subjects (wake
               specificity at 95% sleep sensitivity, ``wake_spec_at_95_sens``, is pooled over them);
- subjects.csv per draw, model, cohort and subject;
- summary.csv  per model, cohort and metric: mean and SD over the pilot draws; for each arm and its
               W branch, the difference to independent ×3 per draw (mean, SD, draws higher) and the
               Wilcoxon p over (draw, subject) pairs;
- members.csv  per draw, arm, cohort and member: the member's calls on PSG-scored epochs pooled over
               the cohort's test subjects (share of epochs called wake/light/deep/REM/gap, after the
               stage divisor), κ 4-stage and κ sleep/wake alone, and ``rem_collapse``
               (1 when REM is called on < 0.5% of epochs);
- summary.md   the table of the pilot's means, differences and p values.

Usage: python evaluate.py [--run RUN] [--baseline-run RUN0] [--labels-root ROOT] [--draws 0 1]
                          [--arm name[=arm dir]]... [--w-branch ARM]...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BIDOZE_ROOT = HERE.parents[1]
RESULTS = BIDOZE_ROOT / "results"
RUN = RESULTS / "tridoze-2026-10-16"
BASELINE = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15")
LABELS = BASELINE.parent / "bidoze-adult-devices-0/setup/labels"
DEFAULT_ARMS = ["tridoze_j", "tridoze_r", "bidoze_xl", "bidoze_g"]
REM_COLLAPSE = 0.005

sys.path.insert(0, str(BIDOZE_ROOT / "experiments/adult_devices"))
_spec = importlib.util.spec_from_file_location("joint_ensemble_evaluate", BIDOZE_ROOT / "experiments/joint_ensemble/evaluate.py")
joint = importlib.util.module_from_spec(_spec)
sys.modules["joint_ensemble_evaluate"] = joint
_spec.loader.exec_module(joint)

from arcasleep.data.cache import discover_cached_subjects  # noqa: E402
from arcasleep.training.config import load_metrics_config  # noqa: E402
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls  # noqa: E402
from pisces_lite.metrics.logger import MetricsLogger  # noqa: E402

COHORTS, METRICS, WAKE_SPEC_95, STAGES = joint.COHORTS, joint.METRICS, joint.WAKE_SPEC_95, joint.STAGES
CACHE_NAME, BETA, BIDOZE = joint.CACHE_NAME, joint.BETA, joint.BIDOZE
subject_metrics, load_subject, night_counts, wake_specificity_at = (
    joint.subject_metrics, joint.load_subject, joint.night_counts, joint.wake_specificity_at)
EXTRA = ["rem%"]
SLEEP_WAKE_ONLY = ["kappa_sleep_wake", "sleep_sensitivity", "wake_specificity"]
SHOWN = ["kappa_4stage", "kappa_sleep_wake", WAKE_SPEC_95, "sleep_sensitivity", "deep%", "rem%"]


def prediction_dir(arm_dir: Path, d: int, name: str) -> Path:
    return arm_dir / f"draw{d}" / "predictions" / f"draw{d}_{name}"


def model_dirs(baseline: Path, arms: dict[str, Path], w_branch: list[str], d: int,
               ensembles: dict[str, list[tuple[Path, int]]] | None = None,
               draw_dirs: dict[tuple[str, int], Path] | None = None) -> dict[str, list[Path]]:
    out = {"independent ×3": [prediction_dir(baseline / "independent", d, f"seed_{s}") for s in range(3)]}
    out |= {f"independent seed {s}": [prediction_dir(baseline / "independent", d, f"seed_{s}")] for s in range(3)}
    for label, parts in (ensembles or {}).items():
        dirs = [prediction_dir(arm_dir, d, f"seed_{seed}") for arm_dir, seed in parts]
        out[label] = dirs
        out |= {f"{label} member {i}": [p] for i, p in enumerate(dirs)}
    for arm, arm_dir in arms.items():
        arm_dir = (draw_dirs or {}).get((arm, d), arm_dir)
        out[arm] = [prediction_dir(arm_dir, d, "seed_0")]
        if arm in w_branch:
            out[f"{arm} W branch"] = out[arm]
        parent = prediction_dir(arm_dir, d, "seed_0").parent
        for m in sorted(parent.glob(f"draw{d}_seed_0_member*")):
            out[f"{arm} member {m.name.rsplit('member', 1)[1]}"] = [m]
    return out


def w_branch_subject(y: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sleep/wake model of p(wake): ``(y, 5-way stand-in, calls)``; wake when p(wake) ≥ 0.5."""
    w = proba[:, 0] / np.clip(1.0 - proba[:, 4], 1e-9, None)
    stand_in = np.zeros_like(proba)
    stand_in[:, 0], stand_in[:, 1] = w, 1.0 - w
    return y, stand_in, np.where(w >= 0.5, 0, 1)


def members_table(df: pd.DataFrame) -> str:
    cells = df.round(3).astype(str)
    return "\n".join(["| " + " | ".join(cells.columns) + " |", "|" + ":---|" * len(cells.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells.to_numpy()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--baseline-run", type=Path, default=BASELINE,
                    help="2026-10-15 joint-ensemble run: independent arm, setup/draws.json, cv and caches")
    ap.add_argument("--labels-root", type=Path, default=LABELS, help="root holding 12Hz_log_PSD_aligned/bidsleep_*")
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arm", action="append", help="name or label=directory of an arm (default: the four pilot arms)")
    ap.add_argument("--w-branch", action="append", default=None, help="arms whose W branch is scored (default tridoze_r)")
    ap.add_argument("--out", type=Path, default=None, help="analysis directory (default <run>/analysis)")
    ap.add_argument("--ensemble", action="append", default=[], help="label=arm:seed,arm:seed,... (arms under --run)")
    ap.add_argument("--draw-dir", action="append", default=[], help="arm:draw=directory under --run")
    args = ap.parse_args()
    ensembles = {}
    for e in args.ensemble:
        label, _, parts = e.partition("=")
        ensembles[label] = [(args.run / a, int(s)) for a, s in (p.split(":") for p in parts.split(","))]
    draw_dirs = {}
    for e in args.draw_dir:
        spec, _, directory = e.partition("=")
        arm, d = spec.rsplit(":", 1)
        draw_dirs[(arm, int(d))] = args.run / directory

    arms = {}
    for a in args.arm or DEFAULT_ARMS:
        label, _, path = a.partition("=")
        arms[label] = Path(path).resolve() if path else args.run / label
    w_branch = args.w_branch if args.w_branch is not None else ["tridoze_r"]
    out = args.out or args.run / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    logger = MetricsLogger(config)
    bidsleep = night_counts([s.path for s in discover_cached_subjects(["bidsleep"], args.labels_root / CACHE_NAME)["bidsleep"]])
    draws = json.loads((args.baseline_run / "setup/draws.json").read_text())

    rows, per_draw, member_rows = [], [], []
    for d in args.draws:
        name = f"draw{d}"
        cv = json.loads((args.baseline_run / f"setup/cv/{name}.json").read_text())
        cache = args.baseline_run / f"setup/cache/{name}" / CACHE_NAME
        train_paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
        divisor = adult_stage_divisor(night_counts(train_paths), bidsleep, BETA)
        for model, dirs in model_dirs(args.baseline_run, arms, w_branch, d, ensembles, draw_dirs).items():
            if not all(p.is_dir() and any(p.glob("*.csv")) for p in dirs):
                print(f"skip draw {d} {model}: predictions missing")
                continue
            only_sleep_wake = model.endswith(" W branch")
            for dataset, cohort in COHORTS.items():
                subjects, calls = {}, {}
                for sid in draws[name][dataset]:
                    loaded = [load_subject(p / f"{dataset}_{sid}.csv") for p in dirs]
                    y, proba = loaded[0][0], np.mean([q for _, q in loaded], axis=0)
                    if only_sleep_wake:
                        y, proba, calls[sid] = w_branch_subject(y, proba)
                    else:
                        calls[sid] = hybrid_calls(proba, proba, divisor)
                    subjects[sid] = (y, proba)
                for sid, (y, proba) in subjects.items():
                    m = subject_metrics(logger, y, proba, calls[sid])
                    scored = (y >= 0) & (y < len(STAGES))
                    stage_calls = calls[sid][scored]
                    m["rem%"] = 100 * np.mean(stage_calls == 3) / max(np.mean(stage_calls < 4), 1e-9)
                    if only_sleep_wake:
                        m = {k: (v if k in SLEEP_WAKE_ONLY else np.nan) for k, v in m.items()}
                    rows.append({"draw": d, "model": model, "cohort": cohort, "subject_id": sid, **m})
                per_draw.append({"draw": d, "model": model, "cohort": cohort, "metric": WAKE_SPEC_95,
                                 "mean": 100 * wake_specificity_at(subjects, 0.95), "n": len(subjects)})
                if " member " in model:
                    y = np.concatenate([subjects[s][0] for s in subjects])
                    c = np.concatenate([calls[s] for s in subjects])
                    scored = (y >= 0) & (y < len(STAGES))
                    share = {f"{k}_call_share": float(np.mean(c[scored] == i)) for i, k in enumerate([*STAGES, "gap"])}
                    arm, member = model.split(" member ")
                    member_rows.append({"draw": d, "arm": arm, "member": int(member), "cohort": cohort,
                                        "epochs": int(scored.sum()), **share,
                                        "rem_collapse": int(share["rem_call_share"] < REM_COLLAPSE)})

    if not rows:
        print("no predictions found; nothing to write")
        return
    subjects_df = pd.DataFrame(rows)
    subjects_df.to_csv(out / "subjects.csv", index=False)
    for (d, model, cohort), g in subjects_df.groupby(["draw", "model", "cohort"], sort=False):
        for metric in [*METRICS, *EXTRA]:
            per_draw.append({"draw": d, "model": model, "cohort": cohort, "metric": metric,
                             "mean": g[metric].mean(), "n": len(g)})
    draw_table = pd.DataFrame(per_draw).sort_values(["cohort", "metric", "model", "draw"])
    draw_table.to_csv(out / "draws.csv", index=False)

    if member_rows:
        members = pd.DataFrame(member_rows)
        alone = draw_table[draw_table.metric.isin(["kappa_4stage", "kappa_sleep_wake"])].copy()
        alone = alone[alone.model.str.contains(" member ")]
        alone["arm"] = alone.model.str.split(" member ").str[0]
        alone["member"] = alone.model.str.split(" member ").str[1].astype(int)
        alone = alone.pivot_table(index=["draw", "arm", "member", "cohort"], columns="metric", values="mean").reset_index()
        members = members.merge(alone.rename(columns={"kappa_4stage": "kappa_4stage_alone",
                                                      "kappa_sleep_wake": "kappa_sleep_wake_alone"}),
                                on=["draw", "arm", "member", "cohort"], how="left")
        members.to_csv(out / "members.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "members.csv", index=False)

    base_name = "independent ×3"
    summary = []
    for (model, cohort, metric), g in draw_table.groupby(["model", "cohort", "metric"], sort=False):
        row = {"model": model, "cohort": cohort, "metric": metric, "draws": len(g),
               "mean": g["mean"].mean(), "sd": g["mean"].std(ddof=1) if len(g) > 1 else np.nan}
        base = draw_table[(draw_table.model == base_name) & (draw_table.cohort == cohort) & (draw_table.metric == metric)]
        diffs = (g.set_index("draw")["mean"] - base.set_index("draw")["mean"]).dropna()
        if model != base_name and len(diffs) > 0:
            row |= {"vs": base_name, "diff": diffs.mean(),
                    "diff_sd": diffs.std(ddof=1) if len(diffs) > 1 else np.nan, "draws_higher": int((diffs > 0).sum())}
            if metric in [*METRICS, *EXTRA]:
                a = subjects_df[(subjects_df.model == model) & (subjects_df.cohort == cohort)].set_index(["draw", "subject_id"])[metric]
                b = subjects_df[(subjects_df.model == base_name) & (subjects_df.cohort == cohort)].set_index(["draw", "subject_id"])[metric]
                paired = (a - b.reindex(a.index)).dropna()
                row |= {"subject_pairs": len(paired),
                        "p_subjects": float(stats.wilcoxon(paired).pvalue) if (paired != 0).any() else 1.0}
            for d, v in diffs.items():
                row[f"diff_draw{d}"] = v
        summary.append(row)
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)

    lines = [f"Pilot draws {', '.join(map(str, args.draws))}; mean over draws of the per-draw cohort mean "
             f"(wake specificity at 95% sensitivity pooled per draw); Δ = model − independent ×3; "
             f"p: Wilcoxon over (draw, subject) pairs.", ""]
    for cohort in COHORTS.values():
        lines += [f"### {cohort}", "", "| Model | " + " | ".join(f"{m} (Δ, p)" for m in SHOWN) + " |",
                  "|:---|" + "--:|" * len(SHOWN)]
        for model in table.model.unique():
            cells = []
            for metric in SHOWN:
                r = table[(table.model == model) & (table.cohort == cohort) & (table.metric == metric)]
                if r.empty or not np.isfinite(r["mean"].iloc[0]):
                    cells.append("")
                    continue
                r = r.iloc[0]
                fine = "kappa" in metric or metric == "sleep_sensitivity"
                text = f"{r['mean']:.3f}" if fine else f"{r['mean']:.1f}"
                if "diff" in r and np.isfinite(r["diff"]):
                    fmt = f"{r['diff']:+.3f}" if fine else f"{r['diff']:+.1f}"
                    p = r.get("p_subjects", np.nan)
                    text += f" ({fmt.replace('-', '−')}" + (f", p = {p:.3f}" if np.isfinite(p) else "") + ")"
                cells.append(text)
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")
    if member_rows:
        lines += ["### Members by themselves (share of PSG-scored epochs called, pooled per draw)", "",
                  members_table(members.drop(columns=["epochs"])), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

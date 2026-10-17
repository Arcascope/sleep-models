"""Why a split-batch arm's SleepAccel κ 4-stage is where it is: members, training trajectory, confusion, subjects, diversity.

For ``--arms`` (default split7_1x split7_3x) against BiDoze-G ×3 (seeds 0–2) and the 7 independent G members, draw ``--draw``:
1. members.csv      per member, cohort: κ 4-stage (release decoder) and κ sleep/wake;
2. trajectory.csv   per arm: steps per epoch, steps and seconds at the best epoch and at the stop, eval κ at fixed steps;
3. confusion.csv    pooled test confusion (release decoder; true stage × called stage incl. gap), recall and shares;
4. subjects.csv     per test subject κ 4-stage of each arm and G ×3 (seeds 0–2), and the difference;
5. diversity.csv    mean pairwise disagreement of the members' raw argmax over PSG-scored test epochs, per cohort.

Writes to ``<run>/analysis/split-diagnosis/``.
Usage: python split_diagnosis.py [--run RUN] [--draw 0] [--arms split7_1x split7_3x]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ensemble_size as es  # noqa: E402
import evaluate  # noqa: E402
import insight  # noqa: E402

STAGES = ["wake", "light", "deep", "rem", "gap"]


def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"\s+epoch\s+(\d+)\s+step\s+(\d+)\s+loss\s+(\S+)\s+eval_kappa\s+(\S+)\s+smoothed\s+(\S+)\s+best\s+(\S+)@(\d+)\s+lr\s+(\S+)\s+epoch_time=([0-9.]+)s", line)
        if m:
            e, s, loss, k, sm, best, be, lr, t = m.groups()
            rows.append({"epoch": int(e), "step": int(s), "eval_kappa": float(k), "smoothed": float(sm), "best": float(best),
                         "best_epoch": int(be), "lr": float(lr), "epoch_time": float(t)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=evaluate.RESULTS / "ensemble-size-2026-10-17")
    ap.add_argument("--tridoze-run", type=Path, default=evaluate.RUN)
    ap.add_argument("--baseline-run", type=Path, default=insight.BASELINE)
    ap.add_argument("--labels-root", type=Path, default=evaluate.LABELS)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["split7_1x", "split7_3x"])
    args = ap.parse_args()
    d, name = args.draw, f"draw{args.draw}"
    out = args.run / "analysis/split-diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    logger = evaluate.MetricsLogger(evaluate.load_metrics_config(evaluate.BIDOZE / "config/metrics.json")[0])
    bidsleep = evaluate.night_counts([s.path for s in evaluate.discover_cached_subjects(
        ["bidsleep"], args.labels_root / evaluate.CACHE_NAME)["bidsleep"]])
    draws = json.loads((args.baseline_run / "setup/draws.json").read_text())[name]
    cv = json.loads((args.baseline_run / f"setup/cv/{name}.json").read_text())
    cache = args.baseline_run / f"setup/cache/{name}" / evaluate.CACHE_NAME
    train_paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
    divisor = np.asarray(evaluate.adult_stage_divisor(evaluate.night_counts(train_paths), bidsleep, evaluate.BETA))

    ind = es.family_members(args.run, args.tridoze_run, d, 7)["independent"]
    models = {f"independent seed {k}": [p] for k, p in ind.items()}
    models["G ×3 (seeds 0–2)"] = [ind[k] for k in range(3)]
    models["independent K=7"] = list(ind.values())
    for arm in args.arms:
        members = [es.pred_dir(args.run / arm, d, f"seed_0_member{m}") for m in range(es.SPLIT_ARMS[arm])]
        models[arm] = [es.pred_dir(args.run / arm, d, "seed_0")]
        models |= {f"{arm} member {m}": [p] for m, p in enumerate(members)}

    cache_loaded = {}
    def load(p, ds, sid):
        if (p, ds, sid) not in cache_loaded:
            cache_loaded[(p, ds, sid)] = evaluate.load_subject(p / f"{ds}_{sid}.csv")
        return cache_loaded[(p, ds, sid)]

    def subjects_of(dirs):
        return {ds: {sid: (load(dirs[0], ds, sid)[0], np.mean([load(p, ds, sid)[1] for p in dirs], axis=0)) for sid in draws[ds]}
                for ds in insight.COHORTS}

    # 1. members
    rows = []
    for label, dirs in models.items():
        for cohort, m in es.score_subjects(logger, divisor, subjects_of(dirs)).items():
            rows.append({"model": label, "cohort": cohort, "kappa_4stage": m["kappa_4stage"], "kappa_sleep_wake": m["kappa_sleep_wake"]})
    pd.DataFrame(rows).to_csv(out / "members.csv", index=False)

    # 2. trajectory
    traj = []
    for arm in [*args.arms, "bidoze_g seed 0"]:
        log = (args.run / arm / f"draw{d}/training.log") if arm in args.arms else args.tridoze_run / f"bidoze_g/draw{d}/training.log"
        t = parse_log(log)
        best = t.loc[t.eval_kappa.idxmax()]
        spe = int(t.step.iloc[1] - t.step.iloc[0])
        row = {"arm": arm, "steps_per_epoch": spe, "epochs": int(t.epoch.iloc[-1]), "stop_step": int(t.step.iloc[-1]),
               "best_epoch": int(best.epoch), "best_step": int(best.step), "best_eval_kappa": best.eval_kappa,
               "best_smoothed_kappa": t.smoothed.max(), "median_epoch_seconds": t.epoch_time.iloc[1:].median(),
               "median_step_seconds": t.epoch_time.iloc[1:].median() / spe, "first_epoch_seconds": t.epoch_time.iloc[0],
               "seconds_to_best": t.epoch_time[t.epoch <= best.epoch].sum(), "total_epoch_seconds": t.epoch_time.sum(),
               "mean_eval_kappa_last_100_epochs": t.eval_kappa.iloc[-100:].mean(),
               "sd_eval_kappa_last_100_epochs": t.eval_kappa.iloc[-100:].std(),
               "lr_drop_steps": " ".join(str(int(s)) for s in t.step[t.lr.diff().fillna(0) < 0])}
        for step in (500, 1000, 2000, 3000, 4000, 6000):
            near = t.iloc[(t.step - step).abs().argmin()]
            row[f"smoothed_kappa_at_step_{step}"] = near.smoothed
        traj.append(row)
        t.to_csv(out / f"log_{arm.replace(' ', '_')}.csv", index=False)
    pd.DataFrame(traj).to_csv(out / "trajectory.csv", index=False)

    # 3. confusion, 4. subjects, 5. diversity (SleepAccel and Clinical)
    conf, subj, div = [], [], []
    g3 = models["G ×3 (seeds 0–2)"]
    compare = {"G ×3 (seeds 0–2)": g3, **{arm: models[arm] for arm in args.arms}}
    for label, dirs in compare.items():
        s = subjects_of(dirs)
        for ds, cohort in insight.COHORTS.items():
            cm = np.zeros((4, 5), dtype=int)
            for y, p5 in s[ds].values():
                calls = evaluate.hybrid_calls(p5, p5, divisor)
                keep = (y >= 0) & (y < 4)
                np.add.at(cm, (y[keep], calls[keep]), 1)
            for i, st in enumerate(STAGES[:4]):
                conf.append({"model": label, "cohort": cohort, "true_stage": st, "true_epochs": int(cm[i].sum()),
                             "recall": cm[i, i] / cm[i].sum(), **{f"called_{c}": int(cm[i, j]) for j, c in enumerate(STAGES)}})
            for j, st in enumerate(STAGES):
                conf.append({"model": label, "cohort": cohort, "true_stage": f"(called share) {st}", "true_epochs": int(cm.sum()),
                             "recall": np.nan, "called_share": cm[:, j].sum() / cm.sum()})
    for ds, cohort in insight.COHORTS.items():
        base = subjects_of(g3)[ds]
        for arm in args.arms:
            s = subjects_of(models[arm])[ds]
            for sid in draws[ds]:
                k = lambda yp: evaluate.subject_metrics(logger, yp[0], yp[1], evaluate.hybrid_calls(yp[1], yp[1], divisor))["kappa_4stage"]
                a, b = k(s[sid]), k(base[sid])
                subj.append({"arm": arm, "cohort": cohort, "subject_id": sid, "kappa_4stage_arm": a, "kappa_4stage_g3": b, "diff": a - b})
    for label, keys in {**{arm: [(arm, m) for m in range(es.SPLIT_ARMS[arm])] for arm in args.arms},
                        "independent G (7 seeds)": [("ind", k) for k in range(7)]}.items():
        dirs = ([es.pred_dir(args.run / a, d, f"seed_0_member{m}") for a, m in keys] if label in args.arms
                else [ind[k] for k in range(7)])
        for ds, cohort in insight.COHORTS.items():
            ps = [np.concatenate([load(p, ds, sid)[1][(load(dirs[0], ds, sid)[0] >= 0) & (load(dirs[0], ds, sid)[0] < 4)].argmax(1)
                                  for sid in draws[ds]]) for p in dirs]
            pair = [float(np.mean(a != b)) for a, b in itertools.combinations(ps, 2)]
            div.append({"model": label, "cohort": cohort, "members": len(dirs), "mean_pairwise_disagreement": np.mean(pair),
                        "min": np.min(pair), "max": np.max(pair)})
    pd.DataFrame(conf).to_csv(out / "confusion.csv", index=False)
    pd.DataFrame(subj).to_csv(out / "subjects.csv", index=False)
    pd.DataFrame(div).to_csv(out / "diversity.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

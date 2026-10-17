"""The proposal §4 gates: each model against independent ×3, paired by draw.

Reads evaluate.py's ``draws.csv`` and ``members.csv`` from one or more analysis directories. For each
model, cohort and metric (κ 4-stage, κ sleep/wake, wake specificity at 95% sleep sensitivity in
points) it gives the mean difference to independent ×3 over the draws and its 95% t-interval
(t with draws − 1 degrees of freedom). Verdict per model:
- fail: a mean κ difference is ≤ −0.02 in either cohort, or a member calls REM on < 0.5% of epochs
  (``rem_collapse``) in any draw and cohort;
- pass: every lower bound is ≥ −0.02 (κ) and ≥ −3.0 pts (wake specificity) in both cohorts;
- inconclusive otherwise.

Writes ``gates.csv`` (differences) and ``gates_verdict.csv`` to the first directory and prints both.

Usage: python gates.py <analysis dir> [<analysis dir> ...] [--models M ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {"kappa_4stage": 1.0, "kappa_sleep_wake": 1.0, "wake_spec_at_95_sens": 1.0}
BOUNDS = {"kappa_4stage": -0.02, "kappa_sleep_wake": -0.02, "wake_spec_at_95_sens": -3.0}
BASE = "independent ×3"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    draws = pd.concat([pd.read_csv(d / "draws.csv") for d in args.dirs]).drop_duplicates(["draw", "model", "cohort", "metric"])
    members = pd.concat([pd.read_csv(d / "members.csv") for d in args.dirs if (d / "members.csv").exists()])
    models = args.models or sorted(m for m in draws.model.unique() if m != BASE and "member" not in m)
    rows, verdicts = [], []
    for model in models:
        fail = pass_ = True
        reasons = []
        for cohort in sorted(draws.cohort.unique()):
            for metric, scale in METRICS.items():
                d = draws[(draws.cohort == cohort) & (draws.metric == metric)]
                diff = (d[d.model == model].set_index("draw")["mean"] - d[d.model == BASE].set_index("draw")["mean"]).dropna() * scale
                n = len(diff)
                half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                rows.append({"model": model, "cohort": cohort, "metric": metric, "draws": n, "mean_diff": diff.mean(),
                             "half_width": half, "lower": diff.mean() - half, "draws_higher": int((diff > 0).sum()),
                             "per_draw": " ".join(f"{v:+.3f}" for v in diff)})
                if metric != "wake_spec_at_95_sens" and diff.mean() <= -0.02:
                    reasons.append(f"{cohort} {metric} mean {diff.mean():+.3f}")
                if not diff.mean() - half >= BOUNDS[metric]:
                    pass_ = False
        if not members.empty:
            m = members[members.arm == model] if "arm" in members else members.iloc[0:0]
            collapsed = m[m.rem_collapse == 1]
            if len(collapsed):
                reasons.append("REM collapse: " + ", ".join(f"draw {r.draw} member {r.member} {r.cohort}" for r in collapsed.itertuples()))
        fail = bool(reasons)
        verdicts.append({"model": model, "verdict": "fail" if fail else "pass" if pass_ else "inconclusive",
                         "reasons": "; ".join(reasons)})
    table, verdict = pd.DataFrame(rows), pd.DataFrame(verdicts)
    table.to_csv(args.dirs[0] / "gates.csv", index=False)
    verdict.to_csv(args.dirs[0] / "gates_verdict.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(table.round(3).to_string(index=False))
    print(verdict.to_string(index=False))


if __name__ == "__main__":
    main()

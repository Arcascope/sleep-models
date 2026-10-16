"""GPU cost of the TriDoze full test, from the 2026-10-15 fits and the pilot logs.

Per-step cost and steps to the best epoch come from the 2026-10-15 analysis/fits.csv (independent = 1
network, joint3 = 3 networks). New arms take their per-step cost from the pilot training.log files
under the run root when they exist; until then tridoze_j and tridoze_r use joint3, bidoze_g uses
1 network, and bidoze_xl is TODO (left empty). Steps per fit are those of the matching 2026-10-15 arm.
Writes <run root>/analysis/cost.csv.

Usage: python cost.py <run root> [2026-10-15 run dir]
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logparse import parse_log  # noqa: E402

BASELINE = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15")
DRAWS = 5
# arm -> (fits in the full test, 2026-10-15 arm assumed for steps per fit, 2026-10-15 arm assumed for cost until piloted)
PLAN = {
    "independent": (15, "independent", "independent"),
    "tridoze_j": (5, "joint3", "joint3"),
    "tridoze_r": (5, "joint3", "joint3"),
    "bidoze_xl": (5, "independent", None),
    "bidoze_g": (15, "independent", "independent"),
}


def baseline_stats(base: Path) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader((base / "analysis/fits.csv").open()))
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        r = [x for x in rows if x["arm"] == arm]
        steps = [float(x["steps"]) for x in r]
        secs = [float(x["train_seconds"]) for x in r]
        epochs = [float(x["epochs"]) for x in r]
        best = [float(x["best_epoch"]) / float(x["epochs"]) * float(x["steps"]) for x in r]
        out[arm] = {"n": len(r), "steps": st.mean(steps), "steps_to_best": st.mean(best),
                    "sec_per_step": sum(secs) / sum(steps), "steps_per_epoch": sum(steps) / sum(epochs)}
    return out


def pilot_sec_per_step(run_root: Path, arm: str) -> tuple[float, int] | None:
    secs = steps = n = 0
    for log in run_root.glob(f"{arm}/draw*/training.log"):
        for fit in parse_log(log):
            if fit.train_seconds and fit.total_steps:
                secs, steps, n = secs + fit.train_seconds, steps + fit.total_steps, n + 1
    return (secs / steps, n) if n else None


def main(run_root: Path, base: Path = BASELINE) -> None:
    b = baseline_stats(base)
    rows = []
    for arm, (fits, steps_arm, cost_arm) in PLAN.items():
        pilot = pilot_sec_per_step(run_root, arm) if arm != "independent" else None
        if pilot:
            sps, source = pilot[0], f"pilot logs ({pilot[1]} fits)"
        elif cost_arm:
            sps, source = b[cost_arm]["sec_per_step"], f"assumed as 2026-10-15 {cost_arm}"
        else:
            sps, source = None, "TODO: no estimate until piloted"
        steps = b[steps_arm]["steps"]
        done = arm == "independent"
        hours = None if sps is None else fits * steps * sps / 3600
        rows.append({
            "arm": arm, "fits": fits, "already_run": "yes" if done else "no",
            "steps_per_fit": round(steps), "steps_to_best_per_fit": round(b[steps_arm]["steps_to_best"]),
            "sec_per_step": "" if sps is None else round(sps, 4),
            "gpu_hours": "" if hours is None else round(hours, 2),
            "gpu_hours_still_to_run": "" if hours is None else round(0 if done else hours, 2),
            "source": source,
        })
    total = sum(r["gpu_hours_still_to_run"] for r in rows if r["gpu_hours_still_to_run"] != "")
    rows.append({"arm": "total still to run (arms with an estimate)", "gpu_hours_still_to_run": round(total, 2)})
    out = run_root / "analysis/cost.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) 
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fields, restval="")
        w.writeheader()
        w.writerows(rows)
    for arm, s in b.items():
        print(f"2026-10-15 {arm}: {s['sec_per_step']:.4f} s/step, {s['steps']:.0f} steps/fit, "
              f"{s['steps_to_best']:.0f} steps to best epoch, {s['steps_per_epoch']:.1f} steps/epoch")
    print(out.read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), *(Path(a) for a in sys.argv[2:3]))

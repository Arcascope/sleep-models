"""The 2×2 (jerk low-pass × floor noise in training) from ``arms_score.py``'s ``nights.csv``.

Models are named ``<cell>-seed<s>`` (and ``release-seed<s>``). Per cohort and metric:
- ``cells.csv``: each cell's mean over nights, per seed and averaged over seeds;
- ``paired.csv``: each cell against ``base`` (and ``release`` against ``base``): the mean
  participant-level difference, each participant's metric first averaged over nights and seeds,
  and the Wilcoxon signed-rank p over participants.

Usage: python twobytwo_summary.py <scores dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

METRICS = ["sleep_sensitivity", "wake_specificity", "kappa_sleep_wake", "kappa_4stage"]
REFERENCE = "base"


def main(out: Path) -> None:
    table = pd.read_csv(out / "nights.csv")
    table[["cell", "seed"]] = table.model.str.rsplit("-seed", n=1, expand=True)
    per_seed = table.groupby(["cohort", "cell", "seed"])[METRICS].mean()
    cells = per_seed.groupby(["cohort", "cell"]).agg(["mean", "min", "max"])
    cells.to_csv(out / "cells.csv", float_format="%.4f")
    by_participant = table.groupby(["cohort", "participant", "cell"])[METRICS].mean()
    rows = []
    for cohort, group in by_participant.groupby("cohort"):
        wide = group.droplevel("cohort").unstack("cell")
        for metric in METRICS:
            for cell in wide[metric].columns:
                if cell == REFERENCE:
                    continue
                d = (wide[(metric, cell)] - wide[(metric, REFERENCE)]).dropna()
                p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
                rows.append({"cohort": cohort, "metric": metric, "cell": cell, "against": REFERENCE,
                             "participants": d.size, "mean_difference": d.mean(), "wilcoxon_p": p})
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "twobytwo_paired.csv", index=False, float_format="%.4f")
    print(per_seed.groupby(["cohort", "cell"]).mean().round(3).to_string())
    print(paired.round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))

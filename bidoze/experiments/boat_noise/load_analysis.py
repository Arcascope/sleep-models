"""Does per-epoch noise load go with PSG wake scored as sleep? Boats against land (SleepAccel-Clinical).

Reads ``<load>/epochs_<cohort>.csv.gz`` from ``noise_load.py``. SURF's Apple Watch nights are left
out. Wake specificity is the share of PSG-wake epochs the adult release calls wake.

- ``wake_mix.csv``: per cohort, PSG-wake epochs split by movement (largest ENMO sample in the
  epoch: still < ``STILL_G`` (0.1 g; boat motion alone puts 90% of SURF deep-sleep epochs
  under 0.05 g), else moving): the share that are still, and the specificity on still
  and on moving wake. Separates "more still wake" from "still wake called sleep more often".
- ``line_effect.csv``: per cohort and channel, on still PSG-wake epochs, each participant's
  Spearman ρ between the per-epoch noise measure and P(wake); the median ρ over participants and
  the Wilcoxon p of the ρ against 0. Negative ρ: more noise, lower P(wake).
- ``noise_by_stage.csv``: per cohort and PSG stage, the median per-epoch noise measures, so the
  boats' load can be read against land.

Usage: python load_analysis.py <load dir> <boats.csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STILL_G = 0.1
MEASURES = ["line_frac_mag", "line_frac_jerk", "lines_max_mag", "lines_max_jerk", "hull_db", "level_db"]
COHORTS = ["SleepAccel-Clinical", "SURF", "Somerset"]


def main(load: Path, boats_csv: Path) -> None:
    e = pd.concat([pd.read_csv(load / f"epochs_{c}.csv.gz") for c in COHORTS], ignore_index=True)
    e = e[~e.night.str.contains("AppleWatch")]
    e["participant"] = e.night.str.split("_").str[0]
    boats = pd.read_csv(boats_csv)
    e["group"] = e.participant.map(dict(zip(boats.participant, boats.boat))).fillna(e.cohort)
    e["called_wake"] = (e.call == "wake").astype(float)
    e["still"] = e.enmo_max < STILL_G
    wake = e[e.psg == "wake"]

    rows = []
    for key, g in list(wake.groupby("cohort")) + list(wake[wake.cohort == "SURF"].groupby("group")):
        per = g.groupby("participant").agg(
            still_share=("still", "mean"),
            spec=("called_wake", "mean"),
            spec_still=("called_wake", lambda x: x[g.loc[x.index, "still"]].mean()),
            spec_moving=("called_wake", lambda x: x[~g.loc[x.index, "still"]].mean()))
        rows.append({"group": key, "participants": len(per), "wake_epochs": len(g)}
                    | {c: 100 * per[c].mean() for c in per.columns})
    pd.DataFrame(rows).to_csv(load / "wake_mix.csv", index=False, float_format="%.2f")
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    rows = []
    still = wake[wake.still]
    for cohort, g in still.groupby("cohort"):
        for m in MEASURES:
            rho = []
            for _, p in g.groupby("participant"):
                if len(p) >= 20 and p[m].std() > 0 and p.p_wake.std() > 0:
                    rho.append(stats.spearmanr(p[m], p.p_wake).statistic)
            rho = np.array(rho)
            rows.append({"cohort": cohort, "measure": m, "participants": rho.size,
                         "median_rho": np.median(rho), "q25": np.percentile(rho, 25),
                         "q75": np.percentile(rho, 75), "wilcoxon_p": stats.wilcoxon(rho).pvalue})
    pd.DataFrame(rows).to_csv(load / "line_effect.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    stage = e.groupby(["cohort", "psg"])[MEASURES + ["enmo_max"]].median()
    stage["still_share"] = e.groupby(["cohort", "psg"]).still.mean()
    stage.to_csv(load / "noise_by_stage.csv", float_format="%.3f")
    print(stage.round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

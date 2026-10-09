"""Summarise ``devices.py`` output as §6 of the Oct 9 log does: the per-night 2–6 Hz floor.

Per night, the floor is the 10th percentile, over PSG-sleep epochs, of each epoch's quietest
frame (``level_db_min``); ``near_floor`` is the share of those epochs within 3 dB of it. Groups
are cohorts, with ``@k`` nights (every k-th sample) as their own group. For each ``@k`` night,
the floor change from the same night at its own rate is reported too.

Writes ``<out>/floors.csv`` (one row per night) and prints the group summaries.

Usage: python floor_summary.py <devices.py out>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(out: Path) -> None:
    e = pd.read_csv(out / "epochs.csv.gz")
    e = e[e.psg != "wake"]
    rows = []
    for (cohort, night), g in e.groupby(["cohort", "night"]):
        floor = g.level_db_min.quantile(0.1)
        rows.append({"cohort": cohort, "night": night, "participant": night.split("_")[0],
                     "group": cohort + (" @" + night.split("@")[1] if "@" in night else ""),
                     "sleep_epochs": len(g), "floor_db": floor,
                     "near_floor": float((g.level_db_min <= floor + 3).mean()),
                     "high_floor_db": g.high_db_min.quantile(0.1)})
    f = pd.DataFrame(rows)
    q = pd.read_csv(out / "quantization.csv")
    f = f.merge(q[["cohort", "night", "fs", "q_x"]], on=["cohort", "night"])
    f.to_csv(out / "floors.csv", index=False, float_format="%.3f")
    print(f.groupby("group").agg(nights=("night", "size"), participants=("participant", "nunique"),
                                 fs=("fs", "median"), step_mg=("q_x", lambda s: 1000 * s.median()),
                                 floor_median=("floor_db", "median"), floor_min=("floor_db", "min"),
                                 floor_max=("floor_db", "max"), high_floor=("high_floor_db", "median"),
                                 near_floor=("near_floor", "mean")).round(2).to_string())
    by_participant = f.groupby(["group", "participant"]).floor_db.median().unstack("group")
    print("\nper-participant floor medians, min to max:")
    print(by_participant.agg(["min", "max"]).round(1).to_string())
    k = f[f.night.str.contains("@")].assign(base=lambda d: d.night.str.split("@").str[0])
    paired = k.merge(f[["cohort", "night", "floor_db"]], left_on=["cohort", "base"], right_on=["cohort", "night"],
                     suffixes=("", "_own_rate"))
    d = paired.floor_db - paired.floor_db_own_rate
    print(f"\nevery 3rd sample minus own rate, {len(d)} nights: median {d.median():.2f} dB "
          f"(10th–90th percentile {d.quantile(0.1):.2f} to {d.quantile(0.9):.2f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

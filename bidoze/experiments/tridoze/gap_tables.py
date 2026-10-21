"""Markdown tables of the TriDoze pilot gap checks, read from <run root>/gap_checks/gap_checks.csv.

Table 1: SleepAccel and SleepAccel-Clinical test adults; % of PSG sleep epochs with data called gap
as cached, and the change in % of PSG sleep called sleep after the late-context shift.
Table 2: Foundry Apple-labelled sleep in the windows with a scored cut; % called sleep with the full
24 h and with the cut, % called gap with the full 24 h, and cut − full.
Rows missing from the CSV are printed as "pending" for the model-draw pairs named in --expect.

Usage: python gap_tables.py <run root> [--expect model:draw ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABELS = {"independent_x3": "Independent ×3", "independent_seed0": "Independent seed 0",
          "independent_seed1": "Independent seed 1", "independent_seed2": "Independent seed 2",
          "tridoze_j": "T1 TriDoze-J", "tridoze_j_split": "T1 TriDoze-J, refit not capped",
          "tridoze_r": "T2 TriDoze-R", "bidoze_xl": "T3 BiDoze-XL", "bidoze_g": "Control BiDoze-G",
          "tridoze_j_b025": "T1 β 0.25", "bidoze_g_x3": "BiDoze-G ×3", "bidoze_g_bag_x3": "Bagged G ×3"}
COHORTS = ["SleepAccel", "SleepAccel-Clinical"]


def num(v: float, digits: int = 1, signed: bool = False) -> str:
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    if signed and abs(v) < 0.5 * 10 ** -digits:
        s = f"{0:.{digits}f}"
    return s.replace("-", "−")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--expect", nargs="*", default=[])
    a = ap.parse_args()
    g = pd.read_csv(a.run_root / "gap_checks/gap_checks.csv")
    g["condition"] = g["condition"].str.replace("\n", " ")
    have = set(zip(g.model, g.draw))
    pairs = sorted(have | {(m, int(d)) for m, d in (e.split(":") for e in a.expect)},
                   key=lambda p: (list(LABELS).index(p[0]), p[1]))

    def val(model, draw, cohort, condition, col):
        r = g[(g.model == model) & (g.draw == draw) & (g.cohort == cohort) & (g.condition == condition)]
        return float(r[col].iloc[0])

    print("| Model | Draw | % of PSG sleep with data called gap, as recorded: SleepAccel / Clinical "
          "| Change in % of PSG sleep called sleep after the shift (pts): SleepAccel / Clinical |")
    print("|:---|--:|--:|--:|")
    for m, d in pairs:
        if (m, d) not in have:
            print(f"| {LABELS[m]} | {d} | pending | pending |")
            continue
        gap = " / ".join(num(val(m, d, c, "cached", "gap_on_sleep_with_data_pct"), 2) for c in COHORTS)
        shift = " / ".join(num(val(m, d, c, "shifted − cached", "sleep_called_sleep_pct"), 1, True) for c in COHORTS)
        print(f"| {LABELS[m]} | {d} | {gap} | {shift} |")
    print()
    print("| Model | Draw | % of Apple sleep called sleep, full 24 h | % called sleep, cut "
          "| % called gap, full 24 h | Cut − full, called sleep (pts) |")
    print("|:---|--:|--:|--:|--:|--:|")
    for m, d in pairs:
        if (m, d) not in have:
            print(f"| {LABELS[m]} | {d} | pending | pending | pending | pending |")
            continue
        f = "FoundryApple"
        full = val(m, d, f, "full, scored windows", "sleep_called_sleep_pct")
        cut = val(m, d, f, "cut, scored windows", "sleep_called_sleep_pct")
        gap = val(m, d, f, "full, scored windows", "gap_on_sleep_with_data_pct")
        print(f"| {LABELS[m]} | {d} | {num(full)} | {num(cut)} | {num(gap)} | {num(cut - full, 1, True)} |")


if __name__ == "__main__":
    main()

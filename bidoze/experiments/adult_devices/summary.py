"""One table and one figure across every scored arm of the device ablation.

Reads ``<run>/analysis/runs.csv`` and ``paired.csv`` (from score.py) and writes:
- ``summary.md``:  one row per arm, 3-seed ensembles; * marks p < 0.05 against the reference,
                   † against ``matched`` (paired Wilcoxon over subjects).
- ``devices.png``: one panel per metric; bars are the ensembles, dots the single seeds.

Arms with a stalled fold model are shown with it left out.

Usage: python summary.py <run dir> [figure copy path]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REFERENCE = "weaver-2026-10-06"
RELEASE = "pediatric release"
ORDER = [RELEASE, REFERENCE, "matched", "matched-actigraph", "all", "actigraph", "actigraph-119", "apple-119",
         "fitbit-119", "garmin-119"]
LABELS = {RELEASE: "pediatric\nrelease", REFERENCE: "Weaver\n(Oct 6)", "all": "all\ndevices",
          "actigraph": "ActiGraph\n236", "actigraph-119": "ActiGraph\n119", "apple-119": "Apple\n119",
          "fitbit-119": "Fitbit\n119", "garmin-119": "Garmin\n119", "matched": "matched\n(revised)",
          "matched-actigraph": "matched,\nActiGraph"}
COHORTS = {"SleepAccel": "#2a78d6", "SleepAccel-Clinical": "#eb6834"}
PSG_DEEP = {"SleepAccel": 14.7, "SleepAccel-Clinical": 8.7}
# (column, scoring, title, scale)
METRICS = [
    ("kappa_4stage", "adult", "κ 4-stage, adult scoring", 1),
    ("kappa_4stage", "alone", "κ 4-stage, stage source alone", 1),
    ("kappa_sleep_wake", "alone", "κ sleep/wake, stage source alone", 1),
    ("wake_specificity", "alone", "Wake specificity (%), stage source alone", 100),
    ("auroc", "alone", "Sleep vs wake AUROC, stage source alone", 1),
    ("deep%", "adult", "Deep sleep (% of epochs), adult scoring", 1),
]
INK, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def arm_rows(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per displayed arm, its rows; an arm with a stalled model uses the stalled-out rows."""
    out = {}
    for arm in ORDER:
        name = f"{arm} (stalled out)" if (runs.arm == f"{arm} (stalled out)").any() else arm
        rows = runs[runs.arm == name]
        if len(rows):
            out[arm] = rows
    return out


def value(rows: pd.DataFrame, arm: str, model: str, scoring: str, cohort: str, column: str) -> float:
    if arm == RELEASE:
        model, scoring = "release", "alone"
    sel = rows[(rows.model == model) & (rows.scoring == scoring) & (rows.cohort == cohort)]
    return float(sel[column].iloc[0]) if len(sel) else float("nan")


def significant(paired: pd.DataFrame, arm_name: str, scoring: str, cohort: str, metric: str,
                against: str = REFERENCE) -> bool:
    sel = paired[(paired.a == arm_name) & (paired.b == against) & (paired.kind == "ensemble")
                 & (paired.scoring == scoring) & (paired.cohort == cohort) & (paired.metric == metric)]
    return bool(len(sel)) and float(sel.p.iloc[0]) < 0.05


def table(runs: pd.DataFrame, paired: pd.DataFrame, arms: dict[str, pd.DataFrame]) -> str:
    cols = [("SleepAccel", "kappa_4stage", "adult", "SleepAccel κ 4-stage, adult"),
            ("SleepAccel", "kappa_4stage", "alone", "SleepAccel κ 4-stage, alone"),
            ("SleepAccel", "kappa_sleep_wake", "alone", "SleepAccel κ sleep/wake, alone"),
            ("SleepAccel", "wake_specificity", "alone", "SleepAccel wake spec., alone"),
            ("SleepAccel", "deep%", "adult", "SleepAccel deep %, adult"),
            ("SleepAccel-Clinical", "kappa_4stage", "adult", "Clinical κ 4-stage, adult"),
            ("SleepAccel-Clinical", "kappa_sleep_wake", "alone", "Clinical κ sleep/wake, alone"),
            ("SleepAccel-Clinical", "deep%", "adult", "Clinical deep %, adult")]
    lines = ["| Arm | " + " | ".join(c[3] for c in cols) + " |",
             "|:---|" + "--:|" * len(cols)]
    for arm, rows in arms.items():
        name = rows.arm.iloc[0]
        cells = []
        for cohort, metric, scoring, _ in cols:
            if arm == RELEASE and scoring == "alone" and metric == "kappa_4stage":
                cells.append("")
                continue
            v = value(rows, arm, "ensemble", scoring, cohort, metric)
            if metric == "wake_specificity":
                text = f"{100 * v:.1f}%"
            elif metric == "deep%":
                text = f"{v:.1f}"
            else:
                text = f"{v:.3f}"
            exact = arm == RELEASE or (scoring == "adult" and metric == "kappa_sleep_wake")
            if not exact and arm != REFERENCE and significant(paired, name, scoring, cohort, metric):
                text += "*"
            if not exact and arm not in (REFERENCE, "matched") and significant(
                    paired, name, scoring, cohort, metric, against="matched"):
                text += "†"
            cells.append(text)
        label = LABELS[arm].replace("\n", " ") + (" (stalled model out)" if name.endswith("(stalled out)") else "")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure(arms: dict[str, pd.DataFrame], path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(3, 2, figsize=(18, 12.5), facecolor=SURFACE)
    names = list(arms)
    width = 0.38
    for ax, (column, scoring, title, scale) in zip(axes.ravel(), METRICS):
        ax.set_facecolor(SURFACE)
        top = 0.0
        labels = []
        for j, (cohort, color) in enumerate(COHORTS.items()):
            offset = (j - 0.5) * (width + 0.02)
            for i, arm in enumerate(names):
                rows = arms[arm]
                if arm == RELEASE and scoring == "alone" and column == "kappa_4stage":
                    continue
                v = scale * value(rows, arm, "ensemble", scoring, cohort, column)
                if pd.isna(v):
                    continue
                x = i + offset
                ax.bar(x, v, width, color=color, edgecolor=SURFACE, linewidth=1, zorder=2)
                seeds = [scale * value(rows, arm, f"seed{s}", scoring, cohort, column) for s in range(3)]
                ax.scatter([x] * 3, seeds, s=9, color=INK, zorder=3, linewidths=0)
                label_at = max([v] + [y for y in seeds if not pd.isna(y)])
                labels.append((x, label_at, f"{v:.3f}" if scale == 1 and column != "deep%" else f"{v:.1f}"))
                top = max(top, label_at)
        if column == "deep%":
            for j, (cohort, color) in enumerate(COHORTS.items()):
                ax.axhline(PSG_DEEP[cohort], color=color, lw=1, ls=(0, (4, 3)), zorder=1)
        pad = top * 0.015
        for x, y, text in labels:
            ax.text(x, y + pad, text, ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([LABELS[a] for a in names], fontsize=7.5)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COHORTS.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK, markersize=3))
    handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3))))
    fig.legend(handles, ["SleepAccel (28 subjects)", "SleepAccel-Clinical (32 subjects)",
                         "single seed (bar: 3-seed ensemble)", "PSG deep %, in the cohort's color"],
               loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main(run: Path, copy_to: Path | None) -> None:
    analysis = run / "analysis"
    runs = pd.read_csv(analysis / "runs.csv")
    paired = pd.read_csv(analysis / "paired.csv")
    arms = arm_rows(runs)
    (analysis / "summary.md").write_text(table(runs, paired, arms))
    figure(arms, analysis / "devices.png")
    if copy_to is not None:
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(analysis / "devices.png", copy_to)
    print((analysis / "summary.md").read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None)

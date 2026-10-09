"""Figures and tables of the released models on the boat cohorts, from ``boat_baseline``'s nights.csv.

- ``per_boat.png``: per boat, every night as a point and the mean over nights with its 95%
  bootstrap CI, for 4-stage κ, sleep/wake κ, sleep sensitivity, wake specificity and TST error.
- ``aggregate.png``: All SURF, All Somerset and All boat (SURF and Somerset together): bars are
  means over participants (each participant's nights averaged first) with 95% participant
  bootstrap CIs; TST error is a boxen plot over nights.
- ``aggregate.csv``: those means and CIs, and paired Wilcoxon p over participants between models.

SURF's Apple Watch recordings (the same nights as SURF004's Galaxy Watch) are left out.

Usage: python baseline_figures.py <boat-baseline results dir> <out dir>
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

MODELS = {"bidoze-adult": "BiDoze adult", "bidoze-pediatric": "BiDoze pediatric", "unet-pediatric": "UNet pediatric"}
PALETTE = dict(zip(MODELS.values(), ["#0173b2", "#de8f05", "#029e73"]))
METRICS = {"kappa_4stage": "Cohen's κ, 4 stages", "kappa_sleep_wake": "Cohen's κ, sleep/wake",
           "sleep_sensitivity": "Sleep sensitivity (%)", "wake_specificity": "Wake specificity (%)",
           "tst_error_minutes": "TST error (min)"}
BOOT = 5000


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    means = rng.choice(x, (BOOT, len(x))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(baseline: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nights = pd.read_csv(baseline / "nights.csv")
    nights = nights[nights.cohort.isin(["SURF", "Somerset"])].copy()
    nights["Model"] = nights.model.map(MODELS)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    rng = np.random.default_rng(0)

    # Per boat.
    boats = sorted(nights.loc[nights.cohort == "SURF", "boat"].unique()) + ["Somerset"]
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(15, 3.1 * len(METRICS)), sharex=True)
    for ax, (metric, label) in zip(axes, METRICS.items()):
        sns.stripplot(nights, x="boat", y=metric, hue="Model", order=boats, palette=PALETTE, dodge=True,
                      size=3.5, alpha=0.45, jitter=0.18, ax=ax, legend=False, zorder=1)
        sns.pointplot(nights, x="boat", y=metric, hue="Model", order=boats, palette=PALETTE, dodge=0.55,
                      errorbar=("ci", 95), n_boot=BOOT, seed=0, linestyle="none", markers="D", markersize=5,
                      err_kws={"linewidth": 1.6}, capsize=0.08, ax=ax, legend=ax is axes[0])
        ax.set_ylabel(label)
        ax.set_xlabel("")
        if metric == "tst_error_minutes":
            ax.axhline(0, color="0.3", lw=0.8)
    counts = nights[nights.model == "bidoze-adult"].groupby("boat").size()
    axes[-1].set_xticks(range(len(boats)), [f"{b}\n{counts[b]} nights" for b in boats])
    axes[0].legend(title="", ncols=3, loc="upper right", frameon=True)
    sns.despine(fig, left=True)
    fig.tight_layout()
    fig.savefig(out / "per_boat.png", dpi=130)
    plt.close(fig)

    # Aggregates over participants.
    groups = {"All SURF": nights[nights.cohort == "SURF"], "All Somerset": nights[nights.cohort == "Somerset"],
              "All boat": nights}
    rows, pairs = [], []
    for group, g in groups.items():
        per = g.groupby(["participant", "Model"])[list(METRICS)].mean().reset_index()
        for metric in METRICS:
            wide = per.pivot(index="participant", columns="Model", values=metric)
            for model in MODELS.values():
                lo, hi = boot_ci(wide[model].to_numpy(float), rng)
                rows.append({"group": group, "metric": metric, "Model": model, "participants": len(wide),
                             "mean": wide[model].mean(), "ci_low": lo, "ci_high": hi})
            for a, b in itertools.combinations(MODELS.values(), 2):
                d = (wide[a] - wide[b]).to_numpy(float)
                lo, hi = boot_ci(d, rng)
                pairs.append({"group": group, "metric": metric, "a": a, "b": b, "diff": d.mean(), "ci_low": lo,
                              "ci_high": hi, "p": float(wilcoxon(d).pvalue)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "aggregate.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "aggregate_paired.csv", index=False)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(19, 4.6))
    order = list(groups)
    for ax, (metric, label) in zip(axes, METRICS.items()):
        if metric == "tst_error_minutes":
            long = pd.concat([g.assign(group=name) for name, g in groups.items()])
            sns.boxenplot(long, x="group", y=metric, hue="Model", order=order, palette=PALETTE, ax=ax,
                          legend=False, linewidth=0.6, k_depth="trustworthy")
            ax.axhline(0, color="0.3", lw=0.8)
            ax.set_title(f"{label}, nights", fontsize=11)
        else:
            t = table[table.metric == metric]
            sns.barplot(t, x="group", y="mean", hue="Model", order=order, palette=PALETTE, ax=ax, legend=False,
                        errorbar=None, gap=0.08)
            for patch_group, model in zip(ax.containers, MODELS.values()):
                tm = t[t.Model == model].set_index("group").loc[order]
                xs = [bar.get_x() + bar.get_width() / 2 for bar in patch_group]
                ax.errorbar(xs, tm["mean"], yerr=[tm["mean"] - tm.ci_low, tm.ci_high - tm["mean"]], fmt="none",
                            ecolor="0.2", elinewidth=1.2, capsize=3)
            ax.set_title(f"{label}, participants", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PALETTE.values()]
    fig.legend(handles, PALETTE.keys(), ncols=3, loc="upper center", frameon=False)
    sns.despine(fig, left=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "aggregate.png", dpi=130)
    plt.close(fig)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

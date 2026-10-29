"""BiDoze comparison, tables, figure and report.md from the outputs of :mod:`arcasleep_watchtree.cv`.

    python -m arcasleep_watchtree.report --run results/cv-2026-10-27

Appends ``experiment == "bidoze"`` rows to per_night.csv and summary.csv (replacing earlier ones).
Every mean is over nights unless a column says pooled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from arcasleep_watchtree import cv
from arcasleep_watchtree.definition import CONFIG_DIR, TreeSpec

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
COHORT_LABEL = {
    "wearanize": "Wearanize+",
    "aauwss": "AAUWSS",
    "sleepaccel": "SleepAccel",
    "weaver_garmin": "Weaver Garmin",
    "weaver_fitbit": "Weaver Fitbit",
}


# ------------------------------------------------------------------------------ BiDoze


def bidoze_nights(run: Path, oof: pd.DataFrame, features_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-night kappa of BiDoze and WatchTree on the same epochs; and the paired table."""
    calls = pd.read_parquet(run / "bidoze_epochs.parquet")
    feats = pd.read_parquet(
        features_dir / "features-wearanize_uncropped.parquet", columns=["subject", "epoch", "in_psg", "y4"]
    )
    first = feats[feats.in_psg].groupby("subject").epoch.min().rename("first")
    calls = calls.merge(first, left_on="subject", right_index=True)
    calls["epoch_stream"] = calls["epoch"] + calls["first"]
    stage = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
    calls["y4"] = calls.y_raw.map(stage).fillna(-1).astype(int)
    calls = calls[(calls.y4 >= 0) & (calls.call != 4)]
    ctx = oof[(oof.experiment == "cv_within") & (oof.config == "with_hr") & (oof.cohort == "wearanize")]
    ctx_keys = ctx[["subject", "epoch"]].rename(columns={"epoch": "epoch_stream"}).assign(ctx=True)
    calls = calls.merge(ctx_keys, on=["subject", "epoch_stream"], how="left")
    calls["ctx"] = calls.ctx.fillna(False).astype(bool)

    parts = {
        (experiment, config): oof[
            (oof.experiment == experiment) & (oof.config == config) & (oof.cohort == "wearanize")
        ]
        for experiment in ("cv_within", "loco")
        for config in ("with_hr", "no_hr")
    }
    # Paired metrics use only the epochs that BiDoze and every WatchTree configuration all scored.
    keys = calls.loc[calls.ctx, ["subject", "epoch_stream"]].rename(columns={"epoch_stream": "epoch"})
    for part in parts.values():
        keys = keys.merge(part[["subject", "epoch"]], on=["subject", "epoch"])
    shared = keys.assign(shared=True)
    calls = calls.merge(
        shared.rename(columns={"epoch": "epoch_stream"}), on=["subject", "epoch_stream"], how="left"
    )
    calls["shared"] = calls.shared.fillna(False).astype(bool)
    matched = calls[calls.shared]
    call_y4 = matched.set_index(["subject", "epoch_stream"]).y4.sort_index()
    for part in parts.values():
        part_y4 = part.merge(shared, on=["subject", "epoch"]).set_index(["subject", "epoch"]).y4
        if not part_y4.sort_index().to_numpy().tolist() == call_y4.to_numpy().tolist():
            raise ValueError("BiDoze and WatchTree disagree on the reference stage of shared epochs")

    def night_table(frame: pd.DataFrame, config: str) -> pd.DataFrame:
        table = cv.per_night(frame.assign(pred=frame.call))
        table.insert(0, "config", config)
        return table

    all_span = night_table(calls, "bidoze_all_span")
    ctx_only = night_table(matched, "bidoze_ctx")
    rows = [all_span, ctx_only]
    for (experiment, config), part in parts.items():
        table = cv.per_night(part.merge(shared, on=["subject", "epoch"]).drop(columns="shared"))
        table.insert(0, "config", f"watchtree_{experiment}_{config}")
        rows.append(table)
    nights = pd.concat(rows, ignore_index=True)
    return nights, calls


def paired(nights: pd.DataFrame, a: str, b: str, column: str = "kappa4") -> dict:
    left = nights[nights.config == a].set_index("subject")[column]
    right = nights[nights.config == b].set_index("subject")[column]
    both = pd.concat([left, right], axis=1, keys=["a", "b"]).dropna()
    diff = both.a - both.b
    p = stats.wilcoxon(diff).pvalue if len(diff) > 5 and (diff != 0).any() else float("nan")
    return {
        "a": a, "b": b, "metric": column, "n_nights": len(both),
        "mean_a": both.a.mean(), "mean_b": both.b.mean(), "mean_diff": diff.mean(),
        "median_diff": diff.median(), "wins_a": int((diff > 0).sum()), "wilcoxon_p": p,
    }


# ------------------------------------------------------------------------------ tables


def fmt(x, digits=3):
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def summary_row(s: pd.Series) -> dict:
    return {
        "n nights": int(s.n_nights),
        "epochs": int(s.n_epochs),
        "4-stage κ (mean over nights, 95% CI)": f"{s.kappa4_mean:.3f} ({s.kappa4_lo:.3f}, {s.kappa4_hi:.3f})",
        "sleep/wake κ": fmt(s.kappa_sw_mean),
        "deep % pred / PSG": f"{100 * s.deep_pred_mean:.1f} / {100 * s.deep_psg_mean:.1f}",
        "REM % pred / PSG": f"{100 * s.rem_pred_mean:.1f} / {100 * s.rem_psg_mean:.1f}",
        "TST error, min (mean / mean abs)": f"{s.tst_err_min_mean:+.1f} / {s.tst_abs_err_min_mean:.1f}".replace("-", "−"),
        "pooled 4-stage κ": fmt(s.pooled_kappa4),
    }


def cv_table(summary: pd.DataFrame, experiment: str, configs: list[str]) -> pd.DataFrame:
    rows = []
    for cohort in cv.COHORT_ORDER:
        for config in configs:
            sel = summary[(summary.experiment == experiment) & (summary.cohort == cohort) & (summary.config == config)]
            if len(sel):
                rows.append({"cohort": COHORT_LABEL[cohort], "model": config, **summary_row(sel.iloc[0])})
    return pd.DataFrame(rows)


def quant_table(per_night: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = per_night[(per_night.experiment == "cv_within")]
    for cohort in cv.COHORT_ORDER:
        ref = base[(base.cohort == cohort) & (base.config == "with_hr")].set_index("subject").kappa4
        for config in ("with_hr", "with_hr_int8", "with_hr_bin15", "with_hr_bin15_int8"):
            cur = base[(base.cohort == cohort) & (base.config == config)].set_index("subject").kappa4
            diff = (cur - ref).dropna()
            p = stats.wilcoxon(diff).pvalue if (diff != 0).sum() > 5 else float("nan")
            s = summary[(summary.experiment == "cv_within") & (summary.cohort == cohort) & (summary.config == config)].iloc[0]
            rows.append({
                "cohort": COHORT_LABEL[cohort], "model": config, "n nights": int(s.n_nights),
                "4-stage κ (mean over nights)": fmt(s.kappa4_mean),
                "change vs max_bin 255 float": f"{diff.mean():+.4f}".replace("-", "−") if config != "with_hr" else "",
                "paired Wilcoxon p": fmt(p, 3) if config != "with_hr" else "",
                "pooled 4-stage κ": fmt(s.pooled_kappa4),
            })
    return pd.DataFrame(rows)


def ablation_table(summary: pd.DataFrame, max_rounds: int) -> pd.DataFrame:
    sets = cv.groups()
    rows = []
    for config in ("with_hr", "no_forward", "no_hr", "no_breathing", "windows_le15"):
        s = summary[(summary.experiment == "cv_within") & (summary.cohort == "wearanize") & (summary.config == config)].iloc[0]
        ops = cv.ops_per_epoch(sets[config], max_rounds)
        rows.append({
            "features dropped": {"with_hr": "none", "no_forward": "forward windows and tnb_min", "no_hr": "heart rate (all hr_*)",
                                 "no_breathing": "breathing (all breath_*)",
                                 "windows_le15": "all windows over 15 min, tsb_min, tnb_min, hr_base, hr_rel_*"}[config],
            "n features": ops["n_features"], "n nights": int(s.n_nights),
            "4-stage κ (mean over nights, 95% CI)": f"{s.kappa4_mean:.3f} ({s.kappa4_lo:.3f}, {s.kappa4_hi:.3f})",
            "sleep/wake κ": fmt(s.kappa_sw_mean),
            "feature ops / epoch": f"{ops['feature_ops']:,.0f}",
            "features + trees ops / epoch": f"{ops['total_ops']:,.0f}",
            "CPU %": f"{ops['total_cpu_percent']:.1f}",
            "window ring buffers, bytes": f"{ops['ring_bytes']:,.0f}",
        })
    return pd.DataFrame(rows)


def leak_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    names = {
        "cropped_all": "old: cropped features, all scored epochs",
        "cropped_ctx": "cropped features, complete-context epochs",
        "ctx": "fixed: complete-context epochs (uncropped features for Wearanize+)",
        "fixed_all": "uncropped features, all scored epochs",
    }
    for cohort in cv.COHORT_ORDER:
        for setup in ("cropped_all", "cropped_ctx", "ctx", "fixed_all"):
            r = {"cohort": COHORT_LABEL[cohort], "setup": names[setup]}
            ok = False
            for label, col in (("model", "model κ"), ("probe_nan", "probe 1: missing-value flags κ"), ("probe_coverage", "probe 2: window-coverage κ")):
                sel = summary[(summary.experiment == "leak") & (summary.cohort == cohort) & (summary.config == f"{setup}/{label}")]
                if len(sel):
                    ok = True
                    r["n nights"] = int(sel.iloc[0].n_nights)
                    r["epochs"] = int(sel.iloc[0].n_epochs)
                    r[col] = f"{sel.iloc[0].kappa4_mean:.3f}"
                    r[col + " pooled"] = f"{sel.iloc[0].pooled_kappa4:.3f}"
            if ok:
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ figure


def figure(per_night: pd.DataFrame, bidoze: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("WatchTree with heart rate", BLUE, "with_hr"), ("WatchTree without heart rate", ORANGE, "no_hr"),
              ("BiDoze v0.2.0 (accelerometer only)", GREEN, None)]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    width = 0.26
    for j, (label, colour, config) in enumerate(series):
        for i, cohort in enumerate(cv.COHORT_ORDER):
            if config is None:
                if cohort != "wearanize":
                    continue
                values = bidoze[bidoze.config == "bidoze_ctx"].kappa4.to_numpy()
            else:
                sel = per_night[(per_night.experiment == "cv_within") & (per_night.cohort == cohort) & (per_night.config == config)]
                values = sel.kappa4.to_numpy()
            mean, lo, hi = cv.ci95(values)
            x = i + (j - 1) * width
            ax.bar(x, mean, width * 0.92, color=colour, label=label if i == 0 or config is None else None, zorder=3)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color="#333333", capsize=3, lw=1, zorder=4)
            ax.text(x, hi + 0.012, f"{mean:.2f}", ha="center", va="bottom", fontsize=9, color="#222222")
    ax.set_xticks(range(len(cv.COHORT_ORDER)))
    ax.set_xticklabels([COHORT_LABEL[c] for c in cv.COHORT_ORDER])
    ax.set_ylabel("4-stage κ (mean over nights, 95% CI)")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Complete-context epochs, subject-wise 5-fold CV within each cohort", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path)


# ------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--training-config", type=Path, default=CONFIG_DIR / "training.json")
    args = parser.parse_args()
    max_rounds = TreeSpec.load(training_config=args.training_config).max_rounds
    run = args.run
    per_night = pd.read_csv(run / "per_night.csv")
    summary = pd.read_csv(run / "summary.csv")
    per_night = per_night[per_night.experiment != "bidoze"]
    summary = summary[summary.experiment != "bidoze"]
    oof = pd.read_parquet(run / "oof_predictions.parquet")
    nights, _ = bidoze_nights(run, oof, run / "features")

    comparisons = []
    for metric in ("kappa4", "kappa_sw"):
        for a, b in (
            ("watchtree_cv_within_no_hr", "bidoze_ctx"),
            ("watchtree_cv_within_with_hr", "bidoze_ctx"),
            ("watchtree_cv_within_no_hr", "bidoze_all_span"),
            ("watchtree_cv_within_with_hr", "bidoze_all_span"),
            ("watchtree_loco_no_hr", "bidoze_ctx"),
            ("watchtree_loco_with_hr", "bidoze_ctx"),
            ("bidoze_ctx", "bidoze_all_span"),
            ("watchtree_cv_within_with_hr", "watchtree_cv_within_no_hr"),
        ):
            comparisons.append(paired(nights, a, b, metric))
    comp = pd.DataFrame(comparisons)
    comp.to_csv(run / "bidoze_paired.csv", index=False)

    bnights = nights.assign(experiment="bidoze", cohort="wearanize")
    per_night = pd.concat([per_night, bnights], ignore_index=True)
    bsum = []
    for config, g in bnights.groupby("config"):
        mean, lo, hi = cv.ci95(g.kappa4)
        bsum.append({"experiment": "bidoze", "cohort": "wearanize", "config": config, "unit": "night", "n_nights": len(g),
                     "n_epochs": int(g.n_epochs.sum()), "kappa4_mean": mean, "kappa4_lo": lo, "kappa4_hi": hi,
                     "kappa_sw_mean": g.kappa_sw.mean(), "deep_pred_mean": g.deep_pred.mean(), "deep_psg_mean": g.deep_psg.mean(),
                     "rem_pred_mean": g.rem_pred.mean(), "rem_psg_mean": g.rem_psg.mean(), "tst_err_min_mean": g.tst_err_min.mean(),
                     "tst_abs_err_min_mean": g.tst_err_min.abs().mean(), "pooled_kappa4": np.nan, "pooled_kappa_sw": np.nan})
    summary = pd.concat([summary, pd.DataFrame(bsum)], ignore_index=True)
    per_night.to_csv(run / "per_night.csv", index=False)
    summary.to_csv(run / "summary.csv", index=False)

    figure(per_night, bnights, run / "kappa_by_cohort.png")

    counts = pd.read_csv(run / "counts.csv")
    counts_md = counts.assign(share_of_scored_kept=lambda d: d.share_of_scored_kept.map(lambda v: f"{100 * v:.1f}%"))
    sections = {
        "counts": md_table(counts_md),
        "leak": md_table(leak_table(summary)),
        "cv": md_table(cv_table(summary, "cv_within", ["with_hr", "no_hr"])),
        "loco": md_table(cv_table(summary, "loco", ["with_hr", "no_hr", "with_hr_no_sibling", "no_hr_no_sibling"])),
        "bidoze": md_table(comp.assign(**{c: comp[c].map(lambda v: fmt(v, 4)) for c in ("mean_a", "mean_b", "mean_diff", "median_diff", "wilcoxon_p")})),
        "quant": md_table(quant_table(per_night, summary)),
        "ablation": md_table(ablation_table(summary, max_rounds)),
        "regions": md_table(pd.read_csv(run / "leak_regions.csv").assign(
            **{c: lambda d, c=c: d[c].map(lambda v: f"{v:.3f}") for c in ("pooled_kappa4", "pooled_kappa_sw", "wake_pred", "wake_psg")})),
    }
    template = (run / "report_template.md").read_text()
    (run / "report.md").write_text(template.format(**sections))
    print("wrote", run / "report.md")


if __name__ == "__main__":
    main()

"""Full evaluation report of the full-test arms against independent ×3: sleep/wake, κ, TST and WASO.

Reads evaluate.py's ``subjects.csv`` and ``draws.csv`` from an analysis directory. For each model,
cohort and draw it takes the mean over the draw's test recordings of:
- sleep sensitivity and wake specificity at the decoded calls (%), κ sleep/wake and κ 4-stage;
- TST and WASO in minutes by the model and by PSG, the error (model − PSG) as its mean (bias), its
  SD over recordings and its mean absolute value (MAE).
Wake specificity at 95% sleep sensitivity (%) is evaluate.py's: one threshold per draw and cohort,
set on the draw's test recordings pooled. TST and WASO count the PSG-scored epochs only; WASO is
the wake from the first to the last sleep epoch of that scorer (model or PSG), and a gap call is
neither sleep nor wake.

The report gives the mean of these over the draws, and for each model the difference to
independent ×3 per draw: its mean, 95% t half-width (t with draws − 1 degrees of freedom) and the
number of draws in which the model is higher.

Writes ``report.csv`` (per model, cohort and quantity) and ``report.md`` (the tables) to the
analysis directory; with ``--figures``, ``tridoze-overnight-tst-waso.png`` there: the TST and WASO
error of each test recording in each draw, one panel per cohort and quantity.

Usage: python sleep_report.py <analysis dir> [--models M ...] [--figures DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

BASE = "independent ×3"
MODELS = [BASE, "BiDoze-G ×3", "Bagged G ×3", "Boosted G ×3", "T1 β 0.5", "T1 β 0.25"]
COHORTS = ["SleepAccel", "SleepAccel-Clinical"]
COLORS = {BASE: "#555555", "BiDoze-G ×3": "#1f77b4", "T1 β 0.5": "#d62728", "T1 β 0.25": "#ff9896",
          "Bagged G ×3": "#2ca02c", "Boosted G ×3": "#9467bd"}
# (key, header, digits); the headers carry the units.
VALUES = [("sleep_sens", "Sleep sens (%)", 1), ("wake_spec", "Wake spec (%)", 1),
          ("ws95", "Wake spec at 95% sleep sens (%)", 1), ("kappa_sw", "κ sleep/wake", 3),
          ("kappa_4", "κ 4-stage", 3),
          ("tst_pred", "TST model (min)", 0), ("tst_psg", "TST PSG (min)", 0), ("tst_bias", "TST bias (min)", 1),
          ("tst_sd", "TST error SD (min)", 1), ("tst_mae", "TST MAE (min)", 1),
          ("waso_pred", "WASO model (min)", 0), ("waso_psg", "WASO PSG (min)", 0),
          ("waso_bias", "WASO bias (min)", 1), ("waso_sd", "WASO error SD (min)", 1),
          ("waso_mae", "WASO MAE (min)", 1)]
DIFFS = ["sleep_sens", "wake_spec", "ws95", "kappa_sw", "kappa_4", "tst_bias", "tst_sd", "tst_mae",
         "waso_bias", "waso_sd", "waso_mae"]
# Columns of the markdown tables: (header, keys joined in one cell, digits).
SHOWN = [("Sleep sens (%)", ["sleep_sens"], 1), ("Wake spec (%)", ["wake_spec"], 1),
         ("Wake spec at 95% sleep sens (%)", ["ws95"], 1), ("κ sleep/wake", ["kappa_sw"], 3),
         ("κ 4-stage", ["kappa_4"], 3),
         ("TST model / PSG (min)", ["tst_pred", "tst_psg"], 0), ("TST bias ± SD (min)", ["tst_bias", "tst_sd"], 1),
         ("TST MAE (min)", ["tst_mae"], 1),
         ("WASO model / PSG (min)", ["waso_pred", "waso_psg"], 0), ("WASO bias ± SD (min)", ["waso_bias", "waso_sd"], 1),
         ("WASO MAE (min)", ["waso_mae"], 1)]
SHOWN_DIFF = [("Sleep sens (pts)", "sleep_sens", 1), ("Wake spec (pts)", "wake_spec", 1),
              ("Wake spec at 95% sleep sens (pts)", "ws95", 1), ("κ sleep/wake", "kappa_sw", 3),
              ("κ 4-stage", "kappa_4", 3), ("TST bias (min)", "tst_bias", 1), ("TST MAE (min)", "tst_mae", 1),
              ("WASO bias (min)", "waso_bias", 1), ("WASO MAE (min)", "waso_mae", 1)]


def per_draw(subjects: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, model, cohort), g in subjects.groupby(["draw", "model", "cohort"], sort=False):
        tst = g.pred_tst_minutes - g.psg_tst_minutes
        waso = g.pred_waso_minutes - g.psg_waso_minutes
        ws95 = draws[(draws.draw == d) & (draws.model == model) & (draws.cohort == cohort)
                     & (draws.metric == "wake_spec_at_95_sens")]["mean"]
        rows.append({"draw": d, "model": model, "cohort": cohort, "recordings": len(g),
                     "sleep_sens": 100 * g.sleep_sensitivity.mean(), "wake_spec": 100 * g.wake_specificity.mean(),
                     "ws95": float(ws95.iloc[0]) if len(ws95) else np.nan,
                     "kappa_sw": g.kappa_sleep_wake.mean(), "kappa_4": g.kappa_4stage.mean(),
                     "tst_pred": g.pred_tst_minutes.mean(), "tst_psg": g.psg_tst_minutes.mean(),
                     "tst_bias": tst.mean(), "tst_sd": tst.std(ddof=1), "tst_mae": tst.abs().mean(),
                     "waso_pred": g.pred_waso_minutes.mean(), "waso_psg": g.psg_waso_minutes.mean(),
                     "waso_bias": waso.mean(), "waso_sd": waso.std(ddof=1), "waso_mae": waso.abs().mean()})
    return pd.DataFrame(rows)


def summarize(draw_table: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        c = draw_table[draw_table.cohort == cohort]
        base = c[c.model == BASE].set_index("draw")
        for model in models:
            m = c[c.model == model].set_index("draw")
            if m.empty:
                continue
            for key, _, _ in VALUES:
                row = {"model": model, "cohort": cohort, "quantity": key, "draws": len(m), "mean": m[key].mean()}
                if model != BASE and key in DIFFS:
                    diff = (m[key] - base[key]).dropna()
                    n = len(diff)
                    half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                    row.update({"diff": diff.mean(), "half_width": half, "draws_higher": int((diff > 0).sum()),
                                "per_draw": " ".join(f"{v:+.3f}" for v in diff)})
                rows.append(row)
    return pd.DataFrame(rows)


def num(v: float, digits: int, signed: bool = False) -> str:
    if not np.isfinite(v):
        return "–"
    v = round(v, digits) + 0.0
    s = f"{v:+.{digits}f}" if signed and v != 0 else f"{v:.{digits}f}"
    return s.replace("-", "−")


def markdown(table: pd.DataFrame, models: list[str]) -> str:
    out = []
    for cohort in COHORTS:
        t = table[table.cohort == cohort]
        present = [m for m in models if m in set(t.model)]
        n = int(t.draws.max())
        out.append(f"**{cohort}: mean over {n} draws of each draw's per-recording mean**\n")
        out.append("| Model | " + " | ".join(h for h, _, _ in SHOWN) + " |")
        out.append("|:---|" + "--:|" * len(SHOWN))
        for m in present:
            v = t[t.model == m].set_index("quantity")["mean"]
            sep = {"tst_bias": " ± ", "waso_bias": " ± "}
            cells = [(sep.get(keys[0], " / ")).join(num(v[k], dg, k.endswith("_bias")) for k in keys)
                     for _, keys, dg in SHOWN]
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("")
        out.append(f"**{cohort}: model − independent ×3, mean ± 95% half-width over {n} draws "
                   f"(draws in which the model is higher)**\n")
        out.append("| Model | " + " | ".join(h for h, _, _ in SHOWN_DIFF) + " |")
        out.append("|:---|" + "--:|" * len(SHOWN_DIFF))
        for m in present:
            if m == BASE:
                continue
            r = t[t.model == m].set_index("quantity")
            cells = [f"{num(r.loc[k, 'diff'], dg, True)} ± {num(r.loc[k, 'half_width'], dg)} "
                     f"({int(r.loc[k, 'draws_higher'])}/{n})" for _, k, dg in SHOWN_DIFF]
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def figure(subjects: pd.DataFrame, models: list[str], path: Path) -> None:
    quantities = [("tst", "TST error, model − PSG (min)"), ("waso", "WASO error, model − PSG (min)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey="row")
    rng = np.random.default_rng(0)
    for r, (q, label) in enumerate(quantities):
        for c, cohort in enumerate(COHORTS):
            ax = axes[r, c]
            s = subjects[subjects.cohort == cohort]
            present = [m for m in models if m in set(s.model)]
            for i, m in enumerate(present):
                g = s[s.model == m]
                err = (g[f"pred_{q}_minutes"] - g[f"psg_{q}_minutes"]).to_numpy()
                ax.scatter(i + rng.uniform(-0.25, 0.25, len(err)), err, s=9, alpha=0.45, color=COLORS.get(m, "#333333"),
                           linewidths=0)
                ax.errorbar(i + 0.36, err.mean(), yerr=err.std(ddof=1), fmt="o", ms=4, color="black", capsize=3,
                            label="Mean ± SD over recordings" if i == 0 else None)
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_xticks(range(len(present)), [m.replace(" ×3", "\n×3").replace(" β", "\nβ") for m in present],
                          fontsize=8)
            ax.set_title(cohort, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        axes[r, 0].set_ylabel(label)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--figures", type=Path, default=None)
    args = ap.parse_args()
    subjects = pd.read_csv(args.analysis / "subjects.csv", dtype={"subject_id": str})
    draws = pd.read_csv(args.analysis / "draws.csv")
    subjects = subjects[subjects.model.isin(args.models)]
    table = summarize(per_draw(subjects, draws), args.models)
    table.to_csv(args.analysis / "report.csv", index=False)
    text = markdown(table, args.models)
    (args.analysis / "report.md").write_text(text + "\n")
    print(text)
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        figure(subjects, args.models, args.figures / "tridoze-overnight-tst-waso.png")


if __name__ == "__main__":
    main()

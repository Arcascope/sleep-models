"""Figures for the overnight TriDoze log (reviews/2026-10-16/tridoze-overnight.md).

Reads ``<run>/analysis/`` (insight/, full-test/draws.csv, oob.csv) and the boosting weight summaries
(``bidoze_g_boost<k>/draw<d>/epoch_weights.json``); writes ``tridoze-overnight-*.png`` to the
directory given (the research-log figures directory).

Usage: python overnight.py <figures dir> [--run RUN]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN = Path(__file__).resolve().parents[3] / "results/tridoze-2026-10-16"
STAGES = ["wake", "light", "deep", "rem"]
DISPLAY = {"bidoze_g": "BiDoze-G\nseed 0"}
STAGE_NAMES = {"wake": "Wake", "light": "Light", "deep": "Deep", "rem": "REM"}
COHORTS = ["SleepAccel", "SleepAccel-Clinical"]
COLORS = {"independent ×3": "#555555", "BiDoze-G ×3": "#1f77b4", "T1 β 0.5": "#d62728", "T1 β 0.25": "#ff9896",
          "Bagged G ×3": "#2ca02c", "Boosted G ×3": "#9467bd"}


def ensembles_present(df: pd.DataFrame) -> list[str]:
    return [e for e in COLORS if e in set(df.ensemble)]


def grouped_bars(ax, table: pd.DataFrame, groups: list[str], labels: dict[str, str], ylabel: str, err=None) -> None:
    names = [e for e in COLORS if e in table.index]
    width = 0.8 / max(len(names), 1)
    for i, e in enumerate(names):
        x = np.arange(len(groups)) + (i - (len(names) - 1) / 2) * width
        ax.bar(x, table.loc[e, groups], width, color=COLORS[e], label=e,
               yerr=None if err is None else err.loc[e, groups], capsize=2)
    ax.set_xticks(np.arange(len(groups)), [labels.get(g, g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def recall_spread(run: Path, out: Path) -> None:
    rec = pd.read_csv(run / "analysis/insight/recall.csv")
    rec = rec[rec.member != "mean"]
    spread = rec.groupby(["ensemble", "draw", "cohort", "stage"]).recall.agg(lambda r: r.max() - r.min()).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        g = spread[spread.cohort == cohort].groupby(["ensemble", "stage"]).recall
        grouped_bars(ax, 100 * g.mean().unstack(), STAGES, STAGE_NAMES,
                     "Member recall spread (pts)", 100 * g.std().unstack())
        ax.set_title(cohort, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tridoze-overnight-recall-spread.png", dpi=150)
    plt.close(fig)


def disagreement(run: Path, out: Path) -> None:
    dis = pd.read_csv(run / "analysis/insight/disagreement.csv")
    groups = ["psg wake", "psg light", "psg deep", "psg rem", "near transition", "away from transition",
              "movement low", "movement mid", "movement high"]
    labels = {"psg wake": "Wake", "psg light": "Light", "psg deep": "Deep", "psg rem": "REM",
              "near transition": "≤ 1 epoch\nfrom change", "away from transition": "> 1 epoch\nfrom change",
              "movement low": "Low\nmovement", "movement mid": "Mid\nmovement", "movement high": "High\nmovement"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        g = dis[dis.cohort == cohort].groupby(["ensemble", "group"]).disagree_share.mean().unstack()
        grouped_bars(ax, 100 * g, groups, labels, "Epochs where\nmembers differ (%)")
        ax.set_title(cohort, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "tridoze-overnight-disagreement.png", dpi=150)
    plt.close(fig)


def diversity(run: Path, out: Path) -> None:
    div = pd.read_csv(run / "analysis/insight/diversity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        d = div[div.cohort == cohort]
        member = d.groupby(["ensemble", "stage"]).member_nll.mean().unstack()
        mean = d.groupby(["ensemble", "stage"]).mean_nll.mean().unstack()
        names = [e for e in COLORS if e in member.index]
        groups = ["all", *STAGES]
        width = 0.8 / len(names)
        for i, e in enumerate(names):
            x = np.arange(len(groups)) + (i - (len(names) - 1) / 2) * width
            ax.bar(x, member.loc[e, groups], width, color=COLORS[e], alpha=0.35)
            ax.bar(x, mean.loc[e, groups], width, color=COLORS[e], label=e)
        ax.set_xticks(np.arange(len(groups)), ["All", *[STAGE_NAMES[s] for s in STAGES]])
        ax.set_ylabel("Log loss (nats/epoch)")
        ax.set_title(cohort, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tridoze-overnight-diversity.png", dpi=150)
    plt.close(fig)


def tick(model: str) -> str:
    label = DISPLAY.get(model, model)
    if "\n" in label:
        return label
    return label.replace(" member ", "\nmember ") if " member " in label else label.replace(" ", "\n", 1)


def kappa_diffs(run: Path, out: Path, models: list[str], name: str, draws_csv: Path | None = None) -> None:
    draws = pd.read_csv(draws_csv or run / "analysis/full-test/draws.csv")
    metrics = [("kappa_4stage", "κ 4-stage", 1), ("kappa_sleep_wake", "κ sleep/wake", 1),
               ("wake_spec_at_95_sens", "Wake spec at 95% sens (pts)", 1)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharey="col")
    for r, cohort in enumerate(COHORTS):
        for c, (metric, title, _) in enumerate(metrics):
            ax = axes[r, c]
            d = draws[(draws.cohort == cohort) & (draws.metric == metric)]
            base = d[d.model == "independent ×3"].set_index("draw")["mean"]
            shown = [m for m in models if m in set(d.model)]
            for i, m in enumerate(shown):
                diff = (d[d.model == m].set_index("draw")["mean"] - base).dropna()
                ax.scatter(np.full(len(diff), i) + np.linspace(-0.12, 0.12, len(diff)), diff, s=14,
                           color=COLORS.get(m, "#8c564b" if "stacked" in m else "#333333"))
                if len(diff) > 1:
                    half = 2.776 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) == 5 else np.nan
                    ax.errorbar(i + 0.25, diff.mean(), yerr=half, fmt="_", color="black", capsize=3)
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_xticks(range(len(shown)), [tick(m) for m in shown], fontsize=7)
            ax.set_title(f"{cohort}: {title}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Model − independent ×3")
    axes[1, 0].set_ylabel("Model − independent ×3")
    fig.tight_layout()
    fig.savefig(out / f"tridoze-overnight-{name}.png", dpi=150)
    plt.close(fig)


def boost_weights(run: Path, out: Path) -> None:
    rows = []
    for k in (1, 2):
        for d in range(5):
            p = run / f"bidoze_g_boost{k}r/draw{d}/epoch_weights.json"
            p = p if p.exists() else run / f"bidoze_g_boost{k}/draw{d}/epoch_weights.json"
            if p.exists():
                s = json.loads(p.read_text())
                rows.append({"member": k, "draw": d, **{f"w_{x}": s["stage_weight_share"][x] for x in STAGES},
                             **{f"e_{x}": s["stage_epoch_share"][x] for x in STAGES},
                             "transition_w": s["transition_weight_share"], "transition_e": s["transition_epoch_share"],
                             "ess": s["ess_fraction"], "err": s["err"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [3, 1.2]})
    groups = [*STAGES, "transition"]
    x = np.arange(len(groups))
    share_e = [df[f"e_{s}"].mean() for s in STAGES] + [df.transition_e.mean()]
    axes[0].bar(x - 0.27, 100 * np.array(share_e), 0.27, color="#bbbbbb", label="Share of epochs")
    for i, k in enumerate((1, 2)):
        d = df[df.member == k]
        share_w = [d[f"w_{s}"].mean() for s in STAGES] + [d.transition_w.mean()]
        axes[0].bar(x + i * 0.27, 100 * np.array(share_w), 0.27, color=["#c5b0d5", "#9467bd"][i],
                    label=f"Share of weight, member {k}")
    axes[0].set_xticks(x, ["Wake", "Light", "Deep", "REM", "≤ 1 epoch\nfrom change"])
    axes[0].set_ylabel("% of fit epochs or weight")
    axes[0].legend(frameon=False, fontsize=8)
    for k, color in zip((1, 2), ["#c5b0d5", "#9467bd"]):
        d = df[df.member == k]
        axes[1].scatter(np.full(len(d), k), 100 * d.ess, color=color, s=18)
    axes[1].set_xticks([1, 2], ["Member 1", "Member 2"])
    axes[1].set_xlim(0.5, 2.5)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Effective sample size (% of epochs)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "tridoze-overnight-boost-weights.png", dpi=150)
    plt.close(fig)


def second_member(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "analysis/iter2-boost/second_member.csv")
    df = df[~df.pair.str.startswith("BiDoze-G")]
    df["pair"] = df.pair.str.replace("independent: seed . given seed 0", "independent seed 1 or 2", regex=True)
    groups = ["second right | first wrong", "second right | first right", "both wrong"]
    labels = ["Right where\nmember 0 is wrong", "Right where\nmember 0 is right", "Both wrong\n(share of all epochs)"]
    g = 100 * df.groupby("pair")[groups].mean()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = {"boosted: member 1 given member 0": "#9467bd", "independent seed 1 or 2": "#555555"}
    names = {"boosted: member 1 given member 0": "Boosted member 1", "independent seed 1 or 2": "Independent seed 1 or 2"}
    for i, (pair, row) in enumerate(g.iterrows()):
        x = np.arange(len(groups)) + (i - 0.5) * 0.38
        ax.bar(x, row[groups], 0.38, color=colors[pair], label=names[pair])
        for xi, v in zip(x, row[groups]):
            ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(groups)), labels)
    ax.set_ylabel("% of fit epochs")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "tridoze-overnight-second-member.png", dpi=150)
    plt.close(fig)


def member_vs_ensemble(run: Path, out: Path) -> None:
    t = pd.read_csv(run / "analysis/full-test/member_vs_ensemble.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, cohort in zip(axes, COHORTS):
        c = t[t.cohort == cohort]
        for e in ensembles_present(c):
            g = c[c.ensemble == e]
            ax.scatter(g.members, g.ensemble_kappa, s=12, color=COLORS[e], alpha=0.45)
            ax.scatter(g.members.mean(), g.ensemble_kappa.mean(), s=70, color=COLORS[e], edgecolor="black", label=e)
        lo = min(c.members.min(), c.ensemble_kappa.min()) - 0.01
        hi = max(c.members.max(), c.ensemble_kappa.max()) + 0.01
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8)
        ax.set_xlabel("Members' mean κ 4-stage (each alone)")
        ax.set_ylabel("Ensemble κ 4-stage (mean output)")
        ax.set_title(cohort, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tridoze-overnight-member-vs-ensemble.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", type=Path)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, help="models of the κ difference figure")
    ap.add_argument("--name", default="kappa", help="κ difference figure name")
    ap.add_argument("--draws-csv", type=Path, default=None, help="evaluate.py draws.csv (default analysis/full-test)")
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    jobs = {"recall": recall_spread, "disagreement": disagreement, "diversity": diversity, "boost": boost_weights, "second": second_member, "members": member_vs_ensemble}
    for name, job in jobs.items():
        if args.only is None or name in args.only:
            job(args.run, args.figures)
    if args.models:
        kappa_diffs(args.run, args.figures, args.models, args.name, args.draws_csv)


if __name__ == "__main__":
    main()

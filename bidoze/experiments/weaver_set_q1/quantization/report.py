"""Score the quantization conditions per arm and compare the arms' penalties.

Models: per arm and draw, the 3-seed mean-of-probabilities ensemble of the Q1 checkpoints, decoded with
``hybrid_calls`` and the draw's original adult stage divisor (as ``analysis/summarize.py``). Test set:
each draw's test adults of SleepAccel and SleepAccel-Clinical that have raw accelerometer data
(``build_features.py``). Conditions: native, quantized_256, stride2, quantized_256_stride2.

Per cohort, arm and condition: per draw, the mean over the draw's test participants of 4-stage kappa,
sleep/wake kappa, sleep sensitivity, wake specificity, recall of each stage and TST bias (model - PSG, minutes);
then the mean and SD over draws. Contrasts, each as mean ± SD over draws with a participant-cluster bootstrap
95% interval (20,000 resamples; one resampling weight per participant reused in every draw and draws
with no participant in a resample are redrawn; each resample averages within draw, then across draws):
- (a) condition - native, per arm
- (b) arm - ``all`` under each condition, for ``matched`` and ``all-adult-share``
- (c) the difference of the two arms' (condition - native): (arm - native change) - (``all`` change)

Writes to ``<run>/quantized/analysis/``: subjects.csv, draws.csv, summary.csv, contrasts.csv, summary.md.

Usage: python report.py <run dir> [--arms matched all all-adult-share]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE))
import summarize as S  # noqa: E402
from build_features import CONDITIONS, DRAW_KEY, test_ids  # noqa: E402

from arcasleep_bidoze.adult import hybrid_calls  # noqa: E402
from score import STAGES, load_subject  # noqa: E402
from card_metrics import subject_metrics  # noqa: E402

SAVED_PREFIX = DRAW_KEY
SEEDS = [0, 1, 2]
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity",
           "recall_wake", "recall_light", "recall_deep", "recall_rem", "tst_bias_minutes"]
POINTS = {"sleep_sensitivity", "wake_specificity", "recall_wake", "recall_light", "recall_deep", "recall_rem"}
REPLICATES = 20000


def score_subjects(run: Path, arms: list[str], baseline: Path, labels_root: Path, missing: dict[str, set[str]]) -> pd.DataFrame:
    config, _ = S.load_metrics_config(S.BIDOZE / "config/metrics.json")
    logger = S.MetricsLogger(config)
    bidsleep = S.night_counts([s.path for s in S.discover_cached_subjects(["bidsleep"], labels_root / S.CACHE_NAME)["bidsleep"]])
    draws = json.loads((HERE.parent / "draws.json").read_text())
    rows = []
    for d in range(5):
        divisor = S.draw_divisor(baseline, d, bidsleep)
        for arm in arms:
            for cohort, key in DRAW_KEY.items():
                for sid in draws[f"draw{d}"][key]:
                    if sid in missing[cohort]:
                        continue
                    files = [run / f"quantized/predictions/{arm}/draw{d}/{cohort}_{sid}_seed{s}.npz" for s in SEEDS]
                    if not all(f.exists() for f in files):
                        print(f"skip {arm} draw{d} {cohort} {sid}: predictions missing")
                        continue
                    p = np.mean([np.load(f)["probabilities"] for f in files], axis=0)
                    y, _ = load_subject(run / f"training/{arm}/draw{d}/predictions/draw{d}_seed_0/{SAVED_PREFIX[cohort]}_{sid}.csv")
                    scored = (y >= 0) & (y < len(STAGES))
                    for k, condition in enumerate(CONDITIONS):
                        calls = hybrid_calls(p[k], p[k], divisor)
                        m = subject_metrics(logger, y, p[k], calls)
                        recalls = {f"recall_{name}": float(np.mean(calls[scored & (y == c)] == c)) if np.any(scored & (y == c)) else np.nan
                                   for c, name in enumerate(STAGES)}
                        tst = 0.5 * (np.sum((calls[scored] > 0) & (calls[scored] < 4)) - np.sum(y[scored] > 0))
                        rows.append(dict(draw=d, arm=arm, cohort=cohort, subject_id=sid, condition=condition,
                                         **{k_: m[k_] for k_ in METRICS[:4]}, **recalls, tst_bias_minutes=float(tst)))
    return pd.DataFrame(rows)


def bootstrap_weights(ids: list[str], present: np.ndarray) -> np.ndarray:
    """Multinomial participant weights, ``(REPLICATES, participants)``, with every draw represented."""
    rng = np.random.default_rng(20261021)
    kept = []
    while sum(len(k) for k in kept) < REPLICATES:
        w = rng.multinomial(len(ids), np.ones(len(ids)) / len(ids), size=50000)
        kept.append(w[np.all(w @ present.T > 0, axis=1)])
    return np.concatenate(kept)[:REPLICATES]


def contrast(delta: pd.Series, ids: list[str], weights: np.ndarray) -> dict:
    """``delta`` is indexed by (draw, subject_id)."""
    by_draw = delta.groupby("draw").mean()
    boot = []
    for d in sorted(delta.index.get_level_values("draw").unique()):
        g = delta.loc[d].dropna()
        w = weights[:, [ids.index(s) for s in g.index]]
        boot.append(w @ g.to_numpy() / w.sum(axis=1))
    lo, hi = np.percentile(np.mean(boot, axis=0), [2.5, 97.5])
    return dict(mean=by_draw.mean(), sd=by_draw.std(), boot_low=lo, boot_high=hi, draws=len(by_draw))


def contrasts(subjects: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rows = []
    for cohort, part in subjects.groupby("cohort"):
        ids = sorted(part.subject_id.unique())
        present = np.array([[s in set(g.subject_id) for s in ids] for _, g in part[part.condition == "native"].groupby(["draw"])], dtype=int)
        weights = bootstrap_weights(ids, present)
        for metric in METRICS:
            wide = part.pivot(index=["draw", "subject_id"], columns=["arm", "condition"], values=metric)
            def add(kind: str, label: str, delta: pd.Series) -> None:
                rows.append(dict(cohort=cohort, kind=kind, contrast=label, metric=metric, **contrast(delta, ids, weights)))
            for arm in arms:
                for c in CONDITIONS[1:]:
                    add("a: condition - native", f"{arm}: {c} - native", wide[(arm, c)] - wide[(arm, "native")])
            for arm in arms:
                if arm == "all" or "all" not in arms:
                    continue
                for c in CONDITIONS:
                    add("b: arm - all", f"{arm} - all: {c}", wide[(arm, c)] - wide[("all", c)])
                for c in CONDITIONS[1:]:
                    add("c: penalty difference", f"{arm} - all: ({c} - native)",
                        (wide[(arm, c)] - wide[(arm, "native")]) - (wide[("all", c)] - wide[("all", "native")]))
    return pd.DataFrame(rows)


def fmt(metric: str, x: float, sd: float | None = None) -> str:
    scale = 100 if metric in POINTS else 1
    digits = 1 if metric == "tst_bias_minutes" or metric in POINTS else 3
    s = f"{x * scale:.{digits}f}".replace("-", "−")
    return s if sd is None else f"{s} ± {sd * scale:.{digits}f}"


def md_table(frame: pd.DataFrame) -> str:
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "|" + ":---|" * len(frame.columns)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in frame.to_numpy()])


def write_report(summary: pd.DataFrame, table: pd.DataFrame, missing: dict[str, set[str]], arms: list[str]) -> str:
    out = ["# Quantization of the Q1 models (October 21 §6b)", "",
           "Sleep sensitivity, wake specificity and recalls are in percentage points; TST bias is model − PSG in minutes; "
           "κ is unitless. Mean ± SD over draws; brackets are participant-cluster bootstrap 95% intervals.", ""]
    for cohort in summary.cohort.unique():
        n = len(test_ids(cohort)) - len(missing[cohort])
        out += [f"## {cohort} ({n} test participants with raw accelerometer data across the 5 draws)", "", "### Per arm and condition", ""]
        s = summary[summary.cohort == cohort]
        rows = [[arm, c] + [fmt(m, *s[(s.arm == arm) & (s.condition == c) & (s.metric == m)][["mean", "sd"]].iloc[0]) for m in METRICS]
                for arm in arms for c in CONDITIONS]
        out += [md_table(pd.DataFrame(rows, columns=["arm", "condition", *METRICS])), ""]
        for kind in sorted(table.kind.unique()):
            t = table[(table.cohort == cohort) & (table.kind == kind)]
            out += [f"### ({kind})", ""]
            rows = []
            for label in t.contrast.unique():
                g = t[t.contrast == label].set_index("metric")
                rows.append([label] + [f"{fmt(m, g.loc[m, 'mean'], g.loc[m, 'sd'])} [{fmt(m, g.loc[m, 'boot_low'])}, {fmt(m, g.loc[m, 'boot_high'])}]"
                                       for m in METRICS])
            out += [md_table(pd.DataFrame(rows, columns=["contrast", *METRICS])), ""]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--arms", nargs="+", default=["matched", "all", "all-adult-share"])
    ap.add_argument("--baseline-run", type=Path, default=S.BASELINE)
    ap.add_argument("--labels-root", type=Path, default=S.LABELS)
    a = ap.parse_args()
    run = a.run.resolve()
    arms = [x for x in a.arms if (run / f"quantized/predictions/{x}").is_dir()]
    missing = {c: set((run / f"quantized/features/{c}/missing.txt").read_text().split()) for c in DRAW_KEY}
    subjects = score_subjects(run, arms, a.baseline_run, a.labels_root, missing)
    out = run / "quantized/analysis"
    out.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(out / "subjects.csv", index=False)
    draws = subjects.groupby(["cohort", "arm", "condition", "draw"])[METRICS].mean().reset_index()
    draws.to_csv(out / "draws.csv", index=False)
    long = draws.melt(id_vars=["cohort", "arm", "condition", "draw"], var_name="metric")
    summary = long.groupby(["cohort", "arm", "condition", "metric"]).value.agg(mean="mean", sd="std").reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    table = contrasts(subjects, arms)
    table.to_csv(out / "contrasts.csv", index=False)
    text = write_report(summary, table, missing, arms)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

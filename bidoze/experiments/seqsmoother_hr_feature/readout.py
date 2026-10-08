"""Does HR add light/deep/REM information beyond the adult release's own output?

Within the epochs the release calls sleep, a multinomial logistic readout chooses light, deep
or REM from the release's cross-validated log-probabilities, with or without HR features. The
release's sleep/wake (and gap) calls are kept as they are, so sleep sensitivity and wake
specificity are the same in every arm.

Arms:
- ``release``: the released rule, light/deep/REM divided by the adult stage divisor.
- ``readout``: log P(wake), log P(light), log P(deep), log P(REM) of the 3-seed mean output.
- ``readout + <hr>``: the same plus HR features of one source (``seqsmoother``, ``jerk``,
  ``watch``), each from the HR within the night only:
  - HR relative to the night's median HR;
  - that, averaged over ±5 min;
  - SD of HR over ±2 min and over ±5 min;
  - HR change from the previous epoch, and its mean absolute value over ±2 min;
  - an indicator for epochs without HR (their features are 0).
  No input describes time since the recording started or position in the night.

The readout trains on PSG light/deep/REM epochs. SleepAccel: cross-validated by subject on the
folds of ``sleepaccel_5fold.json`` (the folds the release's fold models used).
SleepAccel-Clinical: the readout trained on all of SleepAccel, applied unchanged.

Per subject: 4-stage κ (PSG wake/light/deep/REM against the calls; predicted gap left out),
recall of each stage, predicted deep % of the scored epochs and PSG deep %. Paired comparisons:
mean difference over subjects, Wilcoxon p, and a 95% bootstrap CI (10,000 subject resamples).
Held-out multinomial log-loss on PSG sleep epochs is reported as a threshold-free measure.

Per-night deep share: Pearson r of predicted against PSG deep % over subjects, and the mean\nabsolute difference in percentage points.\n\nWrites ``<run>/readout/``: subjects.csv, summary.csv, paired.csv, logloss.csv, deep_tracking.csv.

Usage: python readout.py <run>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

COHORTS = ["SleepAccel", "SleepAccel-Clinical"]
STAGES = ["wake", "light", "deep", "rem"]
SOURCES = {"seqsmoother": "hr_seqsmoother", "jerk": "hr_jerk", "watch": "hr_watch"}
BASE = ["lp_wake", "lp_light", "lp_deep", "lp_rem"]
EPS = 1e-6
BOOT = 10_000


def hr_features(g: pd.DataFrame, column: str) -> pd.DataFrame:
    """HR features of one night (rows in epoch order)."""
    hr = g[column].where(g[column].between(30, 200))
    rel = hr - hr.median()
    roll = lambda s, half, fn: getattr(s.rolling(2 * half + 1, center=True, min_periods=2), fn)()
    diff = hr.diff()
    f = pd.DataFrame({
        "hr_rel": rel,
        "hr_rel_5min": roll(rel, 10, "mean"),
        "hr_sd_2min": roll(hr, 4, "std"),
        "hr_sd_5min": roll(hr, 10, "std"),
        "hr_change": diff,
        "hr_abs_change_2min": roll(diff.abs(), 4, "mean"),
    }, index=g.index)
    f["hr_missing"] = hr.isna().astype(float)
    return f.fillna(0.0)


def build(epochs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    d = epochs.copy()
    for s in STAGES:
        d["lp_" + s] = np.log(d["p_" + s] + EPS)
    columns = {}
    for name, col in SOURCES.items():
        f = d.groupby(["cohort", "subject"], group_keys=False)[[col]].apply(lambda g: hr_features(g, col))
        f.columns = [f"{name}:{c}" for c in f.columns]
        d = d.join(f)
        columns[name] = list(f.columns)
    return d, columns


def release_calls(d: pd.DataFrame, divisors: dict) -> np.ndarray:
    p = d[["p_wake", "p_light", "p_deep", "p_rem", "p_gap"]].to_numpy()
    calls = p.argmax(axis=1)
    div = np.array([divisors["final"] if f < 0 else divisors["folds"][str(f)] for f in d.fold])
    sleep = (calls >= 1) & (calls <= 3)
    calls[sleep] = 1 + (p[sleep, 1:4] / div[sleep, 1:4]).argmax(axis=1)
    return calls


def fit(train: pd.DataFrame, features: list[str]):
    rows = train[train.psg.between(1, 3)]
    scaler = StandardScaler().fit(rows[features])
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(rows[features]), rows.psg)
    return scaler, model


def readout_calls(d: pd.DataFrame, features: list[str], scaler, model) -> tuple[np.ndarray, np.ndarray]:
    """Calls with the readout's stage on model-called sleep epochs; and P(stage) for every epoch."""
    p = d[["p_wake", "p_light", "p_deep", "p_rem", "p_gap"]].to_numpy()
    calls = p.argmax(axis=1)
    proba = model.predict_proba(scaler.transform(d[features]))
    sleep = (calls >= 1) & (calls <= 3)
    calls[sleep] = model.classes_[proba[sleep].argmax(axis=1)]
    return calls, proba


def subject_metrics(y: np.ndarray, calls: np.ndarray) -> dict:
    scored = (y >= 0) & (y <= 3)
    ok = scored & (calls >= 0) & (calls <= 3)
    cm = np.zeros((4, 4))
    np.add.at(cm, (y[ok], calls[ok]), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(1) * cm.sum(0)).sum() / n ** 2
    out = {"kappa_4stage": (po - pe) / (1 - pe)}
    for c, s in enumerate(STAGES):
        m = scored & (y == c)
        out[f"recall_{s}"] = float(np.mean(calls[m] == c)) if m.any() else np.nan
    pred = np.bincount(calls[scored].clip(0, 4), minlength=5)[:4]
    out["deep%"] = 100 * pred[2] / pred.sum()
    out["psg_deep%"] = 100 * np.mean(y[scored] == 2)
    out["sleep_sensitivity"] = float(np.mean((calls[scored & (y > 0)] >= 1) & (calls[scored & (y > 0)] <= 3)))
    out["wake_specificity"] = float(np.mean(calls[scored & (y == 0)] == 0))
    return out


def bootstrap_ci(diff: np.ndarray, rng) -> tuple[float, float]:
    means = diff[rng.integers(len(diff), size=(BOOT, len(diff)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(run: Path) -> None:
    out = run / "readout"
    out.mkdir(exist_ok=True)
    epochs = pd.read_csv(run / "epochs.csv.gz", dtype={"subject": str})
    divisors = json.loads((run / "release_divisors.json").read_text())
    d, hr_cols = build(epochs)
    arms = {"readout": BASE, **{f"readout + {k}": BASE + v for k, v in hr_cols.items()}}
    sa = d[d.cohort.eq("SleepAccel")]
    cl = d[d.cohort.eq("SleepAccel-Clinical")]

    calls = {"release": release_calls(d, divisors)}
    losses = []
    for arm, feats in arms.items():
        c = calls["release"].copy()
        proba = np.full((len(d), 3), np.nan)
        for k in range(5):
            test = d.cohort.eq("SleepAccel") & d.fold.eq(k)
            train = d.cohort.eq("SleepAccel") & ~d.fold.eq(k)
            s, m = fit(d[train], feats)
            c[test.to_numpy()], proba[test.to_numpy()] = readout_calls(d[test], feats, s, m)
        clin = d.cohort.eq("SleepAccel-Clinical").to_numpy()
        s, m = fit(sa, feats)
        c[clin], proba[clin] = readout_calls(cl, feats, s, m)
        if arm == "readout + watch":
            c[clin] = -9  # not scored: no watch HR in SleepAccel-Clinical
        calls[arm] = c
        for cohort in COHORTS:
            rows = (d.cohort.eq(cohort) & d.psg.between(1, 3)).to_numpy()
            if arm == "readout + watch" and cohort != "SleepAccel":
                continue
            per = []
            for sid, g in d[rows].groupby("subject"):
                idx = np.flatnonzero(rows & d.subject.eq(sid).to_numpy())
                per.append(log_loss(d.psg.to_numpy()[idx], proba[idx], labels=[1, 2, 3]))
            losses.append({"cohort": cohort, "arm": arm, "logloss_mean_over_subjects": float(np.mean(per)),
                           "logloss_pooled": float(log_loss(d.psg.to_numpy()[rows], proba[rows], labels=[1, 2, 3]))})
    pd.DataFrame(losses).to_csv(out / "logloss.csv", index=False)

    rows = []
    y = d.psg.to_numpy()
    for arm, c in calls.items():
        for (cohort, sid), g in d.groupby(["cohort", "subject"]):
            idx = g.index.to_numpy()
            if (c[idx] == -9).any():
                continue
            rows.append({"cohort": cohort, "subject": sid, "arm": arm, **subject_metrics(y[idx], c[idx])})
    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(out / "subjects.csv", index=False)
    metrics = [m for m in per_subject.columns if m not in ("cohort", "subject", "arm")]
    summary = per_subject.groupby(["cohort", "arm"], sort=False)[metrics].mean().reset_index()
    summary.insert(2, "n", per_subject.groupby(["cohort", "arm"], sort=False).size().to_numpy())
    summary.to_csv(out / "summary.csv", index=False)
    tracking = per_subject.groupby(["cohort", "arm"], sort=False).apply(lambda g: pd.Series({
        "r_deep_vs_psg": g["deep%"].corr(g["psg_deep%"]),
        "mean_abs_deep_error_points": (g["deep%"] - g["psg_deep%"]).abs().mean()})).reset_index()
    tracking.to_csv(out / "deep_tracking.csv", index=False)

    rng = np.random.default_rng(20261008)
    paired = []
    for cohort in COHORTS:
        sub = per_subject[per_subject.cohort.eq(cohort)].pivot(index="subject", columns="arm")
        for a in [x for x in calls if x not in ("release",)]:
            for b in ["readout", "release"]:
                if a == b or (a, "kappa_4stage") not in [(c[1], c[0]) for c in sub.columns]:
                    continue
                for metric in ["kappa_4stage", "deep%", "recall_light", "recall_deep", "recall_rem"]:
                    diff = (sub[(metric, a)] - sub[(metric, b)]).dropna().to_numpy()
                    if len(diff) == 0:
                        continue
                    lo, hi = bootstrap_ci(diff, rng)
                    paired.append({"cohort": cohort, "a": a, "b": b, "metric": metric, "n": len(diff),
                                   "diff": diff.mean(), "ci_low": lo, "ci_high": hi,
                                   "p": float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0})
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(pd.DataFrame(losses).round(4).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

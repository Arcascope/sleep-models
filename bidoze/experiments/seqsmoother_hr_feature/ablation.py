"""Which HR features carry the stage information: per-group readouts, and HR agreement by stage.

Feature groups (each added alone to the readout's log-probabilities):
- level: HR relative to the night's median, and its ±5-min mean;
- variability: SD of HR over ±2 and ±5 min, HR change and its ±2-min mean absolute value.
Both groups keep the missing-HR indicator. Scored as ``readout.py`` scores: SleepAccel
cross-validated on the BiDoze folds; SleepAccel-Clinical with the readout trained on all
SleepAccel. Also writes the SleepAccel per-stage agreement of SeqSmoother and jerk HR features
with the watch's (PSG sleep epochs; Pearson r over epochs, mean over subjects).

κ differences against the readout without HR: mean over subjects, 95% bootstrap CI, Wilcoxon p.

Writes ``<run>/readout/ablation.csv`` and ``<run>/readout/feature_agreement.csv``.

Usage: python ablation.py <run>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from scipy.stats import wilcoxon

from readout import BASE, bootstrap_ci, SOURCES, build, fit, readout_calls, release_calls, subject_metrics

GROUPS = {"level": ["hr_rel", "hr_rel_5min", "hr_missing"],
          "variability": ["hr_sd_2min", "hr_sd_5min", "hr_change", "hr_abs_change_2min", "hr_missing"]}


def main(run: Path) -> None:
    out = run / "readout"
    d, _ = build(pd.read_csv(run / "epochs.csv.gz", dtype={"subject": str}))
    divisors = json.loads((run / "release_divisors.json").read_text())
    base_calls = release_calls(d, divisors)
    y = d.psg.to_numpy()
    sa = d.cohort.eq("SleepAccel").to_numpy()
    rows, per_subject = [], []
    arms = {"readout": BASE}
    for source in SOURCES:
        for group, names in GROUPS.items():
            arms[f"{source}: {group}"] = BASE + [f"{source}:{n}" for n in names]
    for arm, feats in arms.items():
        c = base_calls.copy()
        proba = np.full((len(d), 3), np.nan)
        for k in range(5):
            test = (sa & d.fold.eq(k)).to_numpy()
            s, m = fit(d[sa & ~d.fold.eq(k).to_numpy()], feats)
            c[test], proba[test] = readout_calls(d[test], feats, s, m)
        clin = ~sa
        s, m = fit(d[sa], feats)
        c[clin], proba[clin] = readout_calls(d[clin], feats, s, m)
        for cohort, mask in (("SleepAccel", sa), ("SleepAccel-Clinical", ~sa)):
            if arm.startswith("watch") and cohort != "SleepAccel":
                continue
            kappas = [subject_metrics(y[g.index], c[g.index])["kappa_4stage"] for _, g in d[mask].groupby("subject")]
            per_subject += [{"cohort": cohort, "arm": arm, "subject": sid, "kappa_4stage": k}
                            for (sid, _), k in zip(d[mask].groupby("subject"), kappas)]
            sleep = mask & (y >= 1) & (y <= 3)
            rows.append({"cohort": cohort, "arm": arm, "kappa_4stage": float(np.mean(kappas)),
                         "logloss_pooled": float(log_loss(y[sleep], proba[sleep], labels=[1, 2, 3]))})
    table = pd.DataFrame(rows)
    subjects = pd.DataFrame(per_subject).pivot_table(index=["cohort", "subject"], columns="arm", values="kappa_4stage")
    rng = np.random.default_rng(20261008)
    for i, row in table.iterrows():
        diff = (subjects.loc[row.cohort, row.arm] - subjects.loc[row.cohort, "readout"]).dropna().to_numpy()
        if row.arm != "readout":
            lo, hi = bootstrap_ci(diff, rng)
            table.loc[i, ["diff_vs_readout", "ci_low", "ci_high", "p"]] = [diff.mean(), lo, hi, wilcoxon(diff).pvalue]
    table.to_csv(out / "ablation.csv", index=False)
    print(table.round(4).to_string(index=False))

    agree = []
    s = d[sa & (y >= 1) & (y <= 3)]
    for feature in ["hr_rel", "hr_rel_5min", "hr_sd_2min", "hr_sd_5min", "hr_change"]:
        for source in ("seqsmoother", "jerk"):
            r = [g[f"{source}:{feature}"].corr(g[f"watch:{feature}"]) for _, g in s.groupby("subject")]
            agree.append({"feature": feature, "source": source, "r_with_watch": float(np.nanmean(r))})
    agreement = pd.DataFrame(agree)
    agreement.to_csv(out / "feature_agreement.csv", index=False)
    print(agreement.round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

"""Score the BiDoze release bundles on EESM19 through each bundle's own prediction path.

Usage (worktree root, ``PYTHONPATH=core/src:bidoze/src``):
    python bidoze/experiments/new_cohorts/eesm19/score_releases.py OUT_DIR

``OUT_DIR`` must hold ``bundles/adult-f2x3-v0.2.0`` and ``bundles/adult-f2x1-v0.2.0``; it receives
per_night.csv, summary.csv, paired.csv, scorer2.csv, alignment.txt and tables.md.

Every night is scored by ``arcasleep.predict.load_predictor(bundle).predict(accel, origin=<first scored
epoch>, ggir_sleep_filter=False)`` and compared on epochs with scorer-1 label >= 0. A predicted gap is a
miss: for sleep/wake it counts as the wrong class, for 4-stage it is its own class.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path("/media/eric/Extreme SSD/pisces2-data/eesm19-pisces")
MAIN = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results")
BUNDLES = {
    "Pediatric v0.1.1": MAIN / "pediatric-weaver-all-2026-10-23/training/final/bundle",
    "Adult v0.1.0": MAIN / "bidoze-adult-v0.1.0/bundle",
    "Adult F2 x3 v0.2.0": "bundles/adult-f2x3-v0.2.0",
    "Adult F2 x1 v0.2.0": "bundles/adult-f2x1-v0.2.0",
}
STAGE_INDEX = {"wake": 0, "light": 1, "deep": 2, "rem": 3, "gap": 4}
FOUR = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3}
EPOCH_S = 30.0
KAPPAS = ["kappa_sleep_wake", "kappa_4stage"]
METRICS = KAPPAS + ["sleep_sensitivity", "wake_specificity", "tst_error_min", "deep_pct_pred", "deep_pct_psg",
                    "abs_err_light_min", "abs_err_deep_min", "abs_err_rem_min"]


def kappa(a, b):
    classes = np.union1d(a, b)
    if classes.size < 2:
        return float("nan")
    t = np.array([[np.sum((a == i) & (b == j)) for j in classes] for i in classes], float)
    n = t.sum()
    pe = float(t.sum(0) @ t.sum(1)) / n**2
    return (np.trace(t) / n - pe) / (1 - pe) if pe < 1 else float("nan")


def read_pairs(path):
    return np.loadtxt(path, ndmin=2)


def read_accel(path):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64).to_numpy()


def four_stage(y):
    return np.array([FOUR[int(v)] for v in y])


def night_metrics(y, pred):
    """y: scorer-1 codes (>= 0); pred: stage index 0-3 or 4 for a gap."""
    truth4 = four_stage(y)
    sleep = truth4 > 0
    p_sleep = (pred > 0) & (pred < 4)
    p_wake = pred == 0
    # sleep/wake kappa: a gap is the wrong class
    p_sw = np.where(pred == 4, np.where(sleep, 0, 1), (pred > 0).astype(int))
    m = dict(kappa_sleep_wake=kappa(sleep.astype(int), p_sw), kappa_4stage=kappa(truth4, pred),
             sleep_sensitivity=float(np.mean(p_sleep[sleep])) if sleep.any() else np.nan,
             wake_specificity=float(np.mean(p_wake[~sleep])) if (~sleep).any() else np.nan,
             tst_error_min=0.5 * float(p_sleep.sum() - sleep.sum()),
             deep_pct_pred=100 * float(np.mean(pred[p_sleep] == 2)) if p_sleep.any() else np.nan,
             deep_pct_psg=100 * float(np.mean(truth4[sleep] == 2)))
    for name, c in (("light", 1), ("deep", 2), ("rem", 3)):
        m[f"abs_err_{name}_min"] = 0.5 * abs(float(np.sum(pred == c) - np.sum(truth4 == c)))
    return m


def aligned(path, times):
    """Rows of ``path`` at the PSG epoch starts (same clock); -1 where absent."""
    other = read_pairs(path)
    idx = np.round(other[:, 0] / EPOCH_S).astype(int)
    out = np.full(len(times), -1)
    k = np.round(times / EPOCH_S).astype(int)
    pos = {v: i for i, v in enumerate(k)}
    for i, v in zip(idx, other[:, 1].astype(int)):
        if i in pos:
            out[pos[i]] = v
    return out


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return float(v.mean()) if n else np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def main(out):
    from arcasleep.predict import load_predictor
    out = Path(out)
    manifest = json.loads((RAW / "conversion_manifest.json").read_text())
    reasons = {}
    for n in manifest:
        k = n["status"] if n["status"] == "converted" else f"dropped: {n.get('reason')}"
        reasons[k] = reasons.get(k, 0) + 1
    ids = sorted(n["id"] for n in manifest if n["status"] == "converted")
    predictors = {name: load_predictor(out / p if not Path(p).is_absolute() else p) for name, p in BUNDLES.items()}
    print({k: type(v).__name__ for k, v in predictors.items()}, flush=True)
    rows, s2rows, mix = [], [], np.zeros(3)
    align = [f"manifest nights: {len(manifest)}; " + "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))]
    for sid in ids:
        psg = read_pairs(RAW / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        times, y = psg[:, 0], psg[:, 1].astype(int)
        accel = read_accel(RAW / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        scored = y >= 0
        pos = {int(round(t / EPOCH_S)): i for i, t in enumerate(times)}
        for name, pr in predictors.items():
            res = pr.predict(accel, origin=float(times[0]), ggir_sleep_filter=False, subject_id=sid)
            labels = np.asarray(res.raw_labels, dtype=object)
            starts = np.asarray(res.epoch_start_s)
            if len(align) == 1:
                align.append(f"{sid} [{name}]: PSG epochs {len(times)}, predicted epochs {len(starts)}; "
                             f"PSG first start {times[0]:.1f} s, last {times[-1]:.1f} s; predicted first start "
                             f"{starts[0]:.1f} s, last {starts[-1]:.1f} s; start times equal for common epochs: "
                             f"{bool(np.allclose(starts[:min(len(starts), len(times))], times[:min(len(starts), len(times))]))}")
            pred = np.full(len(times), 4)  # an epoch the predictor did not output is a miss
            for s, lab in zip(starts, labels):
                i = pos.get(int(round(s / EPOCH_S)))
                if i is not None:
                    pred[i] = STAGE_INDEX[str(lab)]
            row = dict(model=name, night=sid, participant=sid.split("_")[0], n_epochs=int(scored.sum()),
                       n_pred_epochs=len(starts), n_gap_scored=int((pred[scored] == 4).sum()))
            row.update(night_metrics(y[scored], pred[scored]))
            rows.append(row)
        s4 = four_stage(y[scored])
        mix += [np.sum(s4 == 1), np.sum(s4 == 2), np.sum(s4 == 3)]
        s2 = aligned(RAW / "scorer2" / f"{sid}_scorer2.out", times)
        ok = scored & (s2 >= 0)
        a, b = y[ok], s2[ok]
        s2rows.append(dict(model="Scorer 2 vs scorer 1", night=sid, participant=sid.split("_")[0], n_epochs=int(ok.sum()),
                           kappa_sleep_wake=kappa((four_stage(a) > 0).astype(int), (four_stage(b) > 0).astype(int)),
                           kappa_4stage=kappa(four_stage(a), four_stage(b))))
        print("scored", sid, flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_night.csv", index=False)
    s2 = pd.DataFrame(s2rows)
    s2.to_csv(out / "scorer2.csv", index=False)
    (out / "alignment.txt").write_text("\n".join(align) + "\n")

    summ = []
    for name in list(BUNDLES) + ["Scorer 2 vs scorer 1"]:
        t = per[per.model == name] if name in BUNDLES else s2
        for metric in (METRICS if name in BUNDLES else KAPPAS):
            m, lo, hi = ci(t[metric])
            r = dict(model=name, metric=metric, mean=m, ci_low=lo, ci_high=hi, n_nights=int(t[metric].notna().sum()),
                     n_participants=t.participant.nunique())
            if metric in KAPPAS:
                pm, plo, phi = ci(t.groupby("participant")[metric].mean())
                r.update(participant_mean=pm, participant_ci_low=plo, participant_ci_high=phi)
            summ.append(r)
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "summary.csv", index=False)

    paired = []
    ref = "Adult F2 x3 v0.2.0"
    for other in ("Adult v0.1.0", "Pediatric v0.1.1", "Adult F2 x1 v0.2.0"):
        a = per[per.model == ref].set_index("night")
        b = per[per.model == other].set_index("night")
        for metric in KAPPAS:
            d = (a[metric] - b[metric]).dropna()
            p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            paired.append(dict(comparison=f"{ref} vs {other}", metric=metric, n_nights=len(d), mean_diff=d.mean(),
                               wilcoxon_p=p))
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)

    f = lambda x, d=3: "n/a" if pd.isna(x) else f"{x:.{d}f}".replace("-", "−")
    g = lambda r, d=3: f"{f(r['mean'], d)} [{f(r.ci_low, d)}, {f(r.ci_high, d)}]"
    sget = lambda name, metric: summary[(summary.model == name) & (summary.metric == metric)].iloc[0]
    cols = [("κ sleep/wake", "kappa_sleep_wake", 3), ("κ 4-stage", "kappa_4stage", 3), ("sleep sens.", "sleep_sensitivity", 3),
            ("wake spec.", "wake_specificity", 3), ("TST error (min)", "tst_error_min", 1),
            ("deep % pred", "deep_pct_pred", 1), ("abs err light (min)", "abs_err_light_min", 1),
            ("abs err deep (min)", "abs_err_deep_min", 1), ("abs err REM (min)", "abs_err_rem_min", 1)]
    n_n, n_p = per.night.nunique(), per.participant.nunique()
    L = [f"# EESM19: BiDoze release bundles ({n_n} nights, {n_p} participants)", "",
         "Per-night mean [95% CI over nights]. Epochs with scorer-1 label ≥ 0; a predicted gap is a miss. "
         "`ggir_sleep_filter=False`.", "",
         "| model | " + " | ".join(c[0] for c in cols) + " | κ sleep/wake, participant mean | κ 4-stage, participant mean |",
         "|:---|" + ":---|" * (len(cols) + 2)]
    for name in BUNDLES:
        L.append(f"| {name} | " + " | ".join(g(sget(name, k), d) for _, k, d in cols) + " | "
                 + " | ".join(f"{f(sget(name, k).participant_mean)} [{f(sget(name, k).participant_ci_low)}, {f(sget(name, k).participant_ci_high)}]" for k in KAPPAS) + " |")
    L += ["", f"PSG deep sleep as % of sleep, mean over nights: {g(sget('Adult v0.1.0', 'deep_pct_psg'), 1)}.", "",
          "## Scorer 2 vs scorer 1 (human agreement)", "", "| κ | mean over nights [95% CI] | participant mean [95% CI] |", "|:---|:---|:---|"]
    for lab, k in (("sleep/wake", "kappa_sleep_wake"), ("4-stage", "kappa_4stage")):
        r = sget("Scorer 2 vs scorer 1", k)
        L.append(f"| {lab} | {g(r)} (n = {r.n_nights}) | {f(r.participant_mean)} [{f(r.participant_ci_low)}, {f(r.participant_ci_high)}] |")
    L += ["", "## Paired Wilcoxon over nights", "", "| comparison | κ | n nights | mean difference | p |", "|:---|:---|:---|:---|:---|"]
    for _, r in paired.iterrows():
        L.append(f"| {r.comparison} | {r.metric} | {r.n_nights} | {f(r.mean_diff)} | {r.wilcoxon_p:.3g} |")
    tot = mix.sum()
    L += ["", "## PSG stage mix (pooled over scored sleep epochs)", "",
          f"light {100*mix[0]/tot:.1f}%, deep {100*mix[1]/tot:.1f}%, REM {100*mix[2]/tot:.1f}%", "",
          "## Conversion", "", align[0]]
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])

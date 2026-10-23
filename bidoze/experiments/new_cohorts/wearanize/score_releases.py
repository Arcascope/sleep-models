"""Score the BiDoze release bundles on Wearanize+ through each bundle's own prediction path.

Usage (worktree root, ``PYTHONPATH=core/src:bidoze/src``):
    python bidoze/experiments/new_cohorts/wearanize/score_releases.py OUT_DIR [--fold-dirs DIR ...] [--train-cache DIR]

``OUT_DIR`` receives per_night.csv, summary.csv, paired.csv, scorer2.csv, alignment.txt and tables.md.

Every night is scored by ``arcasleep.predict.load_predictor(bundle).predict(accel, origin=<first PSG epoch>,
ggir_sleep_filter=False)`` and compared on epochs with scorer-1 label >= 0. A predicted gap is a miss: for
sleep/wake it counts as the wrong class, for 4-stage it is its own class.

Human references (scorer 2 and U-Sleep, side files of the cohort) are compared with scorer 1 on epochs both
score. Scorer 1 epoch ``i`` starts at ``times[i]``; each side-file epoch is paired with the nearest scorer-1
start (within 15 s), and a shift ``s`` pairs scorer-1 epoch ``i`` with the side epoch ``s`` epochs after its
nearest one. Shift 0 is the nearest pairing; the "best of 5" variant picks, per night, the shift in
-2..+2 with the highest 4-stage kappa (biased upward: it selects the best of 5 on the scored data).

``--fold-dirs`` (optional, for the F2 x3 v0.2.0 cross-validation folds, which train on other Wearanize+
folds) scores each fold's held-out Wearanize+ recordings from its ``predictions/`` seed mean, with the
fold's divisor (``phase_divisor``, which needs ``--train-cache``: a directory holding
``12Hz_log_PSD_aligned/`` with the fold's training links).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path("/media/eric/Extreme SSD/pisces2-data/wearanize-pisces")
MAIN = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results")
BUNDLES = {
    "Pediatric v0.1.1": MAIN / "pediatric-weaver-all-2026-10-23/training/final/bundle",
    "Adult v0.1.0": MAIN / "bidoze-adult-v0.1.0/bundle",
}
FOLD_MODEL = "Adult F2 x3 v0.2.0 (held-out folds)"
STAGE_INDEX = {"wake": 0, "light": 1, "deep": 2, "rem": 3, "gap": 4}
FOUR = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3}
EPOCH_S = 30.0
SHIFTS = [-2, -1, 0, 1, 2]
SIDES = {"Scorer 2": ("scorer2", "scorer2"), "U-Sleep": ("usleep", "usleep")}
KAPPAS = ["kappa_sleep_wake", "kappa_4stage"]
METRICS = KAPPAS + ["sleep_sensitivity", "wake_specificity", "tst_error_min", "deep_pct_pred", "deep_pct_psg",
                    "abs_err_light_min", "abs_err_deep_min", "abs_err_rem_min"]


def read_pairs(path):
    return np.loadtxt(path, ndmin=2)


def read_accel(path):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64).to_numpy()


def four_stage(y):
    return np.array([FOUR[int(v)] for v in y])


def kappa(a, b):
    classes = np.union1d(a, b)
    if classes.size < 2:
        return float("nan")
    t = np.array([[np.sum((a == i) & (b == j)) for j in classes] for i in classes], float)
    n = t.sum()
    pe = float(t.sum(0) @ t.sum(1)) / n**2
    return (np.trace(t) / n - pe) / (1 - pe) if pe < 1 else float("nan")


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


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return float(v.mean()) if n else np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)




def side_pairing(path, times):
    """Side-file epochs as arrays on the scorer-1 grid: nearest side-epoch index (-1 if none within 15 s)."""
    other = read_pairs(path)
    near = np.abs(other[:, 0][None, :] - times[:, None]).argmin(1)
    ok = np.abs(other[near, 0] - times) <= EPOCH_S / 2
    return other[:, 1].astype(int), np.where(ok, near, -1)


def shifted_side(values, near, shift):
    """Side stage code paired with each scorer-1 epoch at ``shift``; -1 where there is none."""
    j = np.where(near >= 0, near + shift, -1)
    ok = (j >= 0) & (j < len(values))
    return np.where(ok, values[np.clip(j, 0, len(values) - 1)], -1)


def reference_metrics(y, side):
    """Metrics of a side scorer against scorer 1 on epochs both score (``side`` in cache codes, -1 absent)."""
    ok = (y >= 0) & (side >= 0)
    m = night_metrics(y[ok], four_stage(side[ok]))
    return m, int(ok.sum())


def predicted_on_grid(starts, labels, times):
    t0 = float(times[0])
    pos = {int(round((t - t0) / EPOCH_S)): i for i, t in enumerate(times)}
    pred = np.full(len(times), 4)  # an epoch the predictor did not output is a miss
    for s, lab in zip(starts, labels):
        i = pos.get(int(round((s - t0) / EPOCH_S)))
        if i is not None:
            pred[i] = STAGE_INDEX[str(lab)]
    return pred


def fold_rows(fold_dirs, train_cache, loaded):
    """Held-out Wearanize+ nights of each fold phase, scored by its seed-mean output and its divisor."""
    from arcasleep_bidoze.adult import hybrid_calls
    from arcasleep_bidoze.release import ADULT_CV_COHORTS, phase_divisor, seed_mean_recordings
    rows = []
    for fold in map(Path, fold_dirs):
        divisor, _, _ = phase_divisor(fold, train_cache)
        dataset = json.loads((fold / "config/cv.json").read_text())["test_sets"][ADULT_CV_COHORTS["Wearanize+"]]
        seeds = [d for d in sorted((fold / "predictions").iterdir()) if d.is_dir()]
        stems = {f.stem[len(dataset) + 1:] for f in seeds[0].glob(f"{dataset}_*.csv")}
        for sid in sorted(stems):
            night = next((n for n in loaded if sid == n or sid.endswith(n)), None)
            if night is None:
                continue
            frames = [pd.read_csv(s / f"{dataset}_{sid}.csv") for s in seeds]
            gap = np.mean([f["gap_head"].to_numpy(float) for f in frames], axis=0)
            stage = np.mean([f[["wake", "light", "deep", "rem"]].to_numpy(float) * (1 - f["gap_head"].to_numpy(float))[:, None]
                             for f in frames], axis=0)
            mean = np.column_stack([stage, gap])
            calls = hybrid_calls(mean, mean, divisor)
            times, y = loaded[night]
            pred = predicted_on_grid(frames[0]["start_s"].to_numpy(float), calls, times)
            scored = y >= 0
            row = dict(model=FOLD_MODEL, night=night, participant=night.split("_")[0], n_epochs=int(scored.sum()),
                       n_pred_epochs=len(frames[0]), n_gap_scored=int((pred[scored] == 4).sum()))
            row.update(night_metrics(y[scored], pred[scored]))
            rows.append(row)
    return rows


def main(out, fold_dirs=(), train_cache=None):
    from arcasleep.predict import load_predictor
    out = Path(out)
    manifest = json.loads((RAW / "conversion_manifest.json").read_text())
    reasons = {}
    for n in manifest:
        k = n["status"] if n["status"] == "converted" else f"dropped: {n.get('reason')}"
        reasons[k] = reasons.get(k, 0) + 1
    ids = sorted(n["id"] for n in manifest if n["status"] == "converted")
    predictors = {name: load_predictor(p) for name, p in BUNDLES.items()}
    rows, ref_rows, mix = [], [], np.zeros(3)
    loaded = {}
    align = [f"manifest nights: {len(manifest)}; " + "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))]
    mismatches = 0
    for sid in ids:
        psg = read_pairs(RAW / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        times, y = psg[:, 0], psg[:, 1].astype(int)
        loaded[sid] = (times, y)
        accel = read_accel(RAW / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        scored = y >= 0
        for name, pr in predictors.items():
            res = pr.predict(accel, origin=float(times[0]), ggir_sleep_filter=False, subject_id=sid)
            labels = np.asarray(res.raw_labels, dtype=object)
            starts = np.asarray(res.epoch_start_s)
            k = min(len(starts), len(times))
            same = bool(np.allclose(starts[:k], times[:k]))
            mismatches += (len(starts) != len(times)) or not same
            align.append(f"{sid} [{name}]: PSG epochs {len(times)}, predicted {len(starts)}; PSG first start "
                         f"{times[0]:.1f} s, last {times[-1]:.1f} s; predicted first {starts[0]:.1f} s, last "
                         f"{starts[-1]:.1f} s; starts equal on common epochs: {same}")
            pred = predicted_on_grid(starts, labels, times)
            row = dict(model=name, night=sid, participant=sid.split("_")[0], n_epochs=int(scored.sum()),
                       n_pred_epochs=len(starts), n_gap_scored=int((pred[scored] == 4).sum()))
            row.update(night_metrics(y[scored], pred[scored]))
            rows.append(row)
        s4 = four_stage(y[scored])
        mix += [np.sum(s4 == 1), np.sum(s4 == 2), np.sum(s4 == 3)]
        for label, (folder, suffix) in SIDES.items():
            path = RAW / folder / f"{sid}_{suffix}.out"
            if not path.exists():
                continue
            values, near = side_pairing(path, times)
            per_shift = {}
            for s in SHIFTS:
                m, n_ok = reference_metrics(y, shifted_side(values, near, s))
                per_shift[s] = (m, n_ok)
            best = max(SHIFTS, key=lambda s: (np.nan_to_num(per_shift[s][0]["kappa_4stage"], nan=-9), -abs(s)))
            for variant, s in (("shift 0", 0), ("best of 5 shifts", best)):
                m, n_ok = per_shift[s]
                ref_rows.append(dict(model=f"{label} vs scorer 1, {variant}", reference=label, variant=variant,
                                     night=sid, participant=sid.split("_")[0], shift=s, n_epochs=n_ok, **m))
        print("scored", sid, flush=True)
    if fold_dirs:
        rows += fold_rows(fold_dirs, train_cache, loaded)
    align.insert(1, f"nights where a bundle's epoch count or starts differ from the PSG: {mismatches}")
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_night.csv", index=False)
    refs = pd.DataFrame(ref_rows)
    refs.to_csv(out / "scorer2.csv", index=False)
    (out / "alignment.txt").write_text("\n".join(align) + "\n")

    model_names = list(per.model.unique())
    ref_names = list(refs.model.unique())
    summ = []
    for name in model_names + ref_names:
        t = per[per.model == name] if name in model_names else refs[refs.model == name]
        for metric in METRICS:
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
    a = per[per.model == "Adult v0.1.0"].set_index("night")
    b = per[per.model == "Pediatric v0.1.1"].set_index("night")
    for metric in KAPPAS:
        d = (a[metric] - b[metric]).dropna()
        p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
        paired.append(dict(comparison="Adult v0.1.0 − Pediatric v0.1.1", metric=metric, n_nights=len(d),
                           mean_diff=d.mean(), wilcoxon_p=p))
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)

    f = lambda x, d=3: "n/a" if pd.isna(x) else f"{x:.{d}f}".replace("-", "−")
    g = lambda r, d=3: f"{f(r['mean'], d)} [{f(r.ci_low, d)}, {f(r.ci_high, d)}]"
    sget = lambda name, metric: summary[(summary.model == name) & (summary.metric == metric)].iloc[0]
    main_cols = [("κ sleep/wake", "kappa_sleep_wake", 3, True), ("κ 4-stage", "kappa_4stage", 3, True),
                 ("sleep sens.", "sleep_sensitivity", 3, False), ("wake spec.", "wake_specificity", 3, False),
                 ("TST error (min)", "tst_error_min", 1, True), ("deep % of predicted sleep", "deep_pct_pred", 1, False)]
    err_cols = [("abs err light (min)", "abs_err_light_min"), ("abs err deep (min)", "abs_err_deep_min"),
                ("abs err REM (min)", "abs_err_rem_min")]
    n_n, n_p = per.night.nunique(), per.participant.nunique()
    order = [n for n in ref_names if n.startswith("Scorer 2")] + [n for n in ref_names if n.startswith("U-Sleep")] + model_names
    label = lambda n: n + (" (biased upward: best of 5 per night)" if "best of 5" in n else "")
    L = [f"# Wearanize+: BiDoze release bundles ({n_n} nights, {n_p} participants)", "",
         "Mean over nights [95% CI] for κ and TST error; other columns are the mean over nights. Epochs with scorer-1 "
         "label ≥ 0; a predicted gap is a miss. `ggir_sleep_filter=False`. References are compared on epochs both "
         "score.", "",
         "| row | " + " | ".join(c[0] for c in main_cols) + " | κ sleep/wake, participant mean | κ 4-stage, participant mean |",
         "|:---|" + ":---|" * (len(main_cols) + 2)]
    for name in order:
        cells = [g(sget(name, k), d) if ci_ else f(sget(name, k)["mean"], d) for _, k, d, ci_ in main_cols]
        pm = [f"{f(sget(name, k).participant_mean)} [{f(sget(name, k).participant_ci_low)}, {f(sget(name, k).participant_ci_high)}]"
              for k in KAPPAS]
        L.append(f"| {label(name)} | " + " | ".join(cells) + " | " + " | ".join(pm) + " |")
    L += ["", "## Absolute stage-time error (min), mean over nights", "",
          "| row | " + " | ".join(c[0] for c in err_cols) + " |", "|:---|" + ":---|" * len(err_cols)]
    for name in order:
        L.append(f"| {label(name)} | " + " | ".join(f(sget(name, k)["mean"], 1) for _, k in err_cols) + " |")
    L += ["", "## Chosen shift (epochs) in the best-of-5 variant", "", "| reference | " + " | ".join(f"{s:+d}".replace("-", "−") for s in SHIFTS) + " |",
          "|:---|" + ":---|" * len(SHIFTS)]
    for lab in SIDES:
        t = refs[(refs.reference == lab) & (refs.variant == "best of 5 shifts")]
        L.append(f"| {lab} (n = {len(t)} nights) | " + " | ".join(str(int((t["shift"] == s).sum())) for s in SHIFTS) + " |")
    L += ["", "## Paired Wilcoxon over nights", "", "| comparison | κ | n nights | mean difference | p |", "|:---|:---|:---|:---|:---|"]
    for _, r in paired.iterrows():
        L.append(f"| {r.comparison} | {r.metric} | {r.n_nights} | {f(r.mean_diff)} | {r.wilcoxon_p:.3g} |")
    tot = mix.sum()
    L += ["", "## PSG stage mix (pooled over scored sleep epochs)", "",
          f"light {100*mix[0]/tot:.1f}%, deep {100*mix[1]/tot:.1f}%, REM {100*mix[2]/tot:.1f}%", "",
          "## Conversion", "", align[0],
          f"Participants are identified by the id prefix before `_` (e.g. `sub005` in `sub005_s1`); every converted night is session `s1`."]
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fold-dirs", nargs="+", default=[])
    ap.add_argument("--train-cache")
    args = ap.parse_args()
    if args.fold_dirs and not args.train_cache:
        ap.error("--fold-dirs needs --train-cache")
    main(args.out, args.fold_dirs, args.train_cache)

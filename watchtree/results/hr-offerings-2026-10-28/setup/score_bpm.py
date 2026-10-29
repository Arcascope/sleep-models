"""Score the Experiment 4 arms out of fold and write per-cohort tables.

Cohorts (rows of every table)
  weaver_garmin, weaver_fitbit  primary; out of fold by wv0..wv4 models (Weaver persons folded 5 ways, seed 0)
  wearanize, sleepaccel         secondary; out of fold by the BiDoze fold models (the nights BiDoze holds out)
  AAUWSS and BIDSleep were not scored in Experiment 3b and are not scored here.
Conditions: ``hr`` (inputs as recorded) and ``hr_removed`` (every column starting ``hr`` or ``bpm_`` is NaN at test time).
``baseline_nohr`` (62 columns, trained without HR) is the explicit accelerometer-only fallback; it has no HR inputs, so
its only condition is hr_removed.
Metrics per night, averaged over the 3 seeds, then over nights; 95% CIs from a participant-cluster bootstrap
(5,000 draws, fixed seed); paired differences against a reference arm on the same nights with a Wilcoxon p over nights.
No multiplicity correction. Decodings: native (argmax of raw scores) and offsets (training-side share-matching offsets).
"""
import argparse
from scipy import stats
from bpm_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--phases", default=",".join(FOLD_PHASES + WV_PHASES))
ap.add_argument("--arms", default=",".join(ARMS))
ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
ap.add_argument("--draws", type=int, default=BOOT_DRAWS)
a = ap.parse_args()
PHASES = a.phases.split(",")
ARM_LIST = a.arms.split(",")
SEED_LIST = list(map(int, a.seeds.split(",")))
STEP = C.EPOCH_S / 60
METRICS = ["kappa4", "rem_sens", "deep_sens", "kappa_sw", "sens", "spec", "tst_bias", "tst_mae"]
COHORTS = ["weaver_garmin", "weaver_fitbit", "wearanize", "sleepaccel"]
tab = table()
tests = test_rows(tab)
tests = {c: t[t.phase.isin(PHASES)].reset_index(drop=True) for c, t in tests.items()}
for c, t in tests.items():
    print(c, "test nights", t.subject.nunique(), "epochs", len(t), flush=True)


def conds_of(arm):
    return ["as_recorded"]


def night_rows(t, pred):
    rows = []
    y = t.y4.to_numpy()
    for subject, ix in t.groupby("subject").indices.items():
        yy, pp = y[ix], pred[ix]
        sleep, wake = yy > 0, yy == 0
        rows.append({"subject": subject,
                     "kappa4": cv._kappa(yy, pp), "kappa_sw": cv._kappa(yy > 0, pp > 0),
                     "rem_sens": np.mean(pp[yy == 3] == 3) if (yy == 3).any() else np.nan,
                     "deep_sens": np.mean(pp[yy == 2] == 2) if (yy == 2).any() else np.nan,
                     "sens": np.mean(pp[sleep] > 0) if sleep.any() else np.nan,
                     "spec": np.mean(pp[wake] == 0) if wake.any() else np.nan,
                     "tst_bias": ((pp > 0).sum() - sleep.sum()) * STEP})
    return pd.DataFrame(rows)


per_night, importance = [], []
for cohort in COHORTS:
    t = tests[cohort]
    if t.empty:
        continue
    persons = {s: cv.person_of(cohort, s) for s in t.subject.unique()}
    for arm in ARM_LIST:
        feats0 = ARMS[arm]
        for cond in conds_of(arm):
            acc = {"native": [], "offset": []}
            for seed in SEED_LIST:
                nat = np.zeros(len(t), int); off = np.zeros(len(t), int)
                for phase in t.phase.unique():
                    m = (t.phase == phase).to_numpy()
                    b, o, feats = load_arm_model(phase, arm, seed)
                    X = t.loc[m, feats].copy()
                    r = b.predict(X.to_numpy(np.float32), num_iteration=b.best_iteration, raw_score=True)
                    nat[m] = r.argmax(1); off[m] = (r + o).argmax(1)
                    if False:
                        g = pd.Series(b.feature_importance("gain"), index=feats)
                        importance.append({"phase": phase, "arm": arm, "seed": seed, "gain_share_bpm": g[[f for f in feats if f.startswith("bpm_")]].sum() / g.sum()})
                for dec, p in (("native", nat), ("offset", off)):
                    n = night_rows(t, p); n["seed"] = seed; acc[dec].append(n)
            for dec, lst in acc.items():
                nm = pd.concat(lst).drop(columns="seed").groupby("subject", as_index=False).mean()
                nm["tst_mae"] = nm.tst_bias.abs()
                nm.insert(0, "decoding", dec); nm.insert(0, "cond", cond); nm.insert(0, "arm", arm); nm.insert(0, "cohort", cohort)
                nm["person"] = nm.subject.map(persons)
                per_night.append(nm)
    print("scored", cohort, flush=True)
pn = pd.concat(per_night, ignore_index=True)
pn.to_csv(OUT / "per_night.csv", index=False)
if importance:
    pd.DataFrame(importance).groupby(["arm"]).gain_share_bpm.mean().to_csv(OUT / "importance_bpm_gain_share.csv")


def cluster_boot(num, den, draws, seed):
    num, den = np.asarray(num, float), np.asarray(den, float)
    point = num.sum() / den.sum()
    idx = np.random.default_rng(seed).integers(0, len(num), (draws, len(num)))
    return point, *np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5])


summary, paired = [], []
for (cohort, dec), g in pn.groupby(["cohort", "decoding"]):
    for (arm, cond), ga in g.groupby(["arm", "cond"]):
        row = {"cohort": cohort, "decoding": dec, "arm": arm, "cond": cond, "n_nights": len(ga)}
        for m in METRICS:
            ps = ga.dropna(subset=[m]).groupby("person")[m].agg(["sum", "size"])
            row[m], row[m + "_lo"], row[m + "_hi"] = cluster_boot(ps["sum"], ps["size"], a.draws, BOOT_SEED)
        summary.append(row)
    AC = "as_recorded"
    PAIRS = [("hr_bpm", AC, "hr", AC), ("hr_bpm_ahr", AC, "hr", AC), ("hr_bpm_ahr", AC, "hr_bpm", AC),
             ("nohr_ahr", AC, "nohr", AC), ("hr", AC, "nohr", AC), ("hr_bpm", AC, "nohr", AC), ("hr_bpm_ahr", AC, "nohr", AC),
             ("hr_bpm_ahr", AC, "nohr_ahr", AC), ("hr_bpm", AC, "nohr_ahr", AC)]
    for arm, cond, ref, rcond in PAIRS:
        ga = g[(g.arm == arm) & (g.cond == cond)].set_index("subject")
        gr = g[(g.arm == ref) & (g.cond == rcond)].set_index("subject")
        both = ga.index.intersection(gr.index)
        if both.empty:
            continue
        for m in METRICS:
            diff = (ga.loc[both, m] - gr.loc[both, m]).dropna()
            if diff.empty:
                continue
            ps = pd.DataFrame({"d": diff, "person": ga.loc[diff.index, "person"]}).groupby("person").d.agg(["sum", "size"])
            pt, lo, hi = cluster_boot(ps["sum"], ps["size"], a.draws, BOOT_SEED)
            p = stats.wilcoxon(diff).pvalue if len(diff) > 5 and (diff != 0).any() else np.nan
            paired.append({"cohort": cohort, "decoding": dec, "arm": arm, "cond": cond, "ref_arm": ref, "ref_cond": rcond, "metric": m,
                           "n_nights": len(diff), "ref_mean": gr.loc[diff.index, m].mean(), "arm_mean": ga.loc[diff.index, m].mean(),
                           "diff": pt, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p})
sm, pr = pd.DataFrame(summary), pd.DataFrame(paired)
sm.to_csv(OUT / "summary.csv", index=False)
pr.to_csv(OUT / "paired.csv", index=False)

# coverage of the added columns on the test rows
cov = []
for cohort, t in tests.items():
    for c in BF.ALL_COLS + AHR_COLS:
        cov.append({"cohort": cohort, "column": c, "non_nan_share": float(t[c].notna().mean())})
pd.DataFrame(cov).pivot(index="column", columns="cohort", values="non_nan_share").reindex(BF.ALL_COLS + AHR_COLS).to_csv(OUT / "column_coverage_after.csv")

# reproduction of section 15: arms hr / nohr against the old run's per-night scores (offsets decoding)
old = pd.read_csv(OLD_EXP / "per_night.csv")
old = old[(old.decoding == "offset") & (((old.arm == "baseline") & (old.cond == "hr")) | ((old.arm == "baseline_nohr") & (old.cond == "hr_removed")))]
old = old.assign(arm=old.arm.map({"baseline": "hr", "baseline_nohr": "nohr"}))
new = pn[(pn.decoding == "offset") & pn.arm.isin(["hr", "nohr"])]
chk = new.merge(old, on=["cohort", "arm", "subject"], suffixes=("", "_old"))
rep = []
for (c, arm), g in chk.groupby(["cohort", "arm"]):
    rep.append({"cohort": c, "arm": arm, "nights": len(g), **{f"max_abs_diff_{m}": float((g[m] - g[m + "_old"]).abs().max()) for m in METRICS}})
pd.DataFrame(rep).to_csv(OUT / "reproduce_section15.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
v = sm[sm.decoding == "offset"][["cohort", "arm", "cond", "n_nights"] + METRICS]
print(v.round(3).to_string())
print(pd.DataFrame(rep).round(6).to_string())

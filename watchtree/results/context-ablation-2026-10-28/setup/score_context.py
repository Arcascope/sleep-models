"""Score the context arms on BiDoze's held-out Wearanize+ and SleepAccel folds (offsets decoding, 3 seeds).

Two epoch sets per arm: shared (epochs the shipped (120, 45) arm can score) and own (every scored epoch with the arm's full
context). Per-night metrics are computed per seed, then seed-averaged, then averaged over nights; confidence intervals are
participant-cluster bootstrap (5,000 draws). Paired differences are against the shipped arm on the shared epochs.
WASO = wake minutes between the first and last PSG sleep epoch of the scored epoch set (predicted: same span of PSG sleep).
"""
import argparse
from scipy import stats
from context_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default=",".join(FOLDS))
ap.add_argument("--arms", default=",".join(arm_name(*a) for a in ARMS))
ap.add_argument("--cfgs", default="hr,nohr")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--draws", type=int, default=BOOT_DRAWS)
ap.add_argument("--out", default=str(EXP))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(exist_ok=True)
FOLD_LIST, CFG_LIST = a.folds.split(","), a.cfgs.split(",")
SEED_LIST = list(map(int, a.seeds.split(",")))
ARM_LIST = a.arms.split(",")
STEP = C.EPOCH_S / 60
COHORTS = {"wearanize": "BiDozeWearanizeFold", "sleepaccel": "BiDozeSleepAccelFold"}
REF = arm_name(*CURRENT)
METRICS = ["kappa4", "kappa_sw", "sens", "spec", "rem_sens", "deep_sens", "tst_bias", "tst_mae", "waso_bias", "waso_mae"]

tabs_all = full_tables()
test = {}
for fold in FOLD_LIST:
    for cohort, prefix in COHORTS.items():
        test[(fold, cohort)] = set(test_ids(fold).get(f"{prefix}{fold[-1]}", []))


def arm_test_table(cohort, arm):
    b, f = next(x for x in ARMS if arm_name(*x) == arm)
    nights, full = tabs_all[cohort]
    in_test = set().union(*[test[(fo, cohort)] for fo in FOLD_LIST]) & nights
    t = arm_table(cohort, full, in_test, b, f)
    t = t[t.scored].reset_index(drop=True)
    fold_of = {s: fo for fo in FOLD_LIST for s in test[(fo, cohort)]}
    t["fold"] = t.subject.map(fold_of)
    return t


def night_metrics(y, p):
    sleep, wake = y > 0, y == 0
    out = {"kappa_sw": cv._kappa(sleep, p > 0), "kappa4": cv._kappa(y, p),
           "sens": np.mean(p[sleep] > 0) if sleep.any() else np.nan,
           "spec": np.mean(p[wake] == 0) if wake.any() else np.nan,
           "rem_sens": np.mean(p[y == 3] == 3) if (y == 3).any() else np.nan,
           "deep_sens": np.mean(p[y == 2] == 2) if (y == 2).any() else np.nan,
           "tst_bias": ((p > 0).sum() - sleep.sum()) * STEP}
    if sleep.any():
        i = np.flatnonzero(sleep); span = slice(i[0], i[-1] + 1)
        out["waso_bias"] = ((p[span] == 0).sum() - (y[span] == 0).sum()) * STEP
    else:
        out["waso_bias"] = np.nan
    return out


per_night, coverage, repro = [], [], []
for cohort in COHORTS:
    persons_of = lambda s: cv.person_of(cohort, s)
    for arm in ARM_LIST:
        tab = arm_test_table(cohort, arm)
        # coverage over scored epochs of the kept test nights
        pos = tab.groupby("subject").cumcount().to_numpy()
        first2h = pos < 240
        for scope, m in (("all PSG", np.ones(len(tab), bool)), ("first 2 h", first2h)):
            c = tab.ctx_arm.to_numpy()[m]
            coverage.append({"cohort": cohort, "arm": arm, "scope": scope, "scored_epochs": int(m.sum()),
                             "scorable_epochs": int(c.sum()), "share": c.mean(), "n_nights": tab.subject.nunique()})
        for cfg in CFG_LIST:
            per = {"shared": [], "own": []}
            for seed in SEED_LIST:
                r = np.zeros((len(tab), 4)); pred = np.zeros(len(tab), int)
                for fold in FOLD_LIST:
                    m = (tab.fold == fold).to_numpy()
                    if not m.any():
                        continue
                    bst, off, feats = load_arm_model(fold, arm, cfg, seed)
                    r[m] = raw(bst, tab[m], feats); pred[m] = (r[m] + off).argmax(1)
                if arm == REF:
                    for fold in FOLD_LIST:  # the shipped arm against the exp3b baseline models, same rows
                        m = (tab.fold == fold).to_numpy()
                        _, off, _ = load_arm_model(fold, arm, cfg, seed)
                        meta = json.load(open(EXP3B_MODELS / f"{fold}_baseline_{cfg}_s{seed}.json"))
                        b3 = lgb.Booster(model_file=str(EXP3B_MODELS / f"{fold}_baseline_{cfg}_s{seed}.txt"))
                        r3 = b3.predict(tab[m][meta["features"]].to_numpy(np.float32), num_iteration=meta["best_iteration"], raw_score=True)
                        repro.append({"cohort": cohort, "cfg": cfg, "seed": seed, "fold": fold, "n_epochs": int(m.sum()),
                                      "max_abs_raw_diff": float(np.abs(r[m] - r3).max()),
                                      "same_offsets": bool(np.allclose(off, meta["offsets"])),
                                      "pred_agreement": float(((r[m] + off).argmax(1) == (r3 + np.array(meta["offsets"])).argmax(1)).mean())})
                y = tab.y4.to_numpy()
                for setname, mask in (("shared", tab.ctx.to_numpy()), ("own", tab.ctx_arm.to_numpy())):
                    sub = tab[mask]; ys, ps = y[mask], pred[mask]
                    rows = []
                    for subject, ix in sub.groupby("subject").indices.items():
                        d = night_metrics(ys[ix], ps[ix]); d["subject"] = subject; d["seed"] = seed
                        d["n_epochs"] = len(ix); rows.append(d)
                    per[setname].append(pd.DataFrame(rows))
            for setname, lst in per.items():
                st = pd.concat(lst)
                nm = st.drop(columns="seed").groupby("subject", as_index=False).mean()
                nm["tst_mae"] = nm.tst_bias.abs(); nm["waso_mae"] = nm.waso_bias.abs()
                nm["person"] = nm.subject.map(persons_of)
                nm.insert(0, "set", setname); nm.insert(0, "offering", cfg); nm.insert(0, "arm", arm); nm.insert(0, "cohort", cohort)
                per_night.append(nm)
            print(cohort, arm, cfg, "done", flush=True)

pn = pd.concat(per_night, ignore_index=True)
pn.to_csv(OUT / "per_night.csv", index=False)
cov = pd.DataFrame(coverage).drop_duplicates(["cohort", "arm", "scope"])
cov.to_csv(OUT / "coverage.csv", index=False)
if repro:
    rp = pd.DataFrame(repro); rp.to_csv(OUT / "current_vs_exp3b_baseline.csv", index=False)
    print(rp.groupby(["cohort", "cfg"]).agg(max_raw_diff=("max_abs_raw_diff", "max"), min_agree=("pred_agreement", "min")))


def cluster_boot(num, den, draws, seed):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (draws, len(num)))
    v = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), *np.percentile(v, [2.5, 97.5])


summary, paired = [], []
for (cohort, cfg, setname), g in pn.groupby(["cohort", "offering", "set"]):
    ref = g[g.arm == REF].set_index("subject")
    for arm, ga in g.groupby("arm", sort=False):
        ga = ga.set_index("subject")
        row = {"cohort": cohort, "offering": cfg, "set": setname, "arm": arm, "n_nights": len(ga),
               "n_persons": ga.person.nunique(), "epochs": int(ga.n_epochs.sum())}
        for m in METRICS:
            ps = ga[[m, "person"]].dropna().groupby("person")[m].agg(["sum", "size"])
            row[m], row[m + "_lo"], row[m + "_hi"] = cluster_boot(ps["sum"], ps["size"], a.draws, BOOT_SEED)
        summary.append(row)
        if setname != "shared" or arm == REF or ref.empty:
            continue
        both = ga.index.intersection(ref.index)
        for m in METRICS:
            diff = (ga.loc[both, m] - ref.loc[both, m]).dropna()
            if diff.empty:
                continue
            ps = pd.DataFrame({"d": diff, "person": ga.loc[diff.index, "person"]}).groupby("person").d.agg(["sum", "size"])
            pt, lo, hi = cluster_boot(ps["sum"], ps["size"], a.draws, BOOT_SEED)
            paired.append({"cohort": cohort, "offering": cfg, "arm": arm, "metric": m, "n_nights": len(diff),
                           "current": ref.loc[diff.index, m].mean(), "arm_mean": ga.loc[diff.index, m].mean(),
                           "diff": pt, "ci_lo": lo, "ci_hi": hi})
sm = pd.DataFrame(summary)
sm.to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(paired).to_csv(OUT / "paired_vs_current.csv", index=False)
pd.set_option("display.width", 250)
print(sm[sm.set == "shared"][["cohort", "offering", "arm", "n_nights", "kappa_sw", "kappa4", "sens", "spec", "tst_bias", "tst_mae"]].round(3).to_string())
print(cov.pivot_table(index=["cohort", "scope"], columns="arm", values="share").round(3).to_string())

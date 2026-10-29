"""Out-of-fold tree predictions of every arm and both offerings on Wearanize+ (whole recordings) and SleepAccel.
Each night is scored by the fold model that held its person out. Saves, per cohort, offset and native argmax calls per seed
and the seed-mean offset softmax of the baseline arms. Reads the fitted models of ablation-exp3b; fits nothing."""
import time
import lightgbm as lgb
from tree_common import *

t0 = time.time()
OUTD = RR / "local"
for cohort, prefix in COHORTS.items():
    full = cohort_table(cohort)
    n = len(full)
    off = {}; nat = {}; prob = {}
    folds = {}
    for k in range(5):
        ids = set(test_ids(f"fold{k}").get(f"{prefix}{k}", []))
        folds[k] = np.flatnonzero(full.subject.isin(ids).to_numpy())
    cov = np.zeros(n, bool)
    for ix in folds.values():
        cov[ix] = True
    print(cohort, "rows", n, "nights", full.subject.nunique(), "test-covered rows", int(cov.sum()), flush=True)
    for arm in ARMS:
        for cfg in CFG_LIST:
            o = np.full((3, n), -1, np.int8); nt = np.full((3, n), -1, np.int8); pr = np.full((3, n, 4), np.nan, np.float32)
            for k, ix in folds.items():
                sub = full.iloc[ix]
                for si, seed in enumerate(SEEDS):
                    b, offs, feats = load_arm_model(f"fold{k}", arm, cfg, seed)
                    r = b.predict(sub[feats].to_numpy(np.float32), num_iteration=b.best_iteration, raw_score=True, num_threads=THREADS)
                    nt[si, ix] = r.argmax(1); o[si, ix] = (r + offs).argmax(1)
                    pr[si, ix] = softmax(r + offs)
            off[(arm, cfg)] = o; nat[(arm, cfg)] = nt
            if arm == "baseline":
                prob[cfg] = pr.mean(0)
        print(cohort, arm, f"{time.time()-t0:.0f}s", flush=True)
    d = {"subject": full.subject.to_numpy().astype(str), "epoch": full.epoch.to_numpy(), "covered": cov}
    for (arm, cfg), v in off.items():
        d[f"off|{arm}|{cfg}"] = v; d[f"nat|{arm}|{cfg}"] = nat[(arm, cfg)]
    for cfg, v in prob.items():
        d[f"prob|baseline|{cfg}"] = v
    np.savez_compressed(OUTD / f"tree_{cohort}.npz", **d)
full_meta = {c: None for c in COHORTS}
print("TREE_DONE", flush=True)

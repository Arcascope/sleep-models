"""Retrain the WatchTree stager on one BiDoze fold for context arms; the recipe of ablation-exp3b fit_ablate.py.

Training rows are scored epochs with full context for the arm (backward reach B, forward reach F)."""
import argparse, time
from context_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--arms", default=",".join(arm_name(*a) for a in ARMS))
ap.add_argument("--cfgs", default="hr,nohr")
ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
t0 = time.time()
tabs = full_tables()
fit_ids, eval_ids = split_ids(a.phase)
fit_s = {x["subject_id"] for x in fit_ids}
eval_s = {x["subject_id"] for x in eval_ids}
(EXP / "models").mkdir(parents=True, exist_ok=True)
for arm in a.arms.split(","):
    b, f = next(x for x in ARMS if arm_name(*x) == arm)
    todo = [(cfg, s) for cfg in a.cfgs.split(",") for s in map(int, a.seeds.split(","))
            if not model_path(a.phase, arm, cfg, s, "json").exists()]
    if not todo:
        print(a.phase, arm, "done already", flush=True); continue
    parts = []
    for name, (nights, full) in tabs.items():
        t = arm_table(name, full, nights, b, f)
        parts.append(t[t.scored & t.ctx_arm])
    table = pd.concat(parts, ignore_index=True)
    train = table[table.subject.isin(fit_s)].reset_index(drop=True)
    valid = table[table.subject.isin(eval_s)].reset_index(drop=True)
    alltrain = pd.concat([train, valid], ignore_index=True)
    print(a.phase, arm, "train", len(train), "eval", len(valid), f"{time.time()-t0:.0f}s", flush=True)
    for cfg, seed in todo:
        feats = arm_features(CFGS[cfg], b, f)
        bst = fit_eval(train, valid, feats, seed, a.threads)
        cv.quantise_leaves(bst)
        ref = raw(bst, valid.iloc[:2000], feats)
        bst.save_model(str(model_path(a.phase, arm, cfg, seed, "txt")), num_iteration=bst.best_iteration)
        off = fit_offsets_share(alltrain.y4.to_numpy(), inner_scores(alltrain, feats, seed, a.threads))
        b2 = lgb.Booster(model_file=str(model_path(a.phase, arm, cfg, seed, "txt")))
        assert np.allclose(b2.predict(valid.iloc[:2000][feats].to_numpy(np.float32), raw_score=True), ref, atol=1e-6)
        json.dump({"best_iteration": int(bst.best_iteration), "offsets": off.tolist(), "n_features": len(feats),
                   "features": feats}, open(model_path(a.phase, arm, cfg, seed, "json"), "w"))
        print(a.phase, arm, cfg, seed, "features", len(feats), "rounds", bst.best_iteration, "offsets", np.round(off, 2),
              f"{time.time()-t0:.0f}s", flush=True)

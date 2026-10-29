"""Fit the Experiment 4 arms on one phase (fold0..4 or wv0..4): 3 seeds, int8 leaves, training-side share-matching offsets.

Same recipe as the Experiment 3b fit (cv-bidoze-scheme fit_eval / inner_scores / fit_offsets_share). Resumable: a model
whose json exists is skipped. No feature selection: every column of an arm goes to LightGBM; early stopping uses the
phase's BiDoze eval subjects (training side); the offsets use inner person folds of the training subjects only.
"""
import argparse, json, time
from bpm_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--arms", default=",".join(ARMS))
ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
t0 = time.time()
train, valid = phase_pools(table(), a.phase)
alltrain = pd.concat([train, valid], ignore_index=True)
print(a.phase, "train", len(train), "nights", train.subject.nunique(), "eval", len(valid), "nights", valid.subject.nunique(),
      f"load {time.time()-t0:.0f}s", flush=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)
for arm in a.arms.split(","):
    feats = ARMS[arm]
    for seed in map(int, a.seeds.split(",")):
        if model_path(a.phase, arm, seed, "json").exists():
            print(a.phase, arm, seed, "cached", flush=True)
            continue
        ts = time.time()
        b = fit_eval(train, valid, feats, seed, a.threads)
        cv.quantise_leaves(b)
        ref = raw(b, valid.iloc[:2000], feats)
        b.save_model(str(model_path(a.phase, arm, seed, "txt")), num_iteration=b.best_iteration)
        off = fit_offsets_share(alltrain.y4.to_numpy(), inner_scores(alltrain, feats, seed, a.threads))
        json.dump({"best_iteration": int(b.best_iteration), "offsets": off.tolist(), "n_features": len(feats), "features": feats,
                   "n_train_epochs": len(train), "n_eval_epochs": len(valid)}, open(model_path(a.phase, arm, seed, "json"), "w"))
        b2, _, _ = load_arm_model(a.phase, arm, seed)
        assert np.allclose(raw(b2, valid.iloc[:2000], feats), ref, atol=1e-6), "saved model differs"
        print(a.phase, arm, seed, "features", len(feats), "rounds", b.best_iteration, "offsets", np.round(off, 2),
              f"fit {time.time()-ts:.0f}s total {time.time()-t0:.0f}s", flush=True)

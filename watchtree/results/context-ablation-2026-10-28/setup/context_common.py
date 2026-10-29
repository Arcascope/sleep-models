"""Shared paths and helpers of the context ablation (reuses cv-bidoze-scheme-2026-10-27/setup/common.py unchanged)."""
import json, os, sys
from pathlib import Path

OLD_SETUP = "/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree/results/cv-bidoze-scheme-2026-10-27/setup"
sys.path.insert(0, OLD_SETUP)
import common as old  # noqa: E402
from common import *  # noqa: E402,F401,F403
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from context_features import ARMS, CURRENT, arm_features, arm_name, ctx_mask  # noqa: E402

EXP = Path(os.environ.get("CTX_OUT", "/home/eric/Engineering/Work/autopisces/sleep-models-tree-context/watchtree/results/context-ablation-2026-10-28"))
FOLDS = [f"fold{k}" for k in range(5)]
COHORT_NAMES = ["wearanize", "sleepaccel", "weaver_garmin", "weaver_fitbit"]
BOOT_DRAWS = 5000
BOOT_SEED = 20261028
EXP3B_MODELS = Path("/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree/results/ablation-exp3b-2026-10-28/models")


def model_path(phase, arm, cfg, seed, suffix):
    return EXP / "models" / f"{phase}_{arm}_{cfg}_s{seed}.{suffix}"


def load_arm_model(phase, arm, cfg, seed):
    meta = json.load(open(model_path(phase, arm, cfg, seed, "json")))
    b = lgb.Booster(model_file=str(model_path(phase, arm, cfg, seed, "txt")))
    b.best_iteration = meta["best_iteration"]
    return b, np.array(meta["offsets"]), meta["features"]


def full_tables():
    """name -> (kept-night complete-context selection of the shipped arm, full table with all epochs, kept nights)."""
    out = {}
    for name in COHORT_NAMES:
        sel, full = old.load(name)
        out[name] = (set(sel.subject.unique()), full)
    return out


def with_arm_columns(full, nights, b, f):
    """Rows of kept nights with the arm's columns substituted and ``ctx_arm`` set; scored and arm-context rows only if asked."""
    t = full[full.subject.isin(nights)].copy()
    n_rows = t.groupby("subject").epoch.transform("size").to_numpy()
    t["ctx_arm"] = ctx_mask(t.epoch.to_numpy(), n_rows, b, f)
    p = EXP / "features" / f"{arm_name(b, f)}_{{}}.parquet"
    return t, p


def arm_table(name, full, nights, b, f):
    t = full[full.subject.isin(nights)].copy()
    n_rows = t.groupby("subject").epoch.transform("size").to_numpy()
    t["ctx_arm"] = ctx_mask(t.epoch.to_numpy(), n_rows, b, f)
    p = EXP / "features" / f"{arm_name(b, f)}_{name}.parquet"
    if p.exists():
        d = pd.read_parquet(p).set_index(["subject", "epoch"])
        idx = pd.MultiIndex.from_arrays([t.subject.to_numpy(), t.epoch.to_numpy()])
        for c in d.columns:
            t[c] = d[c].reindex(idx).to_numpy()
    return t.reset_index(drop=True)

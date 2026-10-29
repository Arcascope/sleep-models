"""Shared paths, tables, arms and helpers of Experiment 4 (BPM variability features).

Reuses cv-bidoze-scheme-2026-10-27/setup/common.py (tables, BiDoze fit/eval/test subject lists, LightGBM fit, offsets)
unchanged, as Experiment 3b does.

Phases
  fold0..fold4  BiDoze v0.2.0 folds: fit/eval subjects as BiDoze; test = Wearanize+ and SleepAccel nights of the fold
  wv0..wv4      Weaver out-of-fold: BiDoze fold k's fit/eval subjects minus the Weaver persons (children) of Weaver
                person-fold k; test = those persons' Garmin and Fitbit nights.
BiDoze folds never hold Weaver out (it is in the training data of all five), so Weaver out-of-fold predictions need
their own models. The tree CV (arcasleep_watchtree.cv) assigns folds with cv.person_folds(sorted persons, 5, seed 0)
within each cohort; here the persons of both Weaver devices are folded together so that a child's other device is never
in training (the pool contains each child's other devices).
"""
import json, os, sys
from pathlib import Path

OLD_SETUP = "/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree/results/cv-bidoze-scheme-2026-10-27/setup"
sys.path.insert(0, OLD_SETUP)
import common as old  # noqa: E402
from common import *  # noqa: E402,F401,F403
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bpm_features as BF  # noqa: E402

EXP = Path("/home/eric/Engineering/Work/autopisces/sleep-models-hr-offerings/watchtree/results/hr-offerings-2026-10-28")
OLD_EXP = WT / "results/bpm-features-exp4-2026-10-28"
OUT = Path(os.environ.get("EXP4_OUT", EXP))
SMOKE = os.environ.get("EXP4_SMOKE", "0") == "1"
FEAT_DIR = OUT / "features"
EXP3B = WT / "results/ablation-exp3b-2026-10-28"
SRCS = ("wearanize", "sleepaccel", "weaver_garmin", "weaver_fitbit")
FOLD_PHASES = [f"fold{k}" for k in range(5)]
WV_PHASES = [f"wv{k}" for k in range(5)]
BOOT_DRAWS = 5000
BOOT_SEED = 20261028
AHR_COLS = ["ahr_bpm", "ahr_q"] + BF.AHR_COLS_5_10
ARMS = {
    "nohr": NOHR,
    "nohr_ahr": NOHR + AHR_COLS,
    "hr": FINAL,
    "hr_bpm": FINAL + BF.COLS_5_10,
    "hr_bpm_ahr": FINAL + BF.COLS_5_10 + AHR_COLS,
}
OLD_ARM = {"nohr": "baseline_nohr", "hr": "baseline"}  # arms whose feature columns are unchanged from section 15


def is_hr_input(col: str) -> bool:
    return col.startswith("hr") or col.startswith("bpm_")


def model_path(phase, arm, seed, suffix):
    return OUT / "models" / f"{phase}_{arm}_s{seed}.{suffix}"


def load_arm_model(phase, arm, seed):
    meta = json.load(open(model_path(phase, arm, seed, "json")))
    b = lgb.Booster(model_file=str(model_path(phase, arm, seed, "txt")))
    b.best_iteration = meta["best_iteration"]
    return b, np.array(meta["offsets"]), meta["features"]


_CACHE: dict = {}


def source_tables() -> dict[str, pd.DataFrame]:
    """Selected (kept nights, complete-context, scored) rows of each source, without BPM columns."""
    if "src" not in _CACHE:
        _CACHE["src"] = {n: load(n)[0].assign(src=n) for n in SRCS}
    return _CACHE["src"]


def weaver_person_folds() -> dict[str, int]:
    s = source_tables()
    persons = sorted(set(s["weaver_garmin"].person) | set(s["weaver_fitbit"].person))
    return cv.person_folds(persons, 5, 0)


def smoke_subjects() -> set[str]:
    """A few nights of fold0 / wv0: 4 fit + 2 eval per source, 3 test per source and phase (BiDoze test nights)."""
    s = source_tables()
    wf = weaver_person_folds()
    fit, ev = split_ids("fold0")
    fit_s, ev_s = {x["subject_id"] for x in fit}, {x["subject_id"] for x in ev}
    t0 = test_ids("fold0")
    test_of = {"wearanize": set(t0["BiDozeWearanizeFold0"]), "sleepaccel": set(t0["BiDozeSleepAccelFold0"])}
    chosen = set()
    for n, t in s.items():
        subs = sorted(t.subject.unique())
        if n.startswith("weaver"):
            held = {x for x in subs if wf[x.split("_")[0]] == 0}
            chosen |= set(sorted(held)[:3])
            rest = [x for x in subs if x not in held]
        else:
            test = [x for x in subs if x in test_of[n]]
            chosen |= set(test[:3])
            rest = [x for x in subs if x not in test_of[n]]
        chosen |= set([x for x in rest if x in fit_s][:4]) | set([x for x in rest if x in ev_s][:2])
    return chosen


def bpm_table(name: str, subjects=None) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((FEAT_DIR / name).glob("*.parquet"))]
    t = pd.concat(parts, ignore_index=True)
    return t if subjects is None else t[t.subject.isin(subjects)]


def table() -> pd.DataFrame:
    """All rows of the four sources with the 25 BPM columns joined on (subject, epoch); smoke keeps a few nights."""
    if "table" in _CACHE:
        return _CACHE["table"]
    parts = []
    keep = smoke_subjects() if SMOKE else None
    for n, t in source_tables().items():
        if keep is not None:
            t = t[t.subject.isin(keep)]
        b = bpm_table(n, set(t.subject))
        m = t.merge(b.rename(columns={"y4": "y4_bpm"}), on=["subject", "epoch"], how="left")
        a = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "features_ahr" / n).glob("*.parquet"))], ignore_index=True)
        m = m.merge(a.drop(columns="y4"), on=["subject", "epoch"], how="left")
        assert m.ahr_valid_10m.notna().all(), f"{n}: scored epochs without accelerometer-HR features"
        assert m.bpm_valid_10m.notna().all(), f"{n}: scored epochs without BPM features ({m.bpm_valid_10m.isna().sum()})"
        assert (m.y4.to_numpy() == m.y4_bpm.to_numpy()).all(), f"{n}: labels differ from the BPM build"
        parts.append(m.drop(columns="y4_bpm"))
    _CACHE["table"] = pd.concat(parts, ignore_index=True)
    return _CACHE["table"]


def phase_pools(tab: pd.DataFrame, phase: str):
    """(train, valid) rows of a phase: BiDoze fold-k fit / eval subjects; wv phases drop the held-out Weaver persons."""
    k = int(phase[-1])
    fit, ev = split_ids(f"fold{k}")
    fit_s, ev_s = {x["subject_id"] for x in fit}, {x["subject_id"] for x in ev}
    if phase.startswith("wv"):
        wf = weaver_person_folds()
        held = tab.src.str.startswith("weaver") & (tab.person.map(wf) == k)
        tab = tab[~held]
    return (tab[tab.subject.isin(fit_s)].reset_index(drop=True), tab[tab.subject.isin(ev_s)].reset_index(drop=True))


def test_rows(tab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Out-of-fold test rows per cohort, with a ``phase`` column naming the model that scores each row."""
    wf = weaver_person_folds()
    out = {}
    for n in ("wearanize", "sleepaccel"):
        prefix = {"wearanize": "BiDozeWearanizeFold", "sleepaccel": "BiDozeSleepAccelFold"}[n]
        parts = []
        for k in range(5):
            ids = set(test_ids(f"fold{k}").get(f"{prefix}{k}", []))
            p = tab[(tab.src == n) & tab.subject.isin(ids)].copy()
            p["phase"] = f"fold{k}"
            parts.append(p)
        out[n] = pd.concat(parts, ignore_index=True)
    for n in ("weaver_garmin", "weaver_fitbit"):
        p = tab[tab.src == n].copy()
        p["phase"] = "wv" + p.person.map(wf).astype(int).astype(str)
        out[n] = p.reset_index(drop=True)
    return out

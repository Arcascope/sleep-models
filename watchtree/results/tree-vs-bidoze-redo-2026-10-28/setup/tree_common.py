"""Shared paths for the redo: tree prediction cache (all arms, both offerings) and loaders. No fitting; the fitted models of
ablation-exp3b-2026-10-28 (5 folds x 8 arms x {hr, nohr} x 3 seeds) are read only."""
import os, sys
from pathlib import Path
WT = Path("/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree")
ABL = WT / "results/ablation-exp3b-2026-10-28"
os.environ["EXP3B_OUT"] = str(ABL)          # model directory of the ablation run (read only)
sys.path.insert(0, str(ABL / "setup"))
import combined_groups  # noqa: F401,E402   (adds the combined arms to groups.arms before ablate_common imports it)
from ablate_common import *  # noqa: F401,F403,E402
import ablate_common as _ac  # noqa: E402

RR = WT / "results/tree-vs-bidoze-redo-2026-10-28"
ARMS = list(arms().keys())
CFG_LIST = ["hr", "nohr"]
THREADS = 10
OOF = Path("/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders/bidoze/results/decisions-redo-2026-10-28/oof")
TSTI = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/tst-investigation-2026-10-28")
COHORTS = {"wearanize": "BiDozeWearanizeFold", "sleepaccel": "BiDozeSleepAccelFold"}


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def cohort_table(cohort):
    """Full table (whole recording for Wearanize+) with kept-night flag, sorted by subject, epoch."""
    sel, full = load(cohort)
    full = full.sort_values(["subject", "epoch"]).reset_index(drop=True)
    full["kept"] = full.subject.isin(set(sel.subject))
    return full

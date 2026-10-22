"""Write the Wearanize+ 5-fold split plan, ``config/splits/wearanize_5fold.json``.

Every participant has one night. Nights are ordered by wake share (the fraction of scored epochs
labelled wake, ties by id) and cut into blocks of 5; within each block a seeded permutation deals
the 5 nights to the 5 folds, so each fold holds one night from every block: 92 nights give folds of 18 or 19 nights with similar wake shares.

Usage: python make_plan.py <cache root> [<output plan>]
(``cache root`` holds ``12Hz_log_PSD_aligned/`` with the ``wearanize`` nights.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from arcasleep.data.cache import discover_cached_subjects
from arcasleep.data.splits import SplitPlan, fold_dataset_name, write_plan

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
CACHE_NAME = "12Hz_log_PSD_aligned"
COHORT = "wearanize"
N_SPLITS = 5
SEED = 0


def wake_share(path: Path) -> float:
    with np.load(path) as z:
        y = z["y"].ravel()
    scored = y >= 0
    return float(np.mean(y[scored] == 0))


def stratified_folds(shares: dict[str, float], n_splits: int, seed: int) -> list[list[str]]:
    ordered = sorted(shares, key=lambda sid: (shares[sid], sid))
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for start in range(0, len(ordered), n_splits):
        block = ordered[start:start + n_splits]
        for sid, fold in zip(block, rng.permutation(n_splits)[:len(block)]):
            folds[int(fold)].append(sid)
    return folds


def main(cache_root: Path, output: Path) -> None:
    subjects = discover_cached_subjects([COHORT], cache_root / CACHE_NAME)[COHORT]
    shares = {s.subject_id: wake_share(s.path) for s in subjects}
    folds = stratified_folds(shares, N_SPLITS, SEED)
    plan = SplitPlan(
        cohort=COHORT, n_splits=N_SPLITS, seed=SEED,
        folds={fold_dataset_name(COHORT, i): sorted(ids) for i, ids in enumerate(folds)},
        provenance={"cache_dir": f"feature_cache/{CACHE_NAME}", "subject_count": len(shares),
                    "one_night_per_participant": True,
                    "stratified_by": "wake share, blocks of 5 nights ordered by wake share",
                    "data_digest": None})
    plan.validate()
    write_plan(plan, output)
    for i, ids in enumerate(folds):
        print(f"fold{i}: {len(ids)} nights, mean wake share {np.mean([shares[s] for s in ids]):.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(),
         Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else BIDOZE / "config/splits/wearanize_5fold.json")

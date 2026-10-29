"""Tuning (training) and test nights per cohort: BiDoze fold 0. Training = the cohort's nights of fold-0 fit + eval subjects
(Weaver: nights of persons not in Weaver person-fold 0); test = the fold-0 test subjects (Weaver: persons in person-fold 0).
Also fold-of-night for every night (BiDoze test fold; Weaver person fold) for step 4."""
import pandas as pd
from bpm_common import source_tables, split_ids, test_ids, weaver_person_folds, SRCS


def night_folds() -> pd.DataFrame:
    s = source_tables(); wf = weaver_person_folds(); rows = []
    for n in ("wearanize", "sleepaccel"):
        pre = {"wearanize": "BiDozeWearanizeFold", "sleepaccel": "BiDozeSleepAccelFold"}[n]
        for k in range(5):
            for sub in test_ids(f"fold{k}").get(f"{pre}{k}", []):
                rows.append((n, sub, k))
    for n in ("weaver_garmin", "weaver_fitbit"):
        for sub, p in s[n][["subject", "person"]].drop_duplicates().itertuples(index=False):
            rows.append((n, sub, wf[p]))
    f = pd.DataFrame(rows, columns=["cohort", "subject", "fold"]).drop_duplicates()
    return f


def split0() -> dict[str, tuple[set, set]]:
    s = source_tables(); f = night_folds(); out = {}
    fit, ev = split_ids("fold0")
    train_b = {x["subject_id"] for x in fit} | {x["subject_id"] for x in ev}
    for n in SRCS:
        subs = set(s[n].subject)
        fo = f[f.cohort == n].set_index("subject").fold
        test = {x for x in subs if x in fo.index and fo[x] == 0}
        if n.startswith("weaver"):
            train = subs - test
        else:
            train = {x for x in subs if x in train_b and x not in test}
        out[n] = (train, test)
    return out

if __name__ == "__main__":
    for n, (a, b) in split0().items():
        print(n, len(a), len(b))

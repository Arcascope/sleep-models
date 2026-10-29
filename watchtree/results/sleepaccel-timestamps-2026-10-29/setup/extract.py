"""Writes the raw (cleaned, native-rate) timestamps of every SleepAccel night to $N/private/<id>.npy (uncommitted) and private/folds.csv
(id, in-fold flag). The ids stay in private/ only."""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from arcasleep.data.prepare import load_dataset
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT
from loadnight import load_with_rows
from splits import night_folds
sys.path.insert(0, os.environ["OLD"] + "/setup")
from build_est import subject_ids
warnings.simplefilter("ignore")
N = Path(os.environ["N"]); out = N / "private"
c = COHORTS["sleepaccel"]; ds = load_dataset(DATA_ROOT / c.directory)
ids, rep = subject_ids(c)
f = night_folds(); infold = set(f[f.cohort == "sleepaccel"].subject)
rows = []
for s in ids:
    try:
        _, r = load_with_rows("sleepaccel", s, rep[s], ds, None)
    except Exception as e:
        print("failed", e); rows.append((rep[s], rep[s] in infold, 0)); continue
    np.save(out / f"{rep[s]}.npy", r[:, 0]); rows.append((rep[s], rep[s] in infold, r.shape[0]))
pd.DataFrame(rows, columns=["id", "in_fold", "n"]).to_csv(out / "folds.csv", index=False)
print(len(rows))

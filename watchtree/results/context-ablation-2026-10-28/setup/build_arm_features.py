"""Write the per-arm changed feature columns of every cohort table to features/<arm>_<cohort>.parquet."""
import sys, time
from context_common import *
from context_features import derive

(EXP / "features").mkdir(exist_ok=True)
tabs = full_tables()
for b, f in ARMS:
    for name, (nights, full) in tabs.items():
        t0 = time.time()
        d = derive(full[full.subject.isin(nights)], b, f)
        if d.shape[1] > 2:
            d.to_parquet(EXP / "features" / f"{arm_name(b, f)}_{name}.parquet", index=False)
        print(arm_name(b, f), name, d.shape, f"{time.time()-t0:.0f}s", flush=True)

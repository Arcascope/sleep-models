"""Per-column coverage of the bpm columns (share of scored epochs, non-NaN) before (section 15 tables) and after, per cohort;
also confirms the unchanged cohorts' feature tables are identical to the section 15 tables."""
import numpy as np, pandas as pd
from bpm_common import OUT, OLD_EXP, SRCS
import bpm_features as BF

cov, same = [], []
for n in SRCS:
    new = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "features" / n).glob("*.parquet"))], ignore_index=True)
    old = pd.concat([pd.read_parquet(p) for p in sorted((OLD_EXP / "features" / n).glob("*.parquet"))], ignore_index=True)
    for c in BF.ALL_COLS:
        cov.append({"cohort": n, "column": c, "before": old[c].notna().mean(), "after": new[c].notna().mean()})
    m = old.merge(new, on=["subject", "epoch"], suffixes=("_o", "_n"))
    mx = max(float(np.nanmax(np.abs(m[c + "_o"] - m[c + "_n"]))) if m[c + "_o"].notna().any() or m[c + "_n"].notna().any() else 0.0 for c in BF.ALL_COLS)
    nanchg = int(sum(((m[c + "_o"].isna()) != (m[c + "_n"].isna())).sum() for c in BF.ALL_COLS))
    same.append({"cohort": n, "rows": len(m), "max_abs_diff": mx, "nan_pattern_changes": nanchg})
c = pd.DataFrame(cov)
c.to_csv(OUT / "bpm_column_coverage_before_after.csv", index=False)
pd.DataFrame(same).to_csv(OUT / "bpm_features_unchanged_check.csv", index=False)
w = c[c.cohort == "sleepaccel"].set_index("column")[["before", "after"]]
print(w.round(3).to_string()); print(pd.DataFrame(same).to_string())

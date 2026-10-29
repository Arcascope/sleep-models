"""Step 2: MAE vs device HR of every estimator in the stillest decile of PSG-sleep epochs, train (BiDoze fold-0 fit/eval) and
import os
test (fold-0 test) participants. Constant predictor = median device HR (same window) of the training sleep epochs."""
import sys
import numpy as np, pandas as pd
from common2 import *
d = load_est(tuple(os.environ.get("COHS", ",".join(COH)).split(",")))
cs = combos(d)
rows = []
for n, g in d.groupby("cohort"):
    for W in (30, 60, 120):
        const = g[(g.split == "train") & g.sleep][f"dev{W}"].median()
        for c in [c for c in cs if c.endswith(f".{W}")]:
            ty, me, wh, _ = c.split(".")
            for sp in ("train", "test"):
                s = g[(g.split == sp) & g.still]
                r = mae_row(s[f"est.{c}"], s[f"dev{W}"], const)
                # per-night median device HR (oracle, not causal)
                nm = s.groupby("subject")[f"dev{W}"].transform("median")
                ok = s[f"est.{c}"].notna() & s[f"dev{W}"].notna()
                r["night_median_mae"] = (s[f"dev{W}"] - nm)[ok].abs().mean()
                r["cov"] = float((s[f"est.{c}"].notna() & s[f"dev{W}"].notna()).mean())
                rows.append(dict(cohort=n, W=W, type=ty, method=me, whiten=int(wh), split=sp, const=const, **r))
T = pd.DataFrame(rows)
T.to_csv(R / "step2_all_combos.csv", index=False)
pd.set_option("display.width", 250)
for n in T.cohort.unique():
    t = T[(T.cohort == n)]
    a = t[t.split == "train"].set_index(["W", "type", "method", "whiten"])
    b = t[t.split == "test"].set_index(["W", "type", "method", "whiten"])
    j = a[["n", "mae", "const_mae"]].join(b[["n", "mae", "const_mae", "within5"]], lsuffix="_train", rsuffix="_test").sort_values("mae_train")
    print(n); print(j.head(12).round(2).to_string())

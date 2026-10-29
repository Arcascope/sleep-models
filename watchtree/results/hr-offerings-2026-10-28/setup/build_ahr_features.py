"""Per-epoch accelerometer-derived HR features from the stride-5 s series (window length W chosen by --window).

ahr_bpm, ahr_q: the estimate and quality of the window that ends at the scored epoch's end.
ahr_<name>_<5|10>m: the bpm_5_10 columns computed on the estimated series (bpm_features.night_features, prefix "ahr",
cohort "accelhr": dt 5 s, mid band 0.04-0.10 Hz). The series is every estimable window; there is no quality gate.
Sample at window end e is stamped e - 1 ms so the epoch ending at e includes it (data before e only).
"""
import argparse
import numpy as np
import pandas as pd
from bpm_common import OUT, SRCS
import bpm_features as BF

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, required=True)
a = ap.parse_args()
W = a.window
for name in SRCS:
    out = OUT / "features_ahr" / name
    out.mkdir(parents=True, exist_ok=True)
    for sp in sorted((OUT / "accelhr" / name / "series").glob("*.parquet")):
        night = sp.stem
        dst = out / f"{night}.parquet"
        if dst.exists():
            continue
        s = pd.read_parquet(sp)
        e = pd.read_parquet(OUT / "accelhr" / name / "epochs" / sp.name)
        ne = (len(s) - 1) // 6 + 1
        t = 30.0 + 5.0 * s.j.to_numpy() - 1e-3
        f = BF.night_features(t, s[f"bpm{W}"].to_numpy(float), ne, e.epoch.to_numpy(), "accelhr", "ahr")
        f.insert(0, "ahr_q", e[f"q{W}"].to_numpy(np.float32))
        f.insert(0, "ahr_bpm", e[f"est{W}"].to_numpy(np.float32))
        f.insert(0, "y4", e.y4.to_numpy())
        f.insert(0, "subject", night)
        f.to_parquet(dst, index=False)
    print(name, len(list(out.glob("*.parquet"))), flush=True)

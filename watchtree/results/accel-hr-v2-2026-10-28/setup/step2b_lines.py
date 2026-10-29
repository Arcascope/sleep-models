"""Pulse line at the device HR in individual stillest-decile windows (no averaging of spectra over windows).
Per window: whitened PSD (running median over +-1.5 Hz removed) at the device-HR frequency f (max over +-1 bin) against control
frequencies 0.8 f and 1.25 f (and at 2f against 1.6 f and 2.5 f). Reported: median ratio over windows and the share of windows in
which the line at f exceeds both controls (chance = 1/3 for max of three equal-law values)."""
import glob
import numpy as np, pandas as pd
from common2 import R, COH
import ahr2 as A
from splits import split0
sp = split0(); rows = []
for n in COH:
    for W in (60, 120):
        for ty in ("sum", "mag"):
            res = {"f": [], "f2": []}
            for f in sorted(glob.glob(str(R / "still" / n / "*.npz"))):
                z = np.load(f)
                key = f"S_{ty}_{W}"
                if key not in z: continue
                S = z[key].astype(float); fr = 0.4 + np.arange(S.shape[1]) / W; dev = z[f"dev{W}"]
                ok = np.isfinite(S).all(axis=1) & np.isfinite(dev) & (dev >= 40) & (dev <= 180)
                if not ok.any(): continue
                Sw = A.whiten(S[ok], fr); hz = dev[ok] / 60.0
                sub = f.split("/")[-1][:-4]
                tag = "train" if sub in sp[n][0] else ("test" if sub in sp[n][1] else "none")
                def at(h):
                    out = np.full(len(h), np.nan)
                    for m in (-1, 0, 1):
                        v = A._interp(Sw, fr, (h + m * (fr[1] - fr[0]))[:, None])[:, 0] if False else None
                    # vectorised per-row interpolation at (h + m df)
                    best = np.full(len(h), -np.inf)
                    for m in (-1, 0, 1):
                        pos = (h + m * (fr[1] - fr[0]) - fr[0]) / (fr[1] - fr[0])
                        i0 = np.clip(np.floor(pos).astype(int), 0, fr.size - 2); w = np.clip(pos - i0, 0, 1)
                        v = Sw[np.arange(len(h)), i0] * (1 - w) + Sw[np.arange(len(h)), i0 + 1] * w
                        best = np.maximum(best, v)
                    return best
                line1, c1a, c1b = at(hz), at(0.8 * hz), at(1.25 * hz)
                line2, c2a, c2b = at(2 * hz), at(1.6 * hz), at(2.5 * hz)
                for k, (a, b, c) in {"f": (line1, c1a, c1b), "f2": (line2, c2a, c2b)}.items():
                    res[k].append(pd.DataFrame({"split": tag, "line": a, "c1": b, "c2": c}))
            for k in ("f", "f2"):
                if not res[k]: continue
                t = pd.concat(res[k])
                for s in ("train", "test"):
                    u = t[t.split == s]
                    rows.append(dict(cohort=n, W=W, type=ty, line=k, split=s, windows=len(u),
                                     median_ratio_vs_control=np.nanmedian(u.line / np.sqrt(u.c1 * u.c2)),
                                     share_line_above_both_controls=float(((u.line > u.c1) & (u.line > u.c2)).mean())))
T = pd.DataFrame(rows); T.to_csv(R / "step2b_pulse_line.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).to_string())

"""Is the ~5.6 Hz line driven by the heartbeat? Per single 30 s window (no averaging over windows) on the explainer stretch and on
still sleep windows of 8 nights: band-pass 4-8 Hz (causal), envelope = |analytic signal| summed over axes, then the envelope's
autocorrelation over 40-180 bpm; compares its peak with Apple Watch HR. Also the same for the 0.67-3 Hz band signal itself."""
import os, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from scipy.signal import butter, sosfilt, hilbert
import ahr2 as A
from arcasleep.data.prepare import load_dataset
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from loadnight import load_with_rows
R = os.environ["R"]
c = COHORTS["sleepaccel"]; ids, reps = subject_ids(c); inv = {reps[s]: s for s in ids}; ds = load_dataset(DATA_ROOT / c.directory)
SOS_R = butter(4, (4.0, 8.0), btype="bandpass", fs=25.0, output="sos")
lag_f = 25.0 * 60.0 / A.GRID_BPM; l0 = np.floor(lag_f).astype(int); fr = lag_f - l0


def curve(x):
    x = x - x.mean(); v = (x**2).sum(); n = x.size
    r = np.zeros(l0.max() + 2)
    for L in range(1, r.size):
        r[L] = (x[:-L] * x[L:]).sum() / (n - L)
    r /= v / n
    return r[l0] * (1 - fr) + r[l0 + 1] * fr


rows = []
for p in sorted(os.listdir(f"{R}/est/sleepaccel")):
    rep = p[:-8]
    d = pd.read_parquet(f"{R}/est/sleepaccel/{p}", columns=["epoch", "stage5", "move", "dev30"])
    d = d[(d.stage5 >= 1) & (d.stage5 <= 5) & d.dev30.notna()]
    ep = d.nsmallest(20, "move")
    n, _ = load_with_rows("sleepaccel", inv[rep], rep, ds, None)
    N = n.labels.size * 750; acc = np.asarray(n.acc25[:N], float); ok = np.asarray(n.valid25[:N], bool)
    yr = A._filt(SOS_R, acc, ok); yl = A._filt(A._SOS_3, acc, ok)
    for e, hr in zip(ep.epoch, ep.dev30):
        s = slice(e * 750, (e + 1) * 750)
        env = np.abs(hilbert(yr[s], axis=0)).sum(axis=1)
        ce = curve(env); cl = np.array([curve(yl[s][:, j]) for j in range(3)]).sum(0)
        rows.append(dict(night=rep, hr=hr, env_peak=A.GRID_BPM[np.argmax(ce)], low_peak=A.GRID_BPM[np.argmax(cl)],
                         env_r_at_hr=np.interp(hr, A.GRID_BPM, ce), env_r_max=ce.max()))
t = pd.DataFrame(rows)
for k in ("env_peak", "low_peak"):
    err = (t[k] - t.hr).abs()
    print(f"{k}: windows {len(t)}, nights {t.night.nunique()}, within 3 bpm of HR {np.mean(err <= 3):.0%}, "
          f"at 2xHR (±4) {np.mean((t[k] - 2 * t.hr).abs() <= 4):.0%}, median |err| {err.median():.1f}")
print("chance of landing within 3 bpm on a 40-180 grid: ~", round(7 / 141, 3))
t.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ring_probe_windows.csv"), index=False)

"""Finds SleepAccel stretches of 40 epochs (20 min) of PSG sleep where the tracked estimate (acf, 0.67-8 Hz, 30 s, lambda 1)
is within 3 bpm of device HR in most epochs while the per-window autocorrelation peak is often off; prints the best few."""
import glob, os
import numpy as np, pandas as pd
R = os.environ["R"]
L = 40
rows = []
for f in sorted(glob.glob(f"{R}/est/sleepaccel/*.parquet")):
    s = os.path.basename(f)[:-8]
    d = pd.read_parquet(f, columns=["epoch", "stage5", "dev30", "move"])
    e2 = pd.read_parquet(f.replace("/est/", "/est2/"), columns=["est.acf.acfg8.0.30"])
    e3 = pd.read_parquet(f.replace("/est/", "/est3/"), columns=["est.acf.acfvd8.0.30"])
    tr = np.abs(e3.iloc[:, 0].to_numpy() - d.dev30.to_numpy())
    pk = np.abs(e2.iloc[:, 0].to_numpy() - d.dev30.to_numpy())
    sl = ((d.stage5 >= 1) & (d.stage5 <= 5)).to_numpy()
    ep = d.epoch.to_numpy()
    for i in range(0, len(d) - L, 5):
        if not sl[i:i + L].all() or ep[i + L - 1] - ep[i] != L - 1:
            continue
        rows.append((s, int(ep[i]), np.nanmean(tr[i:i + L] <= 3), np.nanmean(pk[i:i + L] > 10), np.nanmedian(d.dev30[i:i + L])))
t = pd.DataFrame(rows, columns=["night", "epoch0", "track_ok", "peak_off", "hr"])
t["score"] = t.track_ok + 0.5 * t.peak_off
print(t.sort_values("score", ascending=False).head(10).to_string(index=False))

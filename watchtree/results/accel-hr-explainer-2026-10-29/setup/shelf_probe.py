"""Probe of the fixed autocorrelation bands (~167, 115, 85 bpm) on SleepAccel: per single 30 s window (no averaging over windows),
the full autocorrelation at integer lags 1-40 of the 25 Hz filtered stream, and the strongest PSD line above 3 Hz at 25 Hz and at the
native rate. Prints per-window values for the explainer stretch and per-window peak frequencies for still sleep windows of 6 nights."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
import ahr2 as A
from senpy import compute_nufft_spectrogram
from arcasleep.data.prepare import load_dataset
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from loadnight import load_with_rows
R = os.environ["R"]
c = COHORTS["sleepaccel"]; ids, reps = subject_ids(c); inv = {reps[s]: s for s in ids}; ds = load_dataset(DATA_ROOT / c.directory)


def peaks(t, x, fs_hi, lo=3.0):
    r = compute_nufft_spectrogram(t, x, 30.0, 0.0, kind="psd", origin_s=float(t[0]), empty_windows="keep", min_samples=10)
    out = []
    for i in range(r.Sxx.shape[0]):
        if not r.valid[i]:
            out.append((np.nan, np.nan)); continue
        sel = (r.frequencies >= lo) & (r.frequencies <= fs_hi)
        S = r.Sxx[i][sel]; f = r.frequencies[sel]; j = np.argmax(S)
        out.append((f[j], S[j] / np.median(S)))
    return out


def night(rep, epochs):
    n, rows = load_with_rows("sleepaccel", inv[rep], rep, ds, None)
    ne = n.labels.size; N = ne * 750
    acc = np.asarray(n.acc25[:N], float); ok = np.asarray(n.valid25[:N], bool)
    y = A._filt(A._SOS_8, acc, ok)
    t0 = n.t0; tn = rows[:, 0] - t0
    fs_nat = 1 / np.median(np.diff(rows[:, 0]))
    res = []
    for e in epochs:
        seg = y[e * 750:(e + 1) * 750]; seg = seg - seg.mean(0); v = (seg**2).sum()
        r = np.array([(seg[:-L] * seg[L:]).sum() / (750 - L) for L in range(1, 41)]) / (v / 750)
        t25 = np.arange(e * 750, (e + 1) * 750) / 25.0
        p25 = peaks(t25, seg[:, 2], 12.5)[0]
        m = (tn >= e * 30) & (tn < (e + 1) * 30)
        xn = rows[m, 3] - rows[m, 3].mean()
        pn = peaks(tn[m], xn, fs_nat / 2)[0] if m.sum() > 100 else (np.nan, np.nan)
        res.append(dict(epoch=e, r=r, f25=p25[0], snr25=p25[1], fnat=pn[0], snrnat=pn[1], fs_nat=fs_nat))
    return res


if __name__ == "__main__":
    res = night("781756", [857 + 27, 857 + 5, 857 + 15])
    for d in res:
        r = d["r"]; pk = [L for L in range(2, 40) if r[L - 1] > r[L - 2] and r[L - 1] >= r[L]]
        print(f"epoch {d['epoch']}: native fs {d['fs_nat']:.1f} Hz; ACF local maxima at lags {pk} samples; r at them {[round(r[L-1],2) for L in pk]}")
        print(f"   strongest line >3 Hz: 25 Hz stream {d['f25']:.2f} Hz ({d['snr25']:.0f}x median), native {d['fnat']:.2f} Hz ({d['snrnat']:.0f}x median)")
    # other nights: still sleep windows, per-window peak frequency (no averaging)
    for p in sorted(os.listdir(f"{R}/est/sleepaccel"))[:8]:
        rep = p[:-8]
        d = pd.read_parquet(f"{R}/est/sleepaccel/{p}", columns=["epoch", "stage5", "move"])
        d = d[(d.stage5 >= 1) & (d.stage5 <= 5)]
        ep = d.nsmallest(5, "move").epoch.to_list()
        rr = night(rep, ep)
        print(rep[:2] + "…", "per-window strongest line, 25 Hz:", [round(x["f25"], 2) for x in rr], "native:", [round(x["fnat"], 2) for x in rr],
              "native fs", round(rr[0]["fs_nat"], 1))

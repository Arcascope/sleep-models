"""Step 1: sampling rate, amplitude quantization step and band noise floor of each cohort's accelerometer.
Native-rate rows (cleaned) for rate and quantization; the 25 Hz stream, causal 0.67-3 Hz band-pass, for the noise floor in
the lowest-movement decile of PSG-sleep epochs."""
import warnings, sys
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep.data.prepare import load_dataset
from arcasleep_watchtree.uncropped import RAW_ROOT, sync_table
from bpm_common import OUT, source_tables, SRCS
import accel_hr as A
from loadnight import load_with_rows
import os
R = os.environ["R"]
NN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
rng = np.random.default_rng(0)
out = []
for name in SRCS:
    c = COHORTS[name]
    ds = load_dataset(DATA_ROOT / c.directory)
    sync = sync_table(DATA_ROOT / RAW_ROOT) if name == "wearanize" else None
    sel = set(source_tables()[name].subject)
    ids, rep = subject_ids(c)
    ids = [s for s in ids if rep[s] in sel]
    pick = list(rng.choice(ids, min(NN, len(ids)), replace=False))
    rate, q_step, uniq, rms_still, psd_still, rms_all = [], [], [], [], [], []
    for s in pick:
        night, rows = load_with_rows(name, s, rep[s], ds, sync)
        ne = night.labels.size
        acc = night.acc25[: ne * 750]; ok = night.valid25[: ne * 750]
        if rows is not None:
            t = rows[:, 0]; dt = np.diff(t); dt = dt[dt > 0]
            rate.append(1 / np.median(dt))
            for a in range(3):
                u = np.unique(np.round(rows[:, 1 + a], 9))
                d = np.diff(u)
                q_step.append(d.min() * 1000 if d.size else np.nan)
                uniq.append(u.size)
        # band-passed noise per epoch
        from scipy.signal import sosfilt
        x = np.where(ok[:, None], acc, 0.0); x = x - x[np.flatnonzero(ok)[0]] if ok.any() else x
        y = sosfilt(A._SOS, np.where(ok[:, None], x, 0.0), axis=0)
        ye = y.reshape(ne, 750, 3); oke = ok.reshape(ne, 750).all(axis=1)
        jerk = np.r_[0.0, np.linalg.norm(np.diff(acc, axis=0), axis=1)].reshape(ne, 750).mean(axis=1)
        sleep = (night.labels >= 1) & (night.labels <= 5) & oke
        rms = np.sqrt((ye[:, 100:] ** 2).mean(axis=1)) * 1000  # mg per axis, skip filter start-up; (ne,3)
        mag = np.sqrt((ye[:, 100:] ** 2).sum(axis=2).mean(axis=1)) * 1000
        out.append(pd.DataFrame({"cohort": name, "night": rep[s], "epoch": np.arange(ne), "jerk": jerk, "sleep": sleep,
                                 "rms_x": rms[:, 0], "rms_y": rms[:, 1], "rms_z": rms[:, 2], "rms_mag": mag}))
    meta = {"cohort": name, "nights_probed": len(pick), "native_rate_hz_median": float(np.median(rate)) if rate else np.nan,
            "quant_step_mg_min": float(np.nanmin(q_step)), "quant_step_mg_median": float(np.nanmedian(q_step)),
            "distinct_values_per_axis_median": float(np.median(uniq))}
    print(meta, flush=True)
    pd.DataFrame([meta]).to_csv(f"{R}/logs/step1_meta_{name}.csv", index=False)
D = pd.concat(out)
D.to_parquet(f"{R}/logs/step1_epochs.parquet")

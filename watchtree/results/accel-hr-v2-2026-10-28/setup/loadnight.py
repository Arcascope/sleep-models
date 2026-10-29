"""Night loader that also returns the cleaned native-rate accelerometer rows (t, x, y, z in g)."""
import numpy as np, pandas as pd
from pisces_lite.datasets import align_trim, load_subject, mask_data
from pisces_lite.datasets.constants import PSG_COL, TIMESTAMP_COL
from arcasleep.processing import clean_accelerometer_rows
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, Night
from arcasleep_watchtree.preprocess import to_25hz
from arcasleep_watchtree.uncropped import RAW_ROOT, load_night_uncropped, sync_table


def load_with_rows(name, subject, reported, data_set, sync=None):
    c = COHORTS[name]
    if name == "wearanize":
        from arcasleep_watchtree.uncropped import wearanize_stream, align_uncropped, PSG_SUFFIX
        from arcasleep_watchtree.cohorts import wearanize_hr
        psg = pd.read_csv(DATA_ROOT / c.directory / "cleaned_psg" / f"{subject}{PSG_SUFFIX}", sep=" ", header=None,
                          names=[TIMESTAMP_COL, PSG_COL])
        stream = wearanize_stream(subject, sync, DATA_ROOT / RAW_ROOT)
        cleaned = pd.DataFrame(clean_accelerometer_rows(stream.to_numpy()), columns=stream.columns)
        accel, labels, t0 = align_uncropped(cleaned, psg)
        rows = accel.to_numpy(dtype=np.float64)
        acc25, valid25 = to_25hz(rows[:, 0], rows[:, 1:4], t0, labels.size)
        t, bpm = wearanize_hr(subject, rows)
        return Night("wearanize", reported, acc25, valid25, t - t0, bpm, labels, t0), rows
    loaded = load_subject(data_set, subject)
    cleaned = pd.DataFrame(clean_accelerometer_rows(loaded.accel_df.to_numpy()), columns=loaded.accel_df.columns)
    accel, psg = align_trim(cleaned, loaded.psg_df)
    psg = mask_data(accel, psg)
    t0 = float(psg[TIMESTAMP_COL].iloc[0]); labels = psg[PSG_COL].to_numpy()
    rows = accel.to_numpy(dtype=np.float64)
    acc25, valid25 = to_25hz(rows[:, 0], rows[:, 1:4], t0, labels.size)
    t, bpm = c.heart_rate(subject, rows) if c.heart_rate else (np.empty(0), np.empty(0))
    return Night(name, reported, acc25, valid25, t - t0, bpm, labels, t0), rows

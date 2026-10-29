import os, sys, numpy as np, pandas as pd, time, warnings
warnings.simplefilter("ignore")
import ahr2 as A, ratefree as RF, synth as S
from arcasleep.data.prepare import load_dataset
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.preprocess import to_25hz
from loadnight import load_with_rows
from bpm_common import source_tables
name = sys.argv[1]
ids, rep = subject_ids(COHORTS[name]); sel = set(source_tables()[name].subject)
todo = [s for s in ids if rep[s] in sel]; subj = todo[int(sys.argv[2])]
ds = load_dataset(DATA_ROOT / COHORTS[name].directory)
sync = None
if name == "wearanize":
    from arcasleep_watchtree.uncropped import sync_table, RAW_ROOT; sync = sync_table(DATA_ROOT / RAW_ROOT)
night, rows = load_with_rows(name, subj, rep[subj], ds, sync)
t = rows[:, 0]; t0 = night.t0; ne = 240
m = t < t0 + 30 * ne; t = t[m]; u = t - t0
x = S.pulse(u, 67.0)
y, v = to_25hz(t, x, t0, ne); valid = v.reshape(ne, -1).mean(axis=1) >= .8
tt = time.time()
c, nyq = RF.curves_all(t, x, t0, ne, valid, with_control=(t0 + np.arange(ne*750)[v[:ne*750]] / 25.0, y[v[:ne*750]]))
print("time", time.time() - tt, "nyq", np.nanmedian(nyq))
ca = A.acf_curves(y, v, ne, 30, A._SOS_8)
tr = S.pulse(np.arange(ne * 750) / 25.0, 67.0); cT = A.acf_curves(tr, np.ones(ne * 750, bool), ne, 30, A._SOS_8)
for k, cc in {"A": ca, "truth": cT, **c}.items():
    pk = A.GRID_BPM[np.nanargmax(np.where(np.isfinite(cc), cc, -9), axis=1)]
    print(k, np.nanmedian(np.abs(pk - 67)), np.nanmean(np.abs(A.viterbi_filter(10 * cc, 1.0)[0] - 67)))

"""Convert one fixed environmental accelerometer CSV to a cached npz (naive timestamps read as UTC seconds)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

src, dest = Path(sys.argv[1]), Path(sys.argv[2])
ts, xyz = [], []
for chunk in pd.read_csv(src, chunksize=5_000_000, dtype={'x': np.int32, 'y': np.int32, 'z': np.int32}):
    t = pd.to_datetime(chunk['timestamp'], format='ISO8601')
    ts.append(t.to_numpy('datetime64[us]').astype(np.int64))
    xyz.append(chunk[['x', 'y', 'z']].to_numpy(np.int32))
t_us = np.concatenate(ts); xyz = np.concatenate(xyz)
assert np.abs(xyz).max() < 32768
np.savez(dest, t_us=t_us, xyz=xyz.astype(np.int16), source=str(src))
print(dest.name, len(t_us), flush=True)

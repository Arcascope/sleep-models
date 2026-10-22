"""Compare the weaver_subject_* feature files of three copies, per recording.

Usage: compare_weaver_features.py OUT_CSV B_DIR A_DIR [A_DIR ...]

B_DIR is the reference (training data image copy); each A_DIR is a directory of S3 copies.
Columns: byte_identical, content_identical (all arrays equal), X shape, max abs X diff, y
equality and epoch count, and the first differing array.

— Sorrel
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out_csv, b_dir, *a_dirs):
    b_dir = Path(b_dir)
    rows = []
    for b in sorted(b_dir.glob("weaver_subject_*_features.npz")):
        a = next((Path(d) / b.name for d in a_dirs if (Path(d) / b.name).exists()), None)
        row = {"file": b.name, "subject_id": b.name.split("_")[2], "in_A": a is not None}
        if a is not None:
            row["byte_identical"] = sha(a) == sha(b)
            da, db = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
            diff = [k for k in db.files if k not in da.files or da[k].shape != db[k].shape
                    or not np.array_equal(da[k], db[k], equal_nan=db[k].dtype.kind == "f")]
            row["content_identical"] = not diff and set(da.files) == set(db.files)
            row["differing_arrays"] = "+".join(diff)
            row["X_shape_A"], row["X_shape_B"] = da["X"].shape, db["X"].shape
            row["epochs_A"], row["epochs_B"] = da["y"].shape[-1], db["y"].shape[-1]
            row["X_max_abs_diff"] = (float(np.abs(da["X"] - db["X"]).max())
                                     if da["X"].shape == db["X"].shape else "")
            row["y_equal"] = bool(da["y"].shape == db["y"].shape and np.array_equal(da["y"], db["y"]))
            if da["y"].shape == db["y"].shape:
                row["y_n_differ"] = int((da["y"] != db["y"]).sum())
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:])

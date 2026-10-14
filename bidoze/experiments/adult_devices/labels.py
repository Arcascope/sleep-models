"""Copy the labels of every cached recording, without its features.

score.py reads only the labels of the cache: the stage counts of each fold model's training
nights and of BIDsleep give the adult stage divisor. The cache itself is in the training
image; this writes its labels under the same file names, so the run can be scored where the
cache is not.

Usage: python labels.py <cache root> <out root>   (roots hold ``12Hz_log_PSD_aligned/``)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CACHE_NAME = "12Hz_log_PSD_aligned"


def main(cache_root: Path, out_root: Path) -> None:
    out = out_root / CACHE_NAME
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((cache_root / CACHE_NAME).glob("*_features.npz"))
    for path in paths:
        np.savez_compressed(out / path.name, y=np.load(path)["y"])
    print(f"labels of {len(paths)} recordings -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

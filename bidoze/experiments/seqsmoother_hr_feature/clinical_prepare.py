"""SeqSmoother feature caches for SleepAccel-Clinical, in the SleepAccel cache format.

SleepAccel-Clinical has no heart-rate recording, so every window has reference NaN and
``eligible`` False: the caches are for inference only. Features are those of
``seqsmoother_weaver/prepare.py`` (100-Hz grid, 60-s windows every 15 s, Nightbeat anchor).

Usage: python clinical_prepare.py --data <pisces2-data> --output <run> --vendor <Nightbeat> [--workers N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seqsmoother_weaver"))
from nightbeat_adapter import predict  # noqa: E402
from prepare import FEATURES, cache_arrays, extract, features, resample  # noqa: E402

COHORT = "SleepAccel-Clinical"


def prepare(job: tuple) -> dict:
    sid, data, dest, vendor = job
    if dest.exists():
        return dict(participant=sid, status="already_cached")
    begun = time.monotonic()
    path = data / COHORT / "cleaned_accelerometer" / f"{sid}.csv"
    raw = pd.read_csv(path)
    d = pd.DataFrame({"appletime": raw.TIMESTAMP, "applex": raw.ACC_X, "appley": raw.ACC_Y, "applez": raw.ACC_Z})
    xyz, valid, origin, meta = resample(d, "apple")
    empty = np.array([])
    frame, psd, frequency = extract(xyz, valid, origin, empty, empty, empty, d)
    frame = frame.merge(predict(xyz, valid, origin, vendor), on="endpoint", how="left", validate="one_to_one")
    meta.update(participant=sid, night=1, device="apple", cohort=COHORT, reference_source="none",
                nightbeat=True, windows=len(frame), label_windows=0, source_path=str(path),
                seconds_elapsed=time.monotonic() - begun, feature_names=FEATURES)
    arrays = cache_arrays(frame)
    arrays.update(X=features(frame, psd, frequency), eligible=np.zeros(len(frame), bool),
                  metadata=np.array(json.dumps(meta)))
    np.savez_compressed(dest, **arrays)
    print(f"{sid}: {len(frame)} windows, {meta['seconds_elapsed']:.0f} s", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vendor", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cache = a.output / "clinical_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (a.data / COHORT / "cleaned_accelerometer").glob("*.csv"))
    jobs = [(sid, a.data, cache / f"{sid}.npz", a.vendor) for sid in ids]
    with ProcessPoolExecutor(a.workers) as pool:
        audits = list(pool.map(prepare, jobs))
    (a.output / "clinical_preparation.json").write_text(json.dumps(audits, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

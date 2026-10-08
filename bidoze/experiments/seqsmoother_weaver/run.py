"""Resume the frozen Weaver pilot and report after each completed held-out fold."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIN = "119dd8e818509ae49e7429879f9e4140bc2fb671"
HERE = Path(__file__).resolve().parent


def call(command):
    print("Running: " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wait-preparation", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--train-stride", type=int, default=40)
    args = ap.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    root = args.output.resolve()
    vendor = root / "vendor/Nightbeat"
    pilot = root / "pilot"
    split = root / "splits.json"
    if not split.exists():
        raise FileNotFoundError("Freeze splits before starting: " + str(split))
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise ValueError("Unexpected official Nightbeat commit " + commit)
    protocol = json.loads(split.read_text())
    if args.wait_preparation:
        while not (pilot / "preparation.json").exists():
            print("Waiting for full-night preparation; cached recordings: " +
                  str(len(list((pilot / "cache").glob("*.npz")))), flush=True)
            time.sleep(30)
    elif not (pilot / "preparation.json").exists():
        call([sys.executable, HERE / "prepare.py", "--data", args.data, "--output", pilot,
            "--vendor", vendor, "--devices", "actigraph", "--nightbeat",
            "--workers", args.workers, "--participants", *protocol["pilot_participants"]])
    report = [sys.executable, HERE / "report.py", "--cache", pilot / "cache",
              "--splits", split, "--folds-dir", root / "models", "--output", root / "report"]
    call(report)
    for fold in protocol["folds"]:
        if args.folds is not None and fold["fold"] not in args.folds:
            continue
        call([sys.executable, HERE / "train.py", "--cache", pilot / "cache",
            "--splits", split, "--output", root / "models", "--folds", fold["fold"],
            "--epochs", args.epochs, "--train-stride", args.train_stride])
        call(report)
    print("Requested training folds finished. Inspect report/status.json for full-study completion.", flush=True)


if __name__ == "__main__":
    main()

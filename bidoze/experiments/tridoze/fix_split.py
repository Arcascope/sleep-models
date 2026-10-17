"""Correct the fit/eval key lists of training-recording predictions written before the split was sorted.

train_predictions.py splits a draw's training recordings as training does: ``split_train_eval`` over
the sorted recordings. Files written from the unsorted list hold the wrong ``__fit__`` and ``__eval__``;
the predictions themselves are per recording and stay as they are. This rewrites the two lists from
the sorted split and keeps the earlier lists as ``__fit_unsorted__`` and ``__eval_unsorted__``. Files
whose lists already match are left alone.

Usage: python fix_split.py [--run RUN]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import train_predictions as tp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[2] / "results/tridoze-2026-10-16")
    args = ap.parse_args()
    splits: dict[int, tuple[list[str], list[str]]] = {}
    fixed = same = 0
    for path in sorted(args.run.glob("*/draw*/train_predictions/*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        d = int(path.parent.parent.name.removeprefix("draw"))
        if d not in splits:
            _, fit_refs, eval_refs = tp.training_split(args.run, d)
            splits[d] = [tp.key(r) for r in fit_refs], [tp.key(r) for r in eval_refs]
        fit, evals = splits[d]
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        if sorted(map(str, arrays["__fit__"])) == sorted(fit) and sorted(map(str, arrays["__eval__"])) == sorted(evals):
            same += 1
            continue
        arrays.setdefault("__fit_unsorted__", arrays["__fit__"])
        arrays.setdefault("__eval_unsorted__", arrays["__eval__"])
        arrays["__fit__"], arrays["__eval__"] = np.array(fit), np.array(evals)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(path)
        fixed += 1
        print(f"fixed {path.relative_to(args.run)}", flush=True)
    print(f"{fixed} corrected, {same} already right")


if __name__ == "__main__":
    main()

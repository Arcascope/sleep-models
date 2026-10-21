"""What the boosted members learn on their draw's fit recordings, beside unweighted seeds.

Calls are the argmax of the 4 stage probabilities (no stage divisor) on the PSG-scored epochs of
the fit recordings that training used (``__fit__`` of the training-prediction files).

Writes to ``<run>/analysis/<out>/``:
- ``second_member.csv``: per draw and pair (first, second member), the share of fit epochs the first
  gets wrong, the second's accuracy where the first is wrong and where it is right, the second's
  error and the share both get wrong. Pairs: boosted member 1 after BiDoze-G seed 0; independent
  seeds 1 and 2 after seed 0; BiDoze-G seeds 1 and 2 after seed 0.
- ``boost_targets.csv``: per draw, for the errors of member 0, of the mean of members 0 and 1, and
  of the mean of all 3 boosted members (and member 2 alone): the error rate, the share of the
  errors by called stage and by PSG stage, and the share of epochs called each stage.

Usage: python boost_analysis.py [--run RUN] [--suffix c] [--out boost-c]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[2] / "results/tridoze-2026-10-16"
STAGES = ["wake", "light", "deep", "rem"]


def labels_and_proba(paths: list[Path], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """PSG stage and the mean 4-stage probability of the given members, over scored fit epochs."""
    stores = [np.load(p) for p in paths]
    ys, ps = [], []
    for k in keys:
        y = stores[0][f"__labels__/{k}"].astype(int)
        scored = (y >= 0) & (y < len(STAGES))
        ys.append(y[scored])
        ps.append(np.mean([s[k][scored, :len(STAGES)] for s in stores], axis=0))
    return np.concatenate(ys), np.concatenate(ps)


def npz(run: Path, arm: str, d: int, seed: int) -> Path:
    return run / f"{arm}/draw{d}/train_predictions/draw{d}_seed_{seed}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--suffix", default="c", help="boosted arms are bidoze_g_boost<k><suffix>")
    ap.add_argument("--out", default="boost-c")
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    out = args.run / "analysis" / args.out
    out.mkdir(parents=True, exist_ok=True)
    pair_rows, target_rows = [], []
    for d in args.draws:
        g0 = npz(args.run, "bidoze_g", d, 0)
        b1 = npz(args.run, f"bidoze_g_boost1{args.suffix}", d, 1)
        b2 = npz(args.run, f"bidoze_g_boost2{args.suffix}", d, 2)
        if not (g0.exists() and b1.exists()):
            print(f"skip draw {d}: predictions missing")
            continue
        fit = [str(k) for k in np.load(g0)["__fit__"]]
        pairs = {"boosted: member 1 given member 0": (g0, b1),
                 "independent: seed 1 given seed 0": (npz(args.run, "independent", d, 0), npz(args.run, "independent", d, 1)),
                 "independent: seed 2 given seed 0": (npz(args.run, "independent", d, 0), npz(args.run, "independent", d, 2)),
                 "BiDoze-G: seed 1 given seed 0": (g0, npz(args.run, "bidoze_g_s1", d, 1)),
                 "BiDoze-G: seed 2 given seed 0": (g0, npz(args.run, "bidoze_g_s2", d, 2))}
        for label, (a, b) in pairs.items():
            if not (a.exists() and b.exists()):
                continue
            y, pa = labels_and_proba([a], fit)
            y2, pb = labels_and_proba([b], fit)
            assert (y == y2).all()
            wrong_a, right_b = pa.argmax(1) != y, pb.argmax(1) == y
            pair_rows.append({"draw": d, "pair": label, "first wrong": wrong_a.mean(),
                              "second right | first wrong": right_b[wrong_a].mean(),
                              "second right | first right": right_b[~wrong_a].mean(),
                              "second wrong": (~right_b).mean(), "both wrong": (wrong_a & ~right_b).mean()})
        groups = {"errors of member 0 (boost1 targets)": [g0], "errors of members 0+1 (boost2 targets)": [g0, b1]}
        if b2.exists():
            groups["errors of all 3"] = [g0, b1, b2]
            groups["member 2 alone"] = [b2]
        for label, paths in groups.items():
            y, p = labels_and_proba(paths, fit)
            c = p.argmax(1)
            wrong = c != y
            target_rows.append({"draw": d, "ensemble": label, "err": wrong.mean(),
                                **{f"called {s} wrongly": np.mean(c[wrong] == i) for i, s in enumerate(STAGES)},
                                **{f"missed {s}": np.mean(y[wrong] == i) for i, s in enumerate(STAGES)},
                                **{f"calls {s}": np.mean(c == i) for i, s in enumerate(STAGES)},
                                "mean p(rem)": float(p[:, 3].mean())})
    pairs_df, targets_df = pd.DataFrame(pair_rows), pd.DataFrame(target_rows)
    pairs_df.to_csv(out / "second_member.csv", index=False)
    targets_df.to_csv(out / "boost_targets.csv", index=False)
    pd.set_option("display.width", 220)
    if not pairs_df.empty:
        print(pairs_df.groupby("pair").mean(numeric_only=True).drop(columns="draw").round(3).to_string())
    if not targets_df.empty:
        print(targets_df.groupby("ensemble").mean(numeric_only=True).drop(columns="draw").round(3).to_string())


if __name__ == "__main__":
    main()

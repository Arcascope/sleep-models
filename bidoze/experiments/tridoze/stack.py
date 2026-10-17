"""Stacking: a small learned combiner of the 3 members' outputs, against their plain mean.

For each ensemble (insight.py's list) and draw, the combiner is a multinomial logistic regression
from per-epoch inputs to the 4 stages. It is fitted on the draw's eval-share recordings: the 30% of
the training participants that every arm of the draw holds out of its fit and uses only to choose
the stopping epoch (train_predictions.py scores them). Those recordings are mostly children
(weaverDraw); the test subjects are adults. Classes are weighted to balance, as the members' own
loss is, so the combiner's output goes through the same adult stage divisor as the mean output.
Only PSG-scored epochs whose mean gap probability is < 0.5 are fitted.

Inputs per epoch (standardised):
- ``members``: log of each member's 4 stage probabilities (given not gap), 12 inputs;
- ``members+movement``: the same, plus log(1 + mean absolute arm-angle change, degrees per 5 s)
  and the mean log spectral power of the epoch's input frames.
No input says where in the recording an epoch lies.

Writes the combiner's output in the prediction-CSV format (stage columns replaced by its
probabilities, gap_head the members' mean) to ``<run>/stacked/<slug>-<variant>/draw<d>/predictions/
draw<d>_seed_0/``, for evaluate.py as an arm, and ``<run>/analysis/stacking.csv`` with each fit's
recordings, epochs and per-class weight norms.

Usage: python stack.py [--run RUN] [--draws 0 1 2 3 4] [--ensemble label ...]
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import insight

STAGES = insight.STAGES
CACHE_NAME = insight.CACHE_NAME
VARIANTS = ["members", "members+movement"]


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower().replace("×", "x").replace("β", "b")).strip("-")


def movement(cache: Path, dataset: str, sid: str, n: int) -> np.ndarray:
    """(n, 2): log(1 + angle change), mean log power of the epoch's present frames."""
    z = np.load(cache / f"{dataset}_subject_{sid}_features.npz")
    angle = np.nan_to_num(insight.angle_change(cache, dataset, sid, n), nan=0.0)
    X = z["X"][0]
    fpe = X.shape[0] // max(z["y"].shape[-1], 1)
    frames = X[: n * fpe].reshape(-1, fpe, *X.shape[1:])
    present = ~np.all(frames == -1.0, axis=(2, 3))
    level = frames.mean(axis=(2, 3))
    with np.errstate(all="ignore"):
        power = np.where(present.any(axis=1), (level * present).sum(axis=1) / present.sum(axis=1), np.nan)
    out = np.zeros((n, 2))
    out[:, 0] = np.log1p(angle)
    out[: power.shape[0], 1] = power
    fill = np.nanmin(power) if np.isfinite(power).any() else 0.0
    out[:, 1] = np.where(np.isfinite(out[:, 1]), out[:, 1], fill)
    return out


def features(member_p4: list[np.ndarray], move: np.ndarray | None) -> np.ndarray:
    logs = [np.log(np.clip(p, 1e-6, None)) for p in member_p4]
    return np.column_stack(logs + ([move] if move is not None else []))


def conditional(p5: np.ndarray) -> np.ndarray:
    s = p5[:, :4].sum(axis=1, keepdims=True)
    return p5[:, :4] / np.clip(s, 1e-12, None)


def train_npz(run: Path, pred_dir: Path) -> Path:
    arm, draw = pred_dir.parents[2].name, pred_dir.parents[1].name
    return run / arm / draw / "train_predictions" / f"{pred_dir.name}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=insight.RUN)
    ap.add_argument("--baseline-run", type=Path, default=insight.BASELINE)
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ensemble", nargs="+", default=None)
    args = ap.parse_args()
    draws_json = json.loads((args.baseline_run / "setup/draws.json").read_text())
    defs = insight.ensembles(args.run, args.baseline_run)
    rows = []
    for label in args.ensemble or list(defs):
        for d in args.draws:
            dirs = defs[label](d)
            npzs = [train_npz(args.run, p) for p in dirs]
            if not all(p.is_dir() for p in dirs) or not all(n.exists() for n in npzs):
                print(f"skip {label} draw {d}: predictions missing")
                continue
            stores = [np.load(n) for n in npzs]
            train_cache = args.run / f"setup/cache/draw{d}" / CACHE_NAME
            test_cache = args.baseline_run / f"setup/cache/draw{d}" / CACHE_NAME
            keys = list(stores[0]["__eval__"])
            Xs, ys, n_adult = {v: [] for v in VARIANTS}, [], 0
            for k in keys:
                y = stores[0][f"__labels__/{k}"].astype(int)
                p5 = [s[k] for s in stores]
                keep = (y >= 0) & (y < 4) & (np.mean([p[:, 4] for p in p5], axis=0) < 0.5)
                dataset, sid = k.split("/", 1)
                n_adult += dataset != "weaverDraw"
                move = movement(train_cache, dataset, sid, len(y))
                for v in VARIANTS:
                    Xs[v].append(features([conditional(p) for p in p5], move if v.endswith("movement") else None)[keep])
                ys.append(y[keep])
            y_fit = np.concatenate(ys)
            for v in VARIANTS:
                X_fit = np.concatenate(Xs[v])
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_fit, y_fit)
                coef = model[-1].coef_
                rows.append({"ensemble": label, "draw": d, "variant": v, "recordings": len(keys),
                             "adult_recordings": n_adult, "epochs": len(y_fit),
                             "fit_accuracy": float(model.score(X_fit, y_fit)),
                             **{f"coef_member{m}": float(np.abs(coef[:, 4 * m:4 * m + 4]).sum()) for m in range(3)},
                             **({"coef_movement": float(np.abs(coef[:, 12:]).sum())} if v.endswith("movement") else {})})
                dest = args.run / "stacked" / f"{slug(label)}-{slug(v)}" / f"draw{d}/predictions/draw{d}_seed_0"
                dest.mkdir(parents=True, exist_ok=True)
                for dataset in insight.COHORTS:
                    for sid in draws_json[f"draw{d}"][dataset]:
                        frames = [pd.read_csv(p / f"{dataset}_{sid}.csv", dtype={"psg": str}) for p in dirs]
                        p4 = [f[STAGES].to_numpy(float) for f in frames]
                        move = movement(test_cache, dataset, sid, len(frames[0])) if v.endswith("movement") else None
                        proba = model.predict_proba(features(p4, move))
                        out = frames[0].copy()
                        out[STAGES] = proba[:, np.argsort(model[-1].classes_)].round(6)
                        out["gap_head"] = np.mean([f["gap_head"].to_numpy(float) for f in frames], axis=0).round(6)
                        out.to_csv(dest / f"{dataset}_{sid}.csv", index=False)
                print(f"{label} draw {d} {v}: fit on {len(keys)} recordings ({n_adult} adult), {len(y_fit)} epochs, "
                      f"fit accuracy {rows[-1]['fit_accuracy']:.3f}", flush=True)
    out = args.run / "analysis/stacking.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if not old.empty:
        old = old.merge(new[["ensemble", "draw", "variant"]], how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
    pd.concat([old, new]).to_csv(out, index=False)


if __name__ == "__main__":
    main()

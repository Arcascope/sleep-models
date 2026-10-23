"""Checks on a converted EESM19 layout.

    python checks.py OUT_DIR

Prints, in summary: the shake alignment found by the conversion, how wrist
movement lines up with scorer 1 wake (AUROC of movement for wake vs sleep at
epoch lags -10..+10), the agreement of scorer 1 with scorer 2 (Cohen's kappa),
and stage proportions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

EPOCH_S = 30.0
LAGS = np.arange(-10, 11)
# Wake, light (N1 + N2), deep (N3), REM.
FOUR_STAGE = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3}
SLEEP_WAKE = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1}


def read_pairs(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    classes = np.union1d(a, b)
    if classes.size < 2:
        return float("nan")
    table = np.array([[np.sum((a == i) & (b == j)) for j in classes] for i in classes], float)
    n = table.sum()
    observed = np.trace(table) / n
    expected = float(table.sum(0) @ table.sum(1)) / n**2
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def movement_per_epoch(acc: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Summed absolute change of the acceleration magnitude in each epoch."""
    magnitude = np.linalg.norm(acc[:, 1:], axis=1)
    change = np.abs(np.diff(magnitude, prepend=magnitude[0]))
    bins = np.append(times, times[-1] + EPOCH_S)
    return np.histogram(acc[:, 0], bins=bins, weights=change)[0]


def mapped(stages: np.ndarray, table: dict) -> np.ndarray:
    return np.array([table.get(int(s), -1) for s in stages])


def main() -> None:
    out = Path(sys.argv[1])
    ids = sorted(p.name[: -len("_cleaned_psg.out")] for p in (out / "cleaned_psg").glob("*_cleaned_psg.out"))
    lag_auc, peak_lags = [], []
    k = {"scorer2": ([], [])}
    pooled = np.zeros(6)
    night_props = []
    for sid in ids:
        psg = read_pairs(out / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        times, stages = psg[:, 0], psg[:, 1].astype(int)
        valid = stages >= 0
        pooled += np.bincount(stages[valid], minlength=6)[:6]
        night_props.append(np.bincount(stages[valid], minlength=6)[:6] / max(valid.sum(), 1))

        acc = read_pairs(out / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        move = np.log1p(movement_per_epoch(acc, times))
        aucs = []
        for lag in LAGS:
            if lag >= 0:
                m, s = move[lag:], stages[: stages.size - lag]
            else:
                m, s = move[:lag], stages[-lag:]
            ok = s >= 0
            aucs.append(auroc(m[ok], (s[ok] == 0)))
        aucs = np.array(aucs)
        lag_auc.append(aucs)
        peak_lags.append(int(LAGS[np.nanargmax(aucs)]))

        for name in k:
            path = out / name / f"{sid}_{name}.out"
            if not path.exists():
                continue
            other = read_pairs(path)
            # Both scorers use the EDF clock; each scorer 2 epoch is paired with the scorer 1 epoch that starts nearest to it.
            idx = np.abs(other[:, 0][:, None] - times[None, :]).argmin(1)
            near = np.abs(other[:, 0] - times[idx]) <= EPOCH_S / 2
            a, b = stages[idx[near]], other[near, 1].astype(int)
            ok = (a >= 0) & (b >= 0)
            k[name][0].append(kappa(mapped(a[ok], FOUR_STAGE), mapped(b[ok], FOUR_STAGE)))
            k[name][1].append(kappa(mapped(a[ok], SLEEP_WAKE), mapped(b[ok], SLEEP_WAKE)))
    lag_auc = np.array(lag_auc)
    print(f"nights: {len(ids)}")
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    done = [r for r in manifest if r["status"] == "converted"]
    diffs = np.array([r["clock_difference_s"] for r in done])
    print(f"manifest: {len(done)} converted, {len(manifest) - len(done)} dropped")
    print(f"shake peak: r {np.min([r['alignment']['peak_r'] for r in done]):.2f} to {np.max([r['alignment']['peak_r'] for r in done]):.2f}, "
          f"z {np.min([r['alignment']['peak_z'] for r in done]):.1f} to {np.max([r['alignment']['peak_z'] for r in done]):.1f}")
    print(f"clock difference (s), watch clock vs EDF header: median {np.median(diffs):.1f}, "
          f"nights within 30 s of 5 s: {int(np.sum(np.abs(diffs - 5) <= 30))} of {len(diffs)}")
    print("max timestamp deviation from a 100 Hz clock (s):", max(r["timestamp_deviation_s"] for r in done))
    print("epochs without wrist data, total:", sum(r["epochs_without_watch"] for r in done), "of", sum(r["epochs"] for r in done))
    print("mean wake AUROC of wrist movement by epoch lag (positive lag: movement later than label):")
    for lag, v in zip(LAGS, np.nanmean(lag_auc, 0)):
        print(f"  {lag:+3d}: {v:.3f}")
    values, counts = np.unique(peak_lags, return_counts=True)
    print("per-night peak lag counts:", dict(zip(values.tolist(), counts.tolist())))
    zero = lag_auc[:, list(LAGS).index(0)]
    print(f"AUROC at lag 0: mean {np.nanmean(zero):.3f}, nights above 0.5: {int(np.sum(zero > 0.5))} of {len(zero)}")
    for name in k:
        four, two = k[name]
        print(f"scorer 1 vs {name}: nights {len(four)}, 4-stage kappa {np.nanmean(four):.3f}, sleep/wake kappa {np.nanmean(two):.3f}")
    names = ["Wake", "N1", "N2", "N3", "N4", "REM"]
    frac = pooled / pooled.sum()
    print("stage proportions, pooled scored epochs:", {n: round(float(f) * 100, 1) for n, f in zip(names, frac) if n != "N4"})
    print("stage proportions, mean over nights:", {n: round(float(f) * 100, 1) for n, f in zip(names, np.mean(night_props, 0)) if n != "N4"})
    low = [(sid, round(float(z), 3), p) for sid, z, p in zip(ids, zero, peak_lags) if not z > 0.6]
    print("nights with lag-0 AUROC at or below 0.6:", low)


if __name__ == "__main__":
    main()

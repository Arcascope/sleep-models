"""Score single-seed bundles on land and on the boats, against PSG, with ``inject.score``.

Cohorts: SleepAccel-Clinical (the adult release's test cohort; the land guard), SURF's Galaxy
Watch nights and Somerset. Each night is scored by every bundle with ``Predictor.predict`` (the
GGIR sleep filter on, origin on the PSG's 30-s grid). A bundle whose ``arm.json`` sets ``lowpass``
gets its frames from ``fused.frames`` with the jerk low-pass (``fused.lowpass_taps``), on the
same cleaned samples and origin as the release's features.

Writes ``<out>/predictions/<model>/<night>.csv``, ``<out>/nights.csv`` (per night and model), ``<out>/summary.csv`` (per cohort and model,
means over nights) and ``<out>/paired.csv`` (per cohort, metric and model: the mean
participant-level difference from the first model, and the Wilcoxon signed-rank p over
participants).

Usage: python arms_score.py <data root> <out> <name=bundle> [<name=bundle> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

import fused
from arcasleep.data.prepare import clean_accelerometer_rows
from arcasleep.predict import load_predictor
from inject import score
from survey import load_accel

METRICS = ["sleep_sensitivity", "wake_specificity", "kappa_sleep_wake", "kappa_4stage"]


def nights(root: Path) -> list[tuple[str, str]]:
    out = []
    for cohort, pattern in [("SleepAccel-Clinical", "*.csv"), ("SURF", "*_GalaxyWatch.csv"), ("Somerset", "*.csv")]:
        for p in sorted((root / cohort / "cleaned_psg").glob(pattern)):
            if (root / cohort / "cleaned_accelerometer" / p.name).exists():
                out.append((cohort, p.stem))
    return out


def lowpass_features(released):
    """The release's ``features``, with frames from ``fused.frames`` and the jerk low-pass."""

    def features(acc, origin):
        ref, origin_s = released(acc, origin)
        clean = clean_accelerometer_rows(np.asarray(acc, np.float64))
        p = fused.plan(clean[:, 0], origin_s)
        got = fused.frames(*fused.device_plan(p), jnp.asarray(clean[:, 1:], jnp.float32),
                           jnp.asarray(fused.lowpass_taps(p.median_fs)))
        return np.asarray(got, ref.dtype)[: ref.shape[0]], origin_s

    return features


def load(path: Path):
    predictor = load_predictor(path)
    arm = path / "arm.json"
    if arm.exists() and json.loads(arm.read_text()).get("lowpass"):
        # A composite bundle's frames come from its ensemble; a single bundle's from itself.
        target = getattr(predictor, "ensemble", predictor)
        target.features = lowpass_features(target.features)
    return predictor


def main(root: Path, out: Path, bundles: dict[str, Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    predictors = {name: load(path) for name, path in bundles.items()}
    rows = []
    for cohort, stem in nights(root):
        accel = load_accel(root, cohort, stem)
        psg = pd.read_csv(root / cohort / "cleaned_psg" / f"{stem}.csv")
        origin = float(psg.TIMESTAMP.iloc[0] % 30.0)
        participant = stem if cohort == "SleepAccel-Clinical" else stem.split("_")[0]
        for name, predictor in predictors.items():
            target = out / "predictions" / name / f"{stem}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            predictor.predict(accel, origin).to_csv(target)
            rows.append({"cohort": cohort, "night": stem, "participant": participant, "model": name}
                        | score(psg, pd.read_csv(target)))
        print(f"{cohort}/{stem}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "nights.csv", index=False, float_format="%.4f")
    table.groupby(["cohort", "model"], sort=False)[METRICS + ["epochs"]].agg(["mean", "count"]).to_csv(
        out / "summary.csv", float_format="%.4f")
    first = next(iter(bundles))
    paired = []
    for cohort, group in table.groupby("cohort"):
        by = group.groupby(["participant", "model"])[METRICS].mean().unstack("model")
        for metric in METRICS:
            for name in bundles:
                if name == first:
                    continue
                d = (by[(metric, name)] - by[(metric, first)]).dropna()
                p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
                paired.append({"cohort": cohort, "metric": metric, "model": name, "against": first,
                               "participants": d.size, "mean_difference": d.mean(), "wilcoxon_p": p})
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False, float_format="%.4f")
    print(table.groupby(["cohort", "model"], sort=False)[METRICS].mean().round(3).to_string())
    print(pd.DataFrame(paired).round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]),
         {a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in sys.argv[3:]})

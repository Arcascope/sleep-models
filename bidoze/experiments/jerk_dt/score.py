"""Do the adult release's calls change when its jerk is divided by the time step?

Per night, ``Predictor.predict`` (GGIR sleep filter on, origin on the PSG's 30-s grid) under each
jerk mode of ``modes.py``, with the release's weights unchanged. Against ``release``:

- agreement: the share of epochs with the same ``stage`` (after the GGIR filter) and
  ``stage_raw`` (before), and the mean and largest change in any stage probability;
- the change in predicted total sleep time, in minutes.

The release's JAX NUFFT is not bit-deterministic, so ``release`` is also predicted a second
time (``release_rerun``); its differences from ``release`` are the noise floor.

Where the PSG uses the 0–4 coding (SleepAccel-Clinical): sleep sensitivity, wake specificity,
κ sleep/wake, κ 4-stage and TST error (model − PSG, minutes) over PSG-scored epochs; a gap
counts as a miss. ``paired.csv`` gives, per cohort, metric and mode, the mean difference from
``release`` over participants (a participant's nights averaged first), its 95% t-interval and the
Wilcoxon signed-rank p.

Nights the release refuses (``ValueError``, such as a sampling rate below 12 Hz) are skipped
and named in the log. Writes ``<out>/predictions/<mode>/<cohort>_<night>.csv``, ``<out>/nights.csv``,
``<out>/summary.csv`` (means per cohort and mode) and ``<out>/paired.csv``.

Usage: python score.py <data root> <adult bundle> <out> <cohort> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from arcasleep.predict import load_predictor
from modes import MODES, jerk_mode, load_night, nights

PSG_TO_STAGE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
STAGES = ["wake", "light", "deep", "rem"]
METRICS = ["sleep_sensitivity", "wake_specificity", "kappa_sleep_wake", "kappa_4stage", "tst_error_minutes"]
RUNS = MODES + ("release_rerun",)
AGREEMENT = ["same_stage", "same_stage_raw", "mean_abs_dp", "max_abs_dp", "tst_change_minutes"]


def kappa(y: np.ndarray, p: np.ndarray) -> float:
    cats = np.union1d(y, p)
    cm = np.zeros((cats.size, cats.size))
    np.add.at(cm, (np.searchsorted(cats, y), np.searchsorted(cats, p)), 1)
    n = cm.sum()
    po, pe = np.trace(cm) / n, (cm.sum(0) * cm.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def psg_metrics(psg: pd.DataFrame, pred: pd.DataFrame) -> dict:
    p = pred.set_index(np.rint(pred.epoch_start_s.to_numpy()).astype(int))
    t = np.rint(psg.TIMESTAMP.to_numpy()).astype(int)
    y = np.array([PSG_TO_STAGE.get(int(v), -1) for v in psg.PSG.to_numpy()])
    keep = (y >= 0) & np.isin(t, p.index)
    y, t = y[keep], t[keep]
    call = p.loc[t, "stage"].map({s: i for i, s in enumerate(STAGES)} | {"gap": 4}).to_numpy()
    sleep = y > 0
    sw = np.where(call == 4, 2, (call > 0).astype(int))
    return {"psg_epochs": int(y.size),
            "sleep_sensitivity": 100 * np.mean(sw[sleep] == 1),
            "wake_specificity": 100 * np.mean(sw[~sleep] == 0) if (~sleep).any() else np.nan,
            "kappa_sleep_wake": kappa(sleep.astype(int), sw),
            "kappa_4stage": kappa(y, call),
            "tst_error_minutes": 0.5 * (np.sum(sw == 1) - np.sum(sleep))}


def frame(prediction) -> pd.DataFrame:
    df = pd.DataFrame(prediction.probabilities, columns=list(prediction.stage_names))
    df.insert(0, "epoch_start_s", prediction.epoch_start_s)
    df["stage"] = np.asarray(prediction.stage)
    df["stage_raw"] = np.asarray(prediction.stage_raw)
    return df


def paired(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for cohort, group in table.groupby("cohort", sort=False):
        by = group.groupby(["participant", "mode"])[columns].mean().unstack("mode")
        for metric in columns:
            if by[metric]["release"].isna().all():
                continue
            for mode in RUNS[1:]:
                d = (by[(metric, mode)] - by[(metric, "release")]).dropna().to_numpy(float)
                half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
                p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
                rows.append({"cohort": cohort, "metric": metric, "mode": mode, "participants": d.size,
                             "mean_difference": d.mean(), "ci_low": d.mean() - half, "ci_high": d.mean() + half,
                             "wilcoxon_p": p})
    return pd.DataFrame(rows)


def main(root: Path, bundle: Path, out: Path, night_list: list[tuple[str, str]],
         participant=lambda cohort, stem: stem) -> None:
    predictor = load_predictor(bundle)
    rows = []
    for cohort, stem in night_list:
        accel, psg, origin = load_night(root, cohort, stem)
        preds = {}
        try:
            for mode in RUNS:
                with jerk_mode(mode.removesuffix("_rerun")):
                    preds[mode] = frame(predictor.predict(accel, origin))
        except ValueError as e:  # a recording the release refuses, such as one sampled below 12 Hz
            print(f"{cohort}/{stem}: skipped: {e}", flush=True)
            continue
        for mode in RUNS:
            target = out / "predictions" / mode / f"{cohort}_{stem}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            preds[mode].to_csv(target, index=False)
        base = preds["release"]
        names = [c for c in STAGES if c in base.columns]
        base_sleep = np.isin(base.stage, STAGES[1:]).sum()
        for mode, pred in preds.items():
            dp = np.abs(pred[names].to_numpy() - base[names].to_numpy())
            row = {"cohort": cohort, "night": stem, "participant": participant(cohort, stem), "mode": mode,
                   "epochs": len(pred),
                   "same_stage": float(np.mean(pred.stage == base.stage)),
                   "same_stage_raw": float(np.mean(pred.stage_raw == base.stage_raw)),
                   "mean_abs_dp": float(dp.mean()), "max_abs_dp": float(dp.max()),
                   "tst_change_minutes": 0.5 * float(np.isin(pred.stage, STAGES[1:]).sum() - base_sleep)}
            rows.append(row | (psg_metrics(psg, pred) if psg is not None else {}))
        print(f"{cohort}/{stem}", {r["mode"]: round(r["same_stage"], 4) for r in rows[-len(RUNS):]}, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "nights.csv", index=False, float_format="%.6g")
    columns = AGREEMENT + [m for m in METRICS if m in table.columns]
    summary = table.groupby(["cohort", "mode"], sort=False)[columns].mean()
    summary.to_csv(out / "summary.csv", float_format="%.6g")
    pairs = paired(table, [m for m in METRICS if m in table.columns])
    pairs.to_csv(out / "paired.csv", index=False, float_format="%.6g")
    print(summary.round(4).to_string())
    print(pairs.round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), nights(Path(sys.argv[1]), sys.argv[4:]))

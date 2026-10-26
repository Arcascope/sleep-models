"""The two adult releases, `adult` (F2 ×3, seeds 0 1 2) and `adult-small` (F2 ×1, seed 0), per night.

Usage (worktree root, ``PYTHONPATH=core/src:bidoze/src``, ``XLA_PYTHON_CLIENT_PREALLOCATE=false``):
    python bidoze/experiments/adult_release/compare_sizes.py RELEASE_DIR

``RELEASE_DIR`` holds the built bundles ``adult/`` and ``adult-small/``; ``RELEASE_DIR/comparison/``
receives ``per_night.csv``, ``paired.csv`` and ``tables.md``. Both releases decode with the stage
divisor in their ``config.json`` (the same one).

Cohorts and where their nights are scored:

- SleepAccel (28) and Wearanize+ (92): the fold phases' predictions of the nights each fold held
  out; the mean output over the release's seeds, each fold's own divisor.
- SleepAccel-Clinical (32): the final phase's predictions, mean over the release's seeds, the
  bundle's divisor.
- AAUWSS (13): cached features through the bundle's ``Predictor`` network.
- EESM19 (33): the raw accelerometer through ``Predictor.predict`` with the GGIR filter off,
  as ``model_grid/build.py`` and ``new_cohorts/eesm19/score_releases.py`` do.

The night sets, labels, loaders and metric definitions are those of ``model_grid/build.py``
(``GRID``, a worktree that holds it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GRID = Path("/home/eric/Engineering/Work/autopisces/sleep-models-model-grid/bidoze/experiments")
sys.path[:0] = [str(GRID / "model_grid")]
import build as g  # noqa: E402  (the grid's loaders and metrics; it extends sys.path for its own helpers)
from arcasleep.divisor import divisor_vector, hybrid_calls  # noqa: E402
from arcasleep_bidoze.release import (ADULT_CV_COHORTS, ADULT_FINAL_COHORT, bundle_seeds,  # noqa: E402
                                      phase_divisor, seed_mean_recordings)
from score_releases import EPOCH_S, STAGE_INDEX, night_metrics, read_accel, read_pairs  # noqa: E402

RELEASES = {"adult": "adult", "adult-small": "adult-small"}
COHORTS = ["SleepAccel", "Wearanize+", "SleepAccel-Clinical", "AAUWSS", "EESM19"]
METRICS = [("kappa_sleep_wake", "Sleep/wake κ"), ("kappa_4stage", "4-stage κ"),
           ("sleep_sensitivity_pct", "Sleep sensitivity (%)"), ("wake_specificity_pct", "Wake specificity (%)"),
           ("tst_error_min", "TST error (min, model − PSG)"), ("abs_err_light_min", "Light time error (min, absolute)"),
           ("abs_err_deep_min", "Deep time error (min, absolute)"), ("abs_err_rem_min", "REM time error (min, absolute)"),
           ("deep_share_pct", "Deep share of predicted sleep (%)")]


def recorded_night(logger, y, proba, calls) -> dict:
    """Metrics of one night from PSG stages (-1 unscored), the 5-way probabilities and the calls."""
    m = g.subject_metrics(logger, y, proba, calls)
    scored = (y >= 0) & (y < 4)
    predicted_sleep = scored & (calls >= 1) & (calls <= 3)
    return dict(kappa_sleep_wake=m["kappa_sleep_wake"], kappa_4stage=m["kappa_4stage"],
                sleep_sensitivity_pct=100 * m["sleep_sensitivity"], wake_specificity_pct=100 * m["wake_specificity"],
                tst_error_min=m["tst_error_minutes"], abs_err_light_min=m["light_error_minutes"],
                abs_err_deep_min=m["deep_error_minutes"], abs_err_rem_min=m["rem_error_minutes"],
                deep_share_pct=100 * float(np.mean(calls[predicted_sleep] == 2)) if predicted_sleep.any() else np.nan)


def cross_validation_and_final(logger, name: str, bundle: Path, training: Path, train_cache: Path) -> list[dict]:
    seeds = bundle_seeds(bundle)
    rows = []
    for fold in sorted(training.glob("fold[0-9]*"), key=lambda p: int(p.name[len("fold"):])):
        divisor, _, _ = phase_divisor(fold, train_cache)
        test_sets = json.loads((fold / "config/cv.json").read_text())["test_sets"]
        for cohort, index in ADULT_CV_COHORTS.items():
            for sid, (y, mean) in seed_mean_recordings(fold, test_sets[index], seeds).items():
                rows.append(dict(model=name, dataset=cohort, night=sid,
                                 **recorded_night(logger, y, mean, hybrid_calls(mean, mean, divisor))))
    divisor = divisor_vector(json.loads((bundle / "config.json").read_text())["stage_divisor"])
    for sid, (y, mean) in seed_mean_recordings(training / "final", ADULT_FINAL_COHORT, seeds).items():
        rows.append(dict(model=name, dataset=ADULT_FINAL_COHORT, night=sid,
                         **recorded_night(logger, y, mean, hybrid_calls(mean, mean, divisor))))
    return rows


def aauwss(logger, name: str, predictor) -> list[dict]:
    from arcasleep.data.batching import pool_frames_to_epochs, split_gap_row
    from arcasleep.data.bundle import CacheBundle

    rows = []
    for sid, path in g.cache_nights(g.CACHE_PREFIX["AAUWSS"]).items():
        b = CacheBundle.from_npz(path)
        y = b.y_for_n_classes(4)[0]
        n = len(y)
        is_gap = split_gap_row(pool_frames_to_epochs(b.X[0][None], np.zeros((1, n)))[0])[1] >= 1.0
        logits = g.model_logits(predictor.model, predictor.context_epochs, predictor.model.config.frames_per_epoch,
                                b.X[0], n, predictor.epoch_multiple)
        mean = np.exp(logits)
        stage = mean[:, :4] / mean[:, :4].sum(1, keepdims=True)
        calls = np.where(is_gap, 4, hybrid_calls(stage, stage, predictor.divisor))
        rows.append(dict(model=name, dataset="AAUWSS", night=sid, **recorded_night(logger, y, mean, calls)))
    return rows


def eesm19(predictors: dict) -> list[dict]:
    manifest = json.loads((g.EESM_RAW / "conversion_manifest.json").read_text())
    rows = []
    for sid in sorted(m["id"] for m in manifest if m["status"] == "converted"):
        psg = read_pairs(g.EESM_RAW / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        times, y = psg[:, 0], psg[:, 1].astype(int)
        scored = y >= 0
        accel = read_accel(g.EESM_RAW / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        pos = {int(round(t / EPOCH_S)): i for i, t in enumerate(times)}
        for name, predictor in predictors.items():
            result = predictor.predict(accel, origin=float(times[0]), ggir_sleep_filter=False, subject_id=sid)
            pred = np.full(len(times), 4)
            for start, label in zip(result.epoch_start_s, result.raw_labels):
                i = pos.get(int(round(start / EPOCH_S)))
                if i is not None:
                    pred[i] = STAGE_INDEX[str(label)]
            m = night_metrics(y[scored], pred[scored])
            rows.append(dict(model=name, dataset="EESM19", night=sid, kappa_sleep_wake=m["kappa_sleep_wake"],
                             kappa_4stage=m["kappa_4stage"], sleep_sensitivity_pct=100 * m["sleep_sensitivity"],
                             wake_specificity_pct=100 * m["wake_specificity"], tst_error_min=m["tst_error_min"],
                             abs_err_light_min=m["abs_err_light_min"], abs_err_deep_min=m["abs_err_deep_min"],
                             abs_err_rem_min=m["abs_err_rem_min"], deep_share_pct=m["deep_pct_pred"]))
        print("scored", sid, flush=True)
    return rows


def interval(values) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def paired_table(nights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        big = nights[(nights.model == "adult") & (nights.dataset == cohort)].set_index("night")
        small = nights[(nights.model == "adult-small") & (nights.dataset == cohort)].set_index("night")
        for metric, _ in METRICS:
            d = (big[metric] - small.loc[big.index, metric]).dropna().to_numpy(float)
            mean, lo, hi = interval(d)
            p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            rows.append(dict(dataset=cohort, metric=metric, n=len(d), mean_diff=mean, ci_low=lo, ci_high=hi,
                             median_diff=float(np.median(d)), wilcoxon_p=p))
    return pd.DataFrame(rows)


def fmt(v: float, metric: str) -> str:
    return f"{v:.3f}".replace("-", "−") if metric.startswith("kappa") else f"{v:.1f}".replace("-", "−")


def tables(nights: pd.DataFrame, paired: pd.DataFrame) -> str:
    out = ["# adult (F2 ×3) and adult-small (F2 ×1, seed 0), v0.2.0", "",
           "Both decode with the release's stage divisor. Mean over nights with the 95% t-interval; "
           "`adult − adult-small` is the mean paired difference over nights with the Wilcoxon p "
           "(p ≤ 0.05 in bold). TST error is model − PSG.", ""]
    for cohort in COHORTS:
        t = nights[nights.dataset == cohort]
        out += [f"## {cohort} (n = {t.night.nunique()} nights)", "",
                "| Release | " + " | ".join(label for _, label in METRICS) + " |",
                "|:---|" + "---:|" * len(METRICS)]
        for name in RELEASES:
            tt = t[t.model == name]
            cells = []
            for metric, _ in METRICS:
                mean, lo, hi = interval(tt[metric])
                cells.append(f"{fmt(mean, metric)} [{fmt(lo, metric)}, {fmt(hi, metric)}]")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        cells = []
        for metric, _ in METRICS:
            r = paired[(paired.dataset == cohort) & (paired.metric == metric)].iloc[0]
            p = "p < 0.001" if r.wilcoxon_p < 0.001 else f"p = {r.wilcoxon_p:.3f}"
            s = f"{fmt(r.mean_diff, metric)} ({p})"
            cells.append(f"**{s}**" if r.wilcoxon_p <= 0.05 else s)
        out.append("| adult − adult-small | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main(release_dir: Path) -> None:
    from arcasleep.predict import Predictor

    bundles = {name: release_dir / path for name, path in RELEASES.items()}
    predictors = {name: Predictor(path) for name, path in bundles.items()}
    logger = g.make_logger()
    rows = []
    for name, bundle in bundles.items():
        rows += cross_validation_and_final(logger, name, bundle, g.F2_TRAINING, g.F2_TRAIN_CACHE)
        rows += aauwss(logger, name, predictors[name])
        print("scored", name, flush=True)
    rows += eesm19(predictors)
    nights = pd.DataFrame(rows)
    nights["night"] = nights.night.astype(str)
    out = release_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    nights.to_csv(out / "per_night.csv", index=False)
    paired = paired_table(nights)
    paired.to_csv(out / "paired.csv", index=False)
    (out / "tables.md").write_text(tables(nights, paired) + "\n")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""Model x dataset comparison of the recent BiDoze models: cells.csv, tables.md, grid.png.

Usage (worktree root, ``PYTHONPATH=core/src:bidoze/src``, ``XLA_PYTHON_CLIENT_PREALLOCATE=false``):
    python bidoze/experiments/model_grid/build.py OUT_DIR [--rescore]

A cell is the mean over nights the model did not train on (an external cohort, or held-out
cross-validation folds / draws pooled). A model that trained on all of a dataset, with no held-out
score, is marked trained on and carries no number. Existing numbers are read from the logged runs'
per-night CSVs; cells with no logged score are scored here (``setup/self_scored_nights.csv`` holds
those nights; delete it or pass ``--rescore`` to score again):

- Pediatric v0.1.1 on SleepAccel, Clinical, Wearanize+ and AAUWSS, and Adult F2 x3 v0.2.0 on AAUWSS,
  from the cached features through the bundle's network (the way ``new_cohorts/score_models.py``
  scores bundles);
- g-wearanize, f2-base, f2-wearanize and the five single-cohort runs on the 33 EESM19 nights, from the
  raw accelerometer through the release bundles' feature path (``new_cohorts/eesm19/score_releases.py``)
  and the runs' own checkpoints;
- TST error of the three draw arms (the logged analysis has no TST column), recomputed from the
  runs' predictions with the logged decoding; the logged kappa and wake specificity are used as is
  and checked against the recomputation;
- Adult F2 x3 v0.2.0 on SleepAccel-Clinical, from its final run's predictions with the bundle's divisor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[1]
sys.path[:0] = [str(BIDOZE / "experiments/adult_devices"), str(BIDOZE / "experiments/new_cohorts/eesm19")]

import arcasleep.evaluation  # noqa: E402,F401  (registers the metrics)
from arcasleep.data.cache import discover_cached_subjects  # noqa: E402
from arcasleep.training.config import load_metrics_config  # noqa: E402
from arcasleep_bidoze.adult import adult_stage_divisor, hybrid_calls  # noqa: E402
from card_metrics import subject_metrics  # noqa: E402
from pisces_lite.metrics.logger import MetricsLogger  # noqa: E402
from score import BETA, CACHE_NAME, load_subject, night_counts  # noqa: E402

ROOT = Path("/home/eric/Engineering/Work/autopisces")
MAIN = ROOT / "sleep-models"
R = MAIN / "bidoze/results"
WT = ROOT / "sleep-models-wearanize-training/bidoze/results/wearanize-training-2026-10-22"
NEW = ROOT / "sleep-models-new-cohorts-score/bidoze/results/new-cohorts-2026-10-22/scores"
FEATURES = MAIN / "feature_cache" / CACHE_NAME
BASELINE = R / "bidoze-joint-ensemble-2026-10-15"
LABELS = R / "bidoze-adult-devices-0/setup/labels"
EESM_RAW = Path("/media/eric/Extreme SSD/pisces2-data/eesm19-pisces")
EESM_RUN = R / "eesm19-eval-2026-10-23"
PED_BUNDLE = R / "pediatric-weaver-all-2026-10-23/training/final/bundle"
F2_BUNDLE = EESM_RUN / "bundles/adult-f2x3-v0.2.0"
F2_FINAL = R / "adult-f2-2026-10-23/training/final"

DATASETS = ["SleepAccel", "SleepAccel-Clinical", "Wearanize+", "AAUWSS", "EESM19", "Weaver (children)"]
CACHE_PREFIX = {"SleepAccel": "SleepAccel", "SleepAccel-Clinical": "SleepAccel-Clinical",
                "Wearanize+": "wearanize", "AAUWSS": "aauwss"}
METRICS = ["kappa_sleep_wake", "kappa_4stage", "wake_specificity_pct", "tst_error_min"]
TITLES = {"kappa_sleep_wake": "Sleep/wake κ", "kappa_4stage": "4-stage κ",
          "wake_specificity_pct": "Wake specificity (%)", "tst_error_min": "TST error (min, model − PSG)"}
DRAW_ARMS = ["g-wearanize", "f2-base", "f2-wearanize"]
DRAWS = range(5)
SEEDS = range(3)

PED, ADULT, F2X3 = "Pediatric v0.1.1", "Adult v0.1.0", "Adult F2 ×3 v0.2.0"
BWO, F2WO3, F2WO0 = "BiDoze Wearanize+ only", "F2 Wearanize+ only, loss 0.3", "F2 Wearanize+ only, loss 0"
F2SA, F2AA = "F2 SleepAccel only", "F2 AAUWSS only"
MODELS = [PED, ADULT, *DRAW_ARMS, BWO, F2WO3, F2WO0, F2SA, F2AA, F2X3]
SINGLE_RUNS = {BWO: "wearanize-only-bidoze", F2WO3: "wearanize-only-f2", F2WO0: "wearanize-only-f2-no-strand-loss",
               F2SA: "sleepaccel-only-f2-no-strand-loss", F2AA: "aauwss-only-f2-no-strand-loss"}
#: model -> dataset it trained on entirely, with no held-out score.
TRAINED = {
    **{m: {"Weaver (children)": "trained on all 634 revised Weaver recordings"} for m in DRAW_ARMS},
    BWO: {"Wearanize+": "trained on all 92 Wearanize+ nights"},
    F2WO3: {"Wearanize+": "trained on all 92 Wearanize+ nights"},
    F2WO0: {"Wearanize+": "trained on all 92 Wearanize+ nights"},
    F2SA: {"SleepAccel": "trained on all 28 SleepAccel nights"},
    F2AA: {"AAUWSS": "trained on all 13 AAUWSS nights"},
    F2X3: {"SleepAccel": "trained on all SleepAccel nights", "Wearanize+": "trained on all 92 Wearanize+ nights",
           "Weaver (children)": "trained on all revised Weaver recordings"},
}
#: (model, dataset) -> why the cell is empty.
EMPTY = {
    (ADULT, "Weaver (children)"): "trained on a matched subset of Weaver; no held-out Weaver score exists",
    **{(m, "Weaver (children)"): "Weaver not scored for this run (not trained on Weaver; 634 recordings not scored here)"
       for m in SINGLE_RUNS},
}
DECODING = {PED: "argmax", ADULT: "hybrid (bundle divisor)", F2X3: "hybrid (bundle divisor)",
            **{m: "hybrid (draw divisor)" for m in DRAW_ARMS}, **{m: "argmax" for m in SINGLE_RUNS}}


def cohort_tag(name: str) -> str:
    return {"Wearanize": "Wearanize+"}.get(name, name)


# ---------------------------------------------------------------- existing per-night scores


def frame(model, dataset, night, draw, kw, ks, spec, tst, source):
    return pd.DataFrame({"model": model, "dataset": dataset, "night": night, "draw": draw,
                         "kappa_sleep_wake": kw, "kappa_4stage": ks, "wake_specificity_pct": 100 * np.asarray(spec, float),
                         "tst_error_min": tst, "source": source})


def logged_release_rows() -> pd.DataFrame:
    out = []
    # Pediatric v0.1.1: EESM19 from the EESM19 release scoring, Weaver from its 5 held-out CV folds.
    per = pd.read_csv(EESM_RUN / "per_night.csv")
    for model, name in ((PED, "Pediatric v0.1.1"), (ADULT, "Adult v0.1.0"), (F2X3, "Adult F2 x3 v0.2.0")):
        t = per[per.model == name]
        out.append(frame(model, "EESM19", t.night, -1, t.kappa_sleep_wake, t.kappa_4stage, t.wake_specificity,
                         t.tst_error_min, "bidoze/results/eesm19-eval-2026-10-23/per_night.csv"))
    folds = pd.concat([pd.read_csv(R / f"pediatric-weaver-all-2026-10-23/training/fold{i}/cv_results_4_stages.csv")
                       for i in range(5)])
    w = folds.pivot_table(index="subject_id", columns="metric", values="value")
    out.append(frame(PED, "Weaver (children)", w.index.astype(str), -1, w.kappa_sleep_wake, w.kappa_4stage,
                     w.wake_specificity, w.tst_error_minutes,
                     "bidoze/results/pediatric-weaver-all-2026-10-23/training/fold{0..4}/cv_results_4_stages.csv"))
    # Adult v0.1.0: SleepAccel (cross-validated) and Clinical from its card metrics; the new cohorts from their scoring.
    cm = pd.read_csv(R / "bidoze-adult-v0.1.0/card-metrics/subjects.csv")
    for cohort in ("SleepAccel", "SleepAccel-Clinical"):
        t = cm[(cm.model == "adult") & (cm.cohort == cohort)]
        out.append(frame(ADULT, cohort, t.subject_id.astype(str), -1, t.kappa_sleep_wake, t.kappa_4stage,
                         t.wake_specificity, t.tst_error_minutes, "bidoze/results/bidoze-adult-v0.1.0/card-metrics/subjects.csv"))
    nc = pd.read_csv(NEW / "per_night.csv")
    for cohort in ("Wearanize+", "AAUWSS"):
        t = nc[(nc.model == "v0.1.0 bundle") & (nc.decoding == "hybrid") & (nc.cohort == cohort)]
        out.append(frame(ADULT, cohort, t.subject_id, -1, t.kappa_sleep_wake, t.kappa_4stage, t.wake_specificity,
                         t.tst_bias_min, "sleep-models-new-cohorts-score/bidoze/results/new-cohorts-2026-10-22/scores/per_night.csv"))
    return pd.concat(out)


def logged_single_rows() -> pd.DataFrame:
    out = []
    for model, run in SINGLE_RUNS.items():
        d = pd.read_csv(R / run / "analysis/subject_metrics.csv")
        for cohort, t in d.groupby("cohort"):
            out.append(frame(model, cohort, t.subject, -1, t.kappa_sleep_wake, t.kappa_4stage, t.wake_specificity,
                             t.tst_error_minutes, f"bidoze/results/{run}/analysis/subject_metrics.csv"))
    return pd.concat(out)


# ---------------------------------------------------------------- draw arms (wearanize-training run)


def draw_divisor(draw: int, bidsleep: np.ndarray) -> np.ndarray:
    name = f"draw{draw}"
    cv = json.loads((BASELINE / f"setup/cv/{name}.json").read_text())
    cache = BASELINE / f"setup/cache/{name}" / CACHE_NAME
    paths = [p for t in cv["train_sets"] for p in sorted(cache.glob(f"{t}_subject_*"))]
    return adult_stage_divisor(night_counts(paths), bidsleep, BETA)


def bidsleep_counts() -> np.ndarray:
    return night_counts([s.path for s in discover_cached_subjects(["bidsleep"], LABELS / CACHE_NAME)["bidsleep"]])


def make_logger() -> MetricsLogger:
    config, _ = load_metrics_config(BIDOZE / "config/metrics.json")
    return MetricsLogger(config)


def draw_arm_rows(logger) -> pd.DataFrame:
    """Per night and draw: the logged kappas and wake specificity, and TST error recomputed from the predictions."""
    logged = pd.read_csv(WT / "analysis/subjects.csv", dtype={"subject_id": str})
    logged = logged[logged.decoding == "hybrid"].copy()
    logged["cohort"] = logged.cohort.map(cohort_tag)
    sets = {"testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical", "testWearanize": "Wearanize+",
            "testAAUWSS": "AAUWSS"}
    bids = bidsleep_counts()
    rows = []
    for d in DRAWS:
        divisor = draw_divisor(d, bids)
        for arm in DRAW_ARMS:
            dirs = [WT / f"training/{arm}/draw{d}/predictions/draw{d}_seed_{s}" for s in SEEDS]
            for dataset, cohort in sets.items():
                for f in sorted(dirs[0].glob(f"{dataset}_*.csv")):
                    sid = f.stem[len(dataset) + 1:]
                    loaded = [load_subject(p / f.name) for p in dirs]
                    y, proba = loaded[0][0], np.mean([q for _, q in loaded], axis=0)
                    m = subject_metrics(logger, y, proba, hybrid_calls(proba, proba, divisor))
                    rows.append(dict(model=arm, dataset=cohort, night=sid, draw=d, re_kw=m["kappa_sleep_wake"],
                                     re_ks=m["kappa_4stage"], re_spec=m["wake_specificity"], tst_error_min=m["tst_error_minutes"]))
    re = pd.DataFrame(rows)
    j = re.merge(logged, left_on=["model", "dataset", "night", "draw"], right_on=["arm", "cohort", "subject_id", "draw"],
                 how="left", validate="one_to_one")
    assert j.kappa_4stage.notna().all(), "a recomputed night is missing from the logged subjects.csv"
    diff = max((j.re_kw - j.kappa_sleep_wake).abs().max(), (j.re_ks - j.kappa_4stage).abs().max(),
               (j.re_spec - j.wake_specificity).abs().max())
    print(f"draw arms: {len(j)} night-draws; max |recomputed - logged| over kappas and wake specificity = {diff:.2e}")
    assert diff < 1e-6
    return frame(j.model, j.dataset, j.night, j.draw, j.kappa_sleep_wake, j.kappa_4stage, j.wake_specificity,
                 j.tst_error_min, "sleep-models-wearanize-training/bidoze/results/wearanize-training-2026-10-22/analysis/subjects.csv "
                 "(kappa, wake specificity); TST error recomputed from training/<arm>/draw<d>/predictions")


# ---------------------------------------------------------------- self-scoring


def model_logits(model, context, fpe, X, n):
    """Per-epoch network output for one recording of cached features, windowed as ``Predictor.predict`` does."""
    from arcasleep.data.batching import pad_recording_input, recording_input
    inputs = recording_input(X, n, fpe)
    chunks = [pad_recording_input(inputs[s * fpe:(s + context) * fpe], context, fpe) for s in range(0, n, context)]
    out = model.infer(np.stack(chunks))
    return out.reshape(-1, out.shape[-1])[:n]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def cache_nights(prefix):
    return {p.name[len(prefix) + 9:-len("_features.npz")]: p for p in sorted(FEATURES.glob(f"{prefix}_subject_*_features.npz"))}


def score_cache_bundle(logger, name, bundle, cohorts, adult: bool) -> pd.DataFrame:
    from arcasleep.data.batching import pool_frames_to_epochs, split_gap_row
    from arcasleep.data.bundle import CacheBundle
    from arcasleep.predict import Predictor
    from arcasleep_bidoze.adult import AdultPredictor
    if adult:
        pr = AdultPredictor(bundle)
        ens, divisor = pr.ensemble, pr.divisor
    else:
        ens, divisor = Predictor(bundle), None
    fpe = ens.model.config.frames_per_epoch
    rows = []
    for cohort in cohorts:
        for sid, path in cache_nights(CACHE_PREFIX[cohort]).items():
            b = CacheBundle.from_npz(path)
            y = b.y_for_n_classes(4)[0]
            n = len(y)
            pooled = pool_frames_to_epochs(b.X[0][None], np.zeros((1, n)))[0]
            is_gap = split_gap_row(pooled)[1] >= 1.0
            logits = model_logits(ens.model, ens.context_epochs, fpe, b.X[0], n)
            if adult:
                mean = np.exp(logits)
                stage = mean[:, :4] / mean[:, :4].sum(1, keepdims=True)
                calls, proba = hybrid_calls(stage, stage, divisor), mean
            else:
                stage = softmax(logits[:, :4])
                calls = stage.argmax(1)
                proba = softmax(logits) if logits.shape[1] > 4 else np.column_stack([stage, np.zeros(n)])
            calls = np.where(is_gap, 4, calls)
            m = subject_metrics(logger, y, proba, calls)
            rows.append(dict(model=name, dataset=cohort, night=sid, draw=-1, kappa_sleep_wake=m["kappa_sleep_wake"],
                             kappa_4stage=m["kappa_4stage"], wake_specificity_pct=100 * m["wake_specificity"],
                             tst_error_min=m["tst_error_minutes"], source="scored here (cached features)"))
        print("scored", name, cohort, flush=True)
    return pd.DataFrame(rows)


def final_clinical_rows(logger) -> pd.DataFrame:
    """Adult F2 x3 v0.2.0 on SleepAccel-Clinical: the 3 final seeds' mean output, hybrid with the bundle's divisor."""
    spec = json.loads((F2_BUNDLE / "composite.json").read_text())
    divisor = np.array([1.0] + [float(spec["stage_divisor"][k]) for k in ("light", "deep", "rem")])
    dirs = [F2_FINAL / f"predictions/final_seed_{s}" for s in SEEDS]
    rows = []
    for f in sorted(dirs[0].glob("SleepAccel-Clinical_*.csv")):
        loaded = [load_subject(p / f.name) for p in dirs]
        y, proba = loaded[0][0], np.mean([q for _, q in loaded], axis=0)
        m = subject_metrics(logger, y, proba, hybrid_calls(proba, proba, divisor))
        rows.append(dict(model=F2X3, dataset="SleepAccel-Clinical", night=f.stem[len("SleepAccel-Clinical_"):], draw=-1,
                         kappa_sleep_wake=m["kappa_sleep_wake"], kappa_4stage=m["kappa_4stage"],
                         wake_specificity_pct=100 * m["wake_specificity"], tst_error_min=m["tst_error_minutes"],
                         source="scored here (bidoze/results/adult-f2-2026-10-23/training/final/predictions, bundle divisor)"))
    return pd.DataFrame(rows)


def checkpoints():
    """Ensembles scored on EESM19: name -> list of (draw, [3 checkpoint paths], divisor draw or None)."""
    out = {}
    for arm in DRAW_ARMS:
        out[arm] = [(d, [WT / f"training/{arm}/draw{d}/model_outputs/model_draw{d}_seed_{s}.pisces" for s in SEEDS]) for d in DRAWS]
    for model, run in SINGLE_RUNS.items():
        out[model] = [(-1, [R / run / f"model_outputs/model_0_seed_{s}.pisces" for s in SEEDS])]
    return out


def eesm_rows(logger) -> pd.DataFrame:
    """The 33 EESM19 nights scored as ``score_releases.py`` does, with each run's own checkpoints."""
    from arcasleep.data.batching import pool_frames_to_epochs, split_gap_row
    from arcasleep.models import get_family
    from arcasleep.predict import Predictor
    from arcasleep_bidoze.adult import StackedModel
    from score_releases import (EPOCH_S, STAGE_INDEX, night_metrics, read_accel,  # noqa: F401
                                read_pairs)
    fam = get_family("bidoze")
    ensembles, divisors = {}, {}
    bids = bidsleep_counts()
    for d in DRAWS:
        divisors[d] = draw_divisor(d, bids)
    for name, groups in checkpoints().items():
        for d, paths in groups:
            ensembles[(name, d)] = StackedModel([fam.load_checkpoint(p) for p in paths])
    # Check path: the adult F2 final seeds through this route must equal the bundle's logged scores.
    check = StackedModel([fam.load_checkpoint(F2_FINAL / f"model_outputs/model_final_seed_{s}.pisces") for s in SEEDS])
    spec = json.loads((F2_BUNDLE / "composite.json").read_text())
    f2_div = np.array([1.0] + [float(spec["stage_divisor"][k]) for k in ("light", "deep", "rem")])
    logged = pd.read_csv(EESM_RUN / "per_night.csv")
    logged = logged[logged.model == "Adult F2 x3 v0.2.0"].set_index("night")

    feat = Predictor(PED_BUNDLE)  # preprocessing is identical in every bundle (same preprocessing.json)
    manifest = json.loads((EESM_RAW / "conversion_manifest.json").read_text())
    ids = sorted(n["id"] for n in manifest if n["status"] == "converted")
    rows, checked = [], []
    for sid in ids:
        psg = read_pairs(EESM_RAW / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        times, y = psg[:, 0], psg[:, 1].astype(int)
        scored = y >= 0
        accel = read_accel(EESM_RAW / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        frames, origin = feat.features(accel, origin=float(times[0]))
        n = frames.shape[0] // feat.frames_per_epoch
        pooled = pool_frames_to_epochs(frames[None], np.zeros((1, n)))[0]
        is_gap = split_gap_row(pooled)[1] >= 1.0
        starts = origin + EPOCH_S * np.arange(n)
        pos = {int(round(t / EPOCH_S)): i for i, t in enumerate(times)}
        index = [(pos.get(int(round(s / EPOCH_S))), k) for k, s in enumerate(starts)]

        def night(stack, mode, divisor):
            logits = model_logits(stack, feat.context_epochs, stack.config.frames_per_epoch, frames, n)
            p = np.exp(logits)
            calls = p.argmax(1) if mode == "argmax" else hybrid_calls(p, p, divisor)
            calls = np.where(is_gap, 4, calls)
            pred = np.full(len(times), 4)
            for i, k in index:
                if i is not None:
                    pred[i] = calls[k]
            return night_metrics(y[scored], pred[scored])

        if len(checked) < 3:
            m = night(check, "hybrid", f2_div)
            ref = logged.loc[sid]
            gap = max(abs(m["kappa_sleep_wake"] - ref.kappa_sleep_wake), abs(m["kappa_4stage"] - ref.kappa_4stage),
                      abs(m["tst_error_min"] - ref.tst_error_min))
            checked.append(gap)
            print(f"check {sid}: checkpoint route vs bundle route, max difference {gap:.2e}", flush=True)
            assert gap < 1e-3
        for (name, d), stack in ensembles.items():
            mode = "hybrid" if name in DRAW_ARMS else "argmax"
            m = night(stack, mode, divisors.get(d))
            rows.append(dict(model=name, dataset="EESM19", night=sid, draw=d, kappa_sleep_wake=m["kappa_sleep_wake"],
                             kappa_4stage=m["kappa_4stage"], wake_specificity_pct=100 * m["wake_specificity"],
                             tst_error_min=m["tst_error_min"], source="scored here (raw EESM19 accelerometer, run checkpoints)"))
        print("scored", sid, flush=True)
    return pd.DataFrame(rows)


def self_scored(out: Path, rescore: bool) -> pd.DataFrame:
    path = out / "setup/self_scored_nights.csv"
    if path.exists() and not rescore:
        return pd.read_csv(path, dtype={"night": str})
    logger = make_logger()
    parts = [score_cache_bundle(logger, PED, PED_BUNDLE, ["SleepAccel", "SleepAccel-Clinical", "Wearanize+", "AAUWSS"], False),
             score_cache_bundle(logger, F2X3, F2_BUNDLE, ["AAUWSS"], True),
             final_clinical_rows(logger), draw_arm_rows(logger), eesm_rows(logger)]
    df = pd.concat(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------- cells


def t_interval(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def cell_stats(t: pd.DataFrame, metric: str):
    """Mean over draws of per-draw means (a single value per night when there are no draws); CI over nights."""
    mean = t.groupby("draw")[metric].mean().mean()
    per_night = t.groupby("night")[metric].mean()
    _, lo, hi = t_interval(per_night)
    return float(mean), lo, hi, int(per_night.notna().sum())


def build_cells(nights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            t = nights[(nights.model == model) & (nights.dataset == dataset)]
            trained = TRAINED.get(model, {}).get(dataset)
            reason = EMPTY.get((model, dataset))
            if trained is not None:
                protocol, source = "trained on all of this dataset; no held-out score", ""
            elif t.empty:
                protocol, source = "no score", ""
                reason = reason or "no score available"
            else:
                source = "; ".join(sorted(t.source.unique()))
                scored = source.startswith("scored here")
                if dataset == "Weaver (children)":
                    protocol = "held-out CV folds pooled"
                elif model in DRAW_ARMS:
                    protocol = ("held-out draw subjects, mean over draws of per-draw means"
                                if dataset != "EESM19" else "external cohort; mean over 5 draw ensembles")
                    if dataset == "AAUWSS" or (dataset == "Wearanize+" and model == "f2-base"):
                        protocol = "external cohort; mean over draws of per-draw means"
                elif model == ADULT and dataset == "SleepAccel":
                    protocol = "held-out CV folds pooled"
                else:
                    protocol = "external cohort"
                if scored:
                    protocol += "; scored for this table"
            for metric in METRICS:
                if trained is not None or t.empty:
                    mean = lo = hi = np.nan
                    n = 0
                else:
                    mean, lo, hi, n = cell_stats(t, metric)
                rows.append(dict(model=model, dataset=dataset, metric=metric, mean=mean, ci_low=lo, ci_high=hi,
                                 n_nights=n, source_file=source, protocol=protocol, decoding=DECODING[model],
                                 trained_on=trained is not None,
                                 scored_here=bool(source.startswith("scored here") or "recomputed" in source),
                                 note=trained or reason or ""))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables and figure


def fmt(x, metric):
    if pd.isna(x):
        return ""
    d = 2 if metric.startswith("kappa") else 1
    return f"{x:.{d}f}".replace("-", "−")


def matrix(cells, metric):
    c = cells[cells.metric == metric]
    v = c.pivot(index="dataset", columns="model", values="mean").reindex(index=DATASETS, columns=MODELS)
    tr = c.pivot(index="dataset", columns="model", values="trained_on").reindex(index=DATASETS, columns=MODELS).astype(bool)
    return v, tr


def tables_md(cells) -> str:
    out = ["# Model × dataset comparison, 2026-10-23", "",
           "Mean over nights the model did not train on. T: the model trained on all of this dataset and has no held-out "
           "score. Empty: no score (reasons in `setup/README.md`). Confidence intervals and nights per cell are in `cells.csv`.", ""]
    for metric in METRICS:
        v, tr = matrix(cells, metric)
        out += [f"## {TITLES[metric]}", "", "| Dataset | " + " | ".join(MODELS) + " |", "|:---|" + "--:|" * len(MODELS)]
        for ds in DATASETS:
            out.append(f"| {ds} | " + " | ".join("T" if tr.loc[ds, m] else fmt(v.loc[ds, m], metric) for m in MODELS) + " |")
        out.append("")
    return "\n".join(out)


def figure(cells, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    blue = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    green = LinearSegmentedColormap.from_list("seq_aqua", ["#d6f1e5", "#6fcfa6", "#1b8f66", "#0a4a34"])
    div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ee8f8f", "#c02c2c"])
    kmin = min(cells[cells.metric.str.startswith("kappa")]["mean"].min(), 0.2)
    kmax = cells[cells.metric.str.startswith("kappa")]["mean"].max()
    spec = [("kappa_sleep_wake", blue, (kmin, kmax)), ("kappa_4stage", blue, (kmin, kmax)),
            ("wake_specificity_pct", green, None), ("tst_error_min", div, None)]
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5), facecolor=surface)
    for ax, (metric, cmap, lim) in zip(axes.ravel(), spec):
        v, tr = matrix(cells, metric)
        a = v.to_numpy(float)
        if metric == "tst_error_min":
            m = np.nanmax(np.abs(a))
            norm = TwoSlopeNorm(vcenter=0, vmin=-m, vmax=m)
        elif lim:
            norm = plt.Normalize(*lim)
        else:
            norm = plt.Normalize(np.nanmin(a), np.nanmax(a))
        ax.set_facecolor("white")
        ax.imshow(np.ma.masked_invalid(a), cmap=cmap, norm=norm, aspect="auto")
        for i, ds in enumerate(DATASETS):
            for j, mod in enumerate(MODELS):
                if tr.loc[ds, mod]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#c3c2b7", edgecolor="#898781",
                                               hatch="///", linewidth=0))
                    ax.text(j, i, "T", ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
                elif not np.isnan(a[i, j]):
                    rgba = cmap(norm(a[i, j]))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(j, i, fmt(a[i, j], metric), ha="center", va="center", fontsize=9,
                            color="white" if lum < 0.5 else ink)
        ax.set_xticks(range(len(MODELS)), MODELS, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(DATASETS)), DATASETS, fontsize=10)
        ax.set_xticks(np.arange(-.5, len(MODELS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(DATASETS)), minor=True)
        ax.grid(which="minor", color=surface, linewidth=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(TITLES[metric], loc="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.005, "T (hatched grey): trained on all of this dataset, no held-out score. White: not scored. "
             "Mean over held-out nights; κ panels share one colour scale; TST error is centred at 0.",
             fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=100, facecolor=surface)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    nights = pd.concat([logged_release_rows(), logged_single_rows(), self_scored(out, args.rescore)])
    nights["night"] = nights.night.astype(str)
    nights.to_csv(out / "per_night.csv", index=False)
    cells = build_cells(nights)
    cells.to_csv(out / "cells.csv", index=False)
    (out / "tables.md").write_text(tables_md(cells))
    figure(cells, out / "grid.png")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()

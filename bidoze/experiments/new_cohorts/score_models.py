"""Score BiDoze G x3, F2 x1 and F2 x3 on Wearanize+ and AAUWSS, with the two adult release bundles as references.

Usage (repository root, after ``export PYTHONPATH=core/src:bidoze/src:unet/src``):
    python bidoze/experiments/new_cohorts/score_models.py OUT_DIR

``OUT_DIR`` receives per_night.csv, paired.csv and tables.md.

Models (``RUN`` below, 5 draws, none trained on these cohorts):
- ``F2 x1``: strand-fusion seed 0 fusion head.
- ``F2 x3``: mean of the fusion-head probabilities of seeds 0-2.
- ``G x3``: mean of the probabilities of BiDoze-G seeds 0-2.
Release references, scored once: the v0.1.0 adult bundle and the gap-rule adult ``all`` bundle, each
with its own stage divisor. A reference's gap epochs are the recording's epochs with no data (the
bundle's own rule), not a gap-head decision.

Decodings: ``argmax`` is the argmax over wake/light/deep/REM; ``hybrid`` is ``hybrid_calls`` with the
draw's stage divisor. A gap call is a miss. Epochs labelled -1 are excluded.

Wearanize+ side files (scorer 2, U-Sleep) are compared with scorer 1 on the cohort's cleaned layout
(``WEARANIZE_RAW``): epochs are paired by nearest start time, as in the conversion checks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "strand_fusion/analysis"), str(EXP / "strand_fusion"), str(EXP / "adult_devices")]

import data as D  # noqa: E402
import weaver_children as W  # noqa: E402

REPO = HERE.parents[2]
RUN = Path("/home/eric/Engineering/Work/autopisces/sleep-models-strand-fusion/bidoze/results/strand-fusion-2026-10-21")
MAIN = Path("/home/eric/Engineering/Work/autopisces/sleep-models")
FEATURES = MAIN / "feature_cache" / D.CACHE
BUNDLES = {"v0.1.0 bundle": MAIN / "bidoze/results/bidoze-adult-v0.1.0/bundle",
           "gap-rule adult-all bundle": MAIN / "bidoze/results/gap-retrain-2026-10-21/bundles/adult-all"}
WEARANIZE_RAW = Path("/media/eric/Extreme SSD/pisces2-data/wearanize-pisces")
COHORTS = {"Wearanize+": "wearanize", "AAUWSS": "aauwss"}
OVER_65 = {"s02", "s06"}
MODELS = ["F2 x1", "F2 x3", "G x3"]
DECODINGS = ["argmax", "hybrid"]
METRICS = W.METRICS
STG = W.STG
EPOCH_S = 30.0
LAGS = np.arange(-10, 11)
FOUR_STAGE = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3}
SLEEP_WAKE = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1}
BOOTSTRAPS = 10000


def night_files():
    out = {}
    for cohort, prefix in COHORTS.items():
        out[cohort] = {p.name[len(prefix) + 9:-len("_features.npz")]: p
                       for p in sorted(FEATURES.glob(f"{prefix}_subject_*_features.npz"))}
    return out


def night_row(logger, y, p, calls, subject_metrics):
    m = subject_metrics(logger, y, p, calls)
    scored = (y >= 0) & (y < 4)
    row = {k: float(m[k]) for k in METRICS[:4]}
    for i, st in enumerate(STG):
        on = scored & (y == i)
        row[f"recall_{st}"] = float(np.mean(calls[on] == i)) if on.any() else np.nan
    row["tst_bias_min"] = 0.5 * float(np.sum((calls[scored] > 0) & (calls[scored] < 4)) - np.sum(y[scored] > 0))
    return row


def research_rows(logger, nights, subject_metrics):
    import jax
    import jax.numpy as jnp
    from arcasleep_bidoze.adult import hybrid_calls
    from arcasleep_bidoze.model import BiDozeNetModel
    from fusion_model import StrandFusionModel
    batches = {(c, s): D.recording_batch(p) for c, files in nights.items() for s, p in files.items()}
    rows = []
    for d in W.DRAWS:
        divisor = np.load(W.fit_dir(RUN, "F0", d) / "stage_divisor.npy")
        assert np.allclose(divisor, np.load(RUN / "g3" / f"draw{d}" / "stage_divisor.npy"))
        runners = {}
        for k in range(3):
            m = StrandFusionModel.load_research(W.fit_dir(RUN, f"F2s{k}", d))
            runners[f"F2s{k}"] = jax.jit(lambda x, m=m: jnp.exp(m.module.apply(
                {"params": m.params}, x, jnp.ones(x.shape[:2]), return_outputs=True)["log_head"]))
            g = BiDozeNetModel.loadz(W.g_path(RUN, k, d))
            runners[f"G{k}"] = jax.jit(lambda x, g=g: jax.nn.softmax(
                g.module.apply({"params": g.params}, x, jnp.ones(x.shape[:2]), train=False), axis=-1))
        for (cohort, s), (x, y) in batches.items():
            n = len(y)
            probs = {u: np.asarray(f(x))[0, :n] for u, f in runners.items()}
            model_p = {"F2 x1": probs["F2s0"], "F2 x3": np.mean([probs[f"F2s{k}"] for k in range(3)], axis=0),
                       "G x3": np.mean([probs[f"G{k}"] for k in range(3)], axis=0)}
            for name, p in model_p.items():
                for dec in DECODINGS:
                    calls = p[:, :4].argmax(1) if dec == "argmax" else hybrid_calls(p, p, divisor)
                    rows.append(dict(model=name, decoding=dec, draw=d, cohort=cohort, subject_id=s,
                                     **night_row(logger, y, p, calls, subject_metrics)))
        print("scored draw", d, flush=True)
    return rows


def reference_rows(logger, nights, subject_metrics):
    """Each bundle's seeds as one batched call; probabilities and calls as the bundle's predictor makes them."""
    from arcasleep.data.batching import pad_recording_input, pool_frames_to_epochs, recording_input, split_gap_row
    from arcasleep.data.bundle import CacheBundle
    from arcasleep_bidoze.adult import AdultPredictor, hybrid_calls
    rows = []
    for name, path in BUNDLES.items():
        predictor = AdultPredictor(path)
        ensemble = predictor.ensemble
        window = ensemble.context_epochs
        for cohort, files in nights.items():
            for s, f in files.items():
                bundle = CacheBundle.from_npz(f)
                y = bundle.y_for_n_classes(4)[0]
                n = len(y)
                pooled = pool_frames_to_epochs(bundle.X[0][None], np.zeros((1, n)))[0]
                inputs = recording_input(bundle.X[0], n, ensemble.model.config.frames_per_epoch)
                chunks = [pad_recording_input(inputs[a:a + window], window, 1) for a in range(0, n, window)]
                logits = ensemble.model.infer(np.stack(chunks)).reshape(-1, 5)[:n]
                mean = np.exp(logits)
                stage = mean[:, :4] / mean[:, :4].sum(1, keepdims=True)
                is_gap = split_gap_row(pooled)[1] >= 1.0
                for dec in DECODINGS:
                    calls = stage.argmax(1) if dec == "argmax" else hybrid_calls(stage, stage, predictor.divisor)
                    calls = np.where(is_gap, 4, calls)
                    rows.append(dict(model=name, decoding=dec, draw=-1, cohort=cohort, subject_id=s,
                                     **night_row(logger, y, mean, calls, subject_metrics)))
        print("scored", name, flush=True)
    return rows


def read_pairs(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def kappa(a, b):
    classes = np.union1d(a, b)
    if classes.size < 2:
        return float("nan")
    table = np.array([[np.sum((a == i) & (b == j)) for j in classes] for i in classes], float)
    n = table.sum()
    expected = float(table.sum(0) @ table.sum(1)) / n**2
    return (np.trace(table) / n - expected) / (1 - expected) if expected < 1 else float("nan")


def auroc(score, positive):
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rankdata(score)[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mapped(stages, table):
    return np.array([table.get(int(s), -1) for s in stages])


def wearanize_references(ids):
    """Per night: kappa of scorer 1 against scorer 2 and U-Sleep, and the lag-0 AUROC of E4 movement for scorer 1 wake."""
    rows = []
    for sid in ids:
        psg = read_pairs(WEARANIZE_RAW / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        times, stages = psg[:, 0], psg[:, 1].astype(int)
        row = dict(subject_id=sid)
        for name in ("scorer2", "usleep"):
            path = WEARANIZE_RAW / name / f"{sid}_{name}.out"
            if not path.exists():
                continue
            other = read_pairs(path)
            idx = np.abs(other[:, 0][:, None] - times[None, :]).argmin(1)
            near = np.abs(other[:, 0] - times[idx]) <= EPOCH_S / 2
            a, b = stages[idx[near]], other[near, 1].astype(int)
            ok = (a >= 0) & (b >= 0)
            row[f"{name}_kappa_4stage"] = kappa(mapped(a[ok], FOUR_STAGE), mapped(b[ok], FOUR_STAGE))
            row[f"{name}_kappa_sleep_wake"] = kappa(mapped(a[ok], SLEEP_WAKE), mapped(b[ok], SLEEP_WAKE))
        acc = read_pairs(WEARANIZE_RAW / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        magnitude = np.linalg.norm(acc[:, 1:], axis=1)
        change = np.abs(np.diff(magnitude, prepend=magnitude[0]))
        bins = np.append(times, times[-1] + EPOCH_S)
        move = np.log1p(np.histogram(acc[:, 0], bins=bins, weights=change)[0])
        ok = stages >= 0
        row["movement_auroc_lag0"] = auroc(move[ok], stages[ok] == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(per_night):
    rng = np.random.default_rng(0)
    out = []
    for cohort in COHORTS:
        for dec in DECODINGS:
            t = per_night[(per_night.cohort == cohort) & (per_night.decoding == dec)]
            avg = {m: t[t.model == m].groupby("subject_id")[METRICS].mean() for m in MODELS}
            for a in ("F2 x3", "F2 x1"):
                diff = avg[a] - avg["G x3"]
                for metric in METRICS:
                    v = diff[metric].dropna().to_numpy()
                    boot = np.array([rng.choice(v, len(v)).mean() for _ in range(BOOTSTRAPS)])
                    p = float(stats.wilcoxon(v).pvalue) if (v != 0).any() else 1.0
                    out.append(dict(cohort=cohort, decoding=dec, comparison=f"{a} − G x3", metric=metric, n_nights=len(v),
                                    mean_diff=v.mean(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                                    wilcoxon_p=p))
    return pd.DataFrame(out)


SHORT = dict(kappa_4stage="κ 4-stage", kappa_sleep_wake="κ sleep/wake", sleep_sensitivity="sleep sens",
             wake_specificity="wake spec", recall_rem="REM recall", recall_deep="deep recall",
             tst_bias_min="TST bias (min)")


def fmt(x, digits=3):
    return f"{x:.{digits}f}".replace("-", "−")


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + ":---|" * len(header),
                      *["| " + " | ".join(r) + " |" for r in rows]])


def digits(k):
    return 1 if k == "tst_bias_min" else 3


def write_tables(out, per_night, paired, side):
    lines = ["# Wearanize+ and AAUWSS: BiDoze G x3, F2 x1, F2 x3 and the adult release bundles", "",
             "Each night is scored by 5 draws of every research model (none trained on these cohorts) and once by each bundle. "
             "Gap calls are misses; epochs labelled −1 are excluded. Per-cohort values are the mean over nights.", ""]
    for cohort in COHORTS:
        n = per_night[per_night.cohort == cohort].subject_id.nunique()
        for dec in DECODINGS:
            t = per_night[(per_night.cohort == cohort) & (per_night.decoding == dec)]
            rows = []
            for m in MODELS:
                per_draw = t[t.model == m].groupby("draw")[list(SHORT)].mean()
                rows.append([m, *[f"{fmt(per_draw[k].mean(), digits(k))} ± {fmt(per_draw[k].std(ddof=1), digits(k))}"
                                  for k in SHORT]])
            for m in BUNDLES:
                v = t[t.model == m][list(SHORT)].mean()
                rows.append([m, *[fmt(v[k], digits(k)) for k in SHORT]])
            lines += [f"## {cohort} ({n} nights), decoding: {dec}", "",
                      "Research models: mean ± SD over 5 draws. Bundles: one score.", "",
                      md(["model", *SHORT.values()], rows), ""]
    for cohort in COHORTS:
        for dec in DECODINGS:
            sub = paired[(paired.cohort == cohort) & (paired.decoding == dec)]
            rows = []
            for c in sub.comparison.unique():
                r = [c]
                for k in SHORT:
                    x = sub[(sub.comparison == c) & (sub.metric == k)].iloc[0]
                    dg = digits(k)
                    r.append(f"{fmt(x.mean_diff, dg)} [{fmt(x.ci_low, dg)}, {fmt(x.ci_high, dg)}] p={x.wilcoxon_p:.3g}")
                rows.append(r)
            lines += [f"## {cohort}: paired differences, decoding: {dec}", "",
                      "Per-night differences averaged over 5 draws; mean [bootstrap 95% interval over nights] and Wilcoxon p.", "",
                      md(["comparison", *SHORT.values()], rows), ""]
    ceiling = [[name, str(int(side[f"{key}_kappa_4stage"].notna().sum())),
                fmt(side[f"{key}_kappa_4stage"].mean()), fmt(side[f"{key}_kappa_sleep_wake"].mean())]
               for name, key in (("scorer 1 vs scorer 2", "scorer2"), ("scorer 1 vs U-Sleep", "usleep"))]
    lines += ["## Wearanize+: human and automated ceiling", "",
              "Cohen's κ against scorer 1, mean over nights that have the side file.", "",
              md(["comparison", "nights", "κ 4-stage", "κ sleep/wake"], ceiling), ""]
    for dec in DECODINGS:
        t = per_night[(per_night.cohort == "AAUWSS") & (per_night.decoding == dec)]
        avg = {m: t[t.model == m].groupby("subject_id")[METRICS].mean() for m in ("F2 x3", "G x3")}
        rows = []
        for s in sorted(avg["F2 x3"].index):
            tag = " (age > 65)" if s in OVER_65 else ""
            rows.append([s + tag, *[fmt(avg[m].loc[s, k], digits(k)) for m in avg
                                    for k in ("kappa_4stage", "kappa_sleep_wake", "tst_bias_min")]])
        lines += [f"## AAUWSS: per night, draw-mean, decoding: {dec}", "",
                  md(["night", "F2 x3 κ 4-stage", "F2 x3 κ sleep/wake", "F2 x3 TST bias (min)",
                      "G x3 κ 4-stage", "G x3 κ sleep/wake", "G x3 TST bias (min)"], rows), ""]
    for dec in DECODINGS:
        t = per_night[(per_night.cohort == "Wearanize+") & (per_night.decoding == dec) & (per_night.model == "F2 x3")]
        worst = t.groupby("subject_id")[METRICS].mean().sort_values("kappa_4stage").head(10)
        g = per_night[(per_night.cohort == "Wearanize+") & (per_night.decoding == dec) & (per_night.model == "G x3")]
        g = g.groupby("subject_id")[METRICS].mean()
        rows = [[s, fmt(r.kappa_4stage), fmt(g.loc[s, "kappa_4stage"]), fmt(r.kappa_sleep_wake), fmt(r.tst_bias_min, 1),
                 fmt(side.set_index("subject_id").loc[s, "movement_auroc_lag0"])] for s, r in worst.iterrows()]
        lines += [f"## Wearanize+: 10 worst nights by F2 x3 4-stage κ (draw-mean), decoding: {dec}", "",
                  md(["night", "F2 x3 κ 4-stage", "G x3 κ 4-stage", "F2 x3 κ sleep/wake", "F2 x3 TST bias (min)",
                      "lag-0 movement AUROC"], rows), ""]
    (out / "tables.md").write_text("\n".join(lines))


def main(out: Path):
    from arcasleep.training.config import load_metrics_config
    from card_metrics import subject_metrics
    from pisces_lite.metrics.logger import MetricsLogger
    import arcasleep.evaluation  # noqa: F401  (registers the metrics)
    out.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(load_metrics_config(REPO / "bidoze/config/metrics.json")[0])
    nights = night_files()
    print({c: len(f) for c, f in nights.items()})
    rows = research_rows(logger, nights, subject_metrics) + reference_rows(logger, nights, subject_metrics)
    per_night = pd.DataFrame(rows)
    per_night.to_csv(out / "per_night.csv", index=False)
    paired = paired_rows(per_night)
    paired.to_csv(out / "paired.csv", index=False)
    side = wearanize_references(sorted(nights["Wearanize+"]))
    side.to_csv(out / "wearanize_references.csv", index=False)
    write_tables(out, per_night, paired, side)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

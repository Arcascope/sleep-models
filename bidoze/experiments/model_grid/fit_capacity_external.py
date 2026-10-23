"""Score the fit-capacity checkpoints and the adult F2 release on the external adult cohorts.

Usage (worktree root, ``PYTHONPATH=core/src:bidoze/src``, ``XLA_PYTHON_CLIENT_PREALLOCATE=false``):
    python bidoze/experiments/model_grid/fit_capacity_external.py OUT_DIR

Cohorts: SleepAccel-Clinical and AAUWSS from the cached features, EESM19 from the raw accelerometer,
each as ``build.py`` scores them. Every model is one network (seed 0) except the release 3-seed ensemble,
and every model is decoded with the release bundle's stage divisor (``hybrid_calls``) and, for the 4-stage
kappa only, also with the argmax of the 4 stage probabilities.

Writes ``<OUT_DIR>/per_night.csv``, ``paired.csv`` and ``tables.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build as g  # noqa: E402

R = g.R
LOCKED = R / "fit-capacity-lr-locked-2026-10-23/training"
DROP = R / "fit-capacity-2026-10-23/training"
COHORTS = ["SleepAccel-Clinical", "AAUWSS", "EESM19"]
METRICS = [("kappa_sleep_wake", "Sleep/wake κ"), ("kappa_4stage", "4-stage κ (divisor)"),
           ("kappa_4stage_argmax", "4-stage κ (argmax)"), ("wake_specificity_pct", "Wake specificity (%)"),
           ("sleep_sensitivity_pct", "Sleep sensitivity (%)"), ("tst_error_min", "TST error (min, model − PSG)"),
           ("deep_share_pct", "Deep share of predicted sleep (%)")]
LOCKED_FINAL = "lr-locked, final model (model_outputs)"
RELEASE_ENS = "Release F2 ×3, 3-seed ensemble"
RELEASE_S0 = "Release F2, seed 0"
LOCKED_NAME = "lr-locked, epoch {}"
DROP_NAME = "LR-drop, epoch 160"
LOCKED_EPOCHS = [35, 59, 80, 160]


def model_paths() -> tuple[dict[str, list[Path]], list[str]]:
    """Model name -> checkpoint paths, and a note per requested checkpoint that does not exist."""
    notes, paths = [], {}
    for e in LOCKED_EPOCHS:
        p = LOCKED / f"diagnostics/epoch_{e}_seed_0.pisces"
        if p.exists():
            paths[LOCKED_NAME.format(e)] = [p]
        else:
            notes.append(f"lr-locked epoch {e}: no checkpoint ({p.name} not in {p.parent}); skipped")
    paths[LOCKED_FINAL] = [LOCKED / "model_outputs/model_final_seed_0.pisces"]
    paths[DROP_NAME] = [DROP / "diagnostics/epoch_160_seed_0.pisces"]
    final = [g.F2_FINAL / f"model_outputs/model_final_seed_{s}.pisces" for s in g.SEEDS]
    paths[RELEASE_S0] = final[:1]
    paths[RELEASE_ENS] = final
    return paths, notes


def cached_night(logger, stack, ctx, divisor, y, X, n, is_gap):
    logits = g.model_logits(stack, ctx.context_epochs, stack.config.frames_per_epoch, X, n, ctx.epoch_multiple)
    mean = np.exp(logits)
    stage = mean[:, :4] / mean[:, :4].sum(1, keepdims=True)
    out = {}
    for mode, calls in (("div", g.hybrid_calls(stage, stage, divisor)), ("argmax", stage.argmax(1))):
        calls = np.where(is_gap, 4, calls)
        out[mode] = (calls, g.subject_metrics(logger, y, mean, calls))
    calls, m = out["div"]
    scored = (y >= 0) & (y < 4)
    p_sleep = scored & (calls >= 1) & (calls <= 3)
    return dict(kappa_sleep_wake=m["kappa_sleep_wake"], kappa_4stage=m["kappa_4stage"],
                kappa_4stage_argmax=out["argmax"][1]["kappa_4stage"], wake_specificity_pct=100 * m["wake_specificity"],
                sleep_sensitivity_pct=100 * m["sleep_sensitivity"], tst_error_min=m["tst_error_minutes"],
                deep_share_pct=100 * float(np.mean(calls[p_sleep] == 2)) if p_sleep.any() else np.nan)


def score_cached(logger, stacks, ctx, divisor) -> pd.DataFrame:
    from arcasleep.data.batching import pool_frames_to_epochs, split_gap_row
    from arcasleep.data.bundle import CacheBundle
    rows = []
    for cohort in ("SleepAccel-Clinical", "AAUWSS"):
        for sid, path in g.cache_nights(g.CACHE_PREFIX[cohort]).items():
            b = CacheBundle.from_npz(path)
            y = b.y_for_n_classes(4)[0]
            n = len(y)
            pooled = pool_frames_to_epochs(b.X[0][None], np.zeros((1, n)))[0]
            is_gap = split_gap_row(pooled)[1] >= 1.0
            for name, stack in stacks.items():
                rows.append(dict(model=name, dataset=cohort, night=sid,
                                 **cached_night(logger, stack, ctx, divisor, y, b.X[0], n, is_gap)))
        print("scored", cohort, flush=True)
    return pd.DataFrame(rows)


def score_eesm(stacks, divisor) -> pd.DataFrame:
    from arcasleep.data.batching import pool_frames_to_epochs, split_gap_row
    from arcasleep.predict import Predictor
    from score_releases import EPOCH_S, night_metrics, read_accel, read_pairs
    feat = Predictor(g.PED_BUNDLE)
    manifest = json.loads((g.EESM_RAW / "conversion_manifest.json").read_text())
    rows = []
    for sid in sorted(m["id"] for m in manifest if m["status"] == "converted"):
        psg = read_pairs(g.EESM_RAW / "cleaned_psg" / f"{sid}_cleaned_psg.out")
        times, y = psg[:, 0], psg[:, 1].astype(int)
        scored = y >= 0
        accel = read_accel(g.EESM_RAW / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out")
        frames, origin = feat.features(accel, origin=float(times[0]))
        n = frames.shape[0] // feat.frames_per_epoch
        pooled = pool_frames_to_epochs(frames[None], np.zeros((1, n)))[0]
        is_gap = split_gap_row(pooled)[1] >= 1.0
        starts = origin + EPOCH_S * np.arange(n)
        pos = {int(round(t / EPOCH_S)): i for i, t in enumerate(times)}
        index = [(pos.get(int(round(s / EPOCH_S))), k) for k, s in enumerate(starts)]

        def night(calls):
            calls = np.where(is_gap, 4, calls)
            pred = np.full(len(times), 4)
            for i, k in index:
                if i is not None:
                    pred[i] = calls[k]
            return night_metrics(y[scored], pred[scored])

        for name, stack in stacks.items():
            p = np.exp(g.model_logits(stack, feat.context_epochs, stack.config.frames_per_epoch, frames, n))
            m, a = night(g.hybrid_calls(p, p, divisor)), night(p.argmax(1))
            rows.append(dict(model=name, dataset="EESM19", night=sid, kappa_sleep_wake=m["kappa_sleep_wake"],
                             kappa_4stage=m["kappa_4stage"], kappa_4stage_argmax=a["kappa_4stage"],
                             wake_specificity_pct=100 * m["wake_specificity"],
                             sleep_sensitivity_pct=100 * m["sleep_sensitivity"], tst_error_min=m["tst_error_min"],
                             deep_share_pct=m["deep_pct_pred"]))
        print("scored", sid, flush=True)
    return pd.DataFrame(rows)


def paired_tests(nights: pd.DataFrame, pairs) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        for cohort in COHORTS:
            for metric, _ in METRICS:
                ta = nights[(nights.model == a) & (nights.dataset == cohort)].set_index("night")[metric]
                tb = nights[(nights.model == b) & (nights.dataset == cohort)].set_index("night")[metric]
                d = (ta - tb.loc[ta.index]).dropna().to_numpy(float)
                p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
                rows.append(dict(a=a, b=b, dataset=cohort, metric=metric, n=len(d), mean_diff=d.mean(),
                                 median_diff=float(np.median(d)), wilcoxon_p=p))
    return pd.DataFrame(rows)


def fmt(v, metric):
    s = f"{v:.3f}" if metric.startswith("kappa") else f"{v:.1f}"
    return s.replace("-", "−")


def tables(nights: pd.DataFrame, paired: pd.DataFrame, models: list[str]) -> str:
    out = ["# Fit-capacity checkpoints on the external adult cohorts, 2026-10-23", "",
           "Mean over nights. Every model is one network (seed 0) except the release ensemble; the lr-locked final model is the checkpoint in `model_outputs/`, not a scored epoch; all use the release "
           "bundle's stage divisor, and the argmax column decodes the 4 stages without it. TST error is model − PSG.", ""]
    for cohort in COHORTS:
        t = nights[nights.dataset == cohort]
        out += [f"## {cohort} (n = {t.night.nunique()} nights)", "",
                "| Model | " + " | ".join(lab for _, lab in METRICS) + " |", "|:---|" + "---:|" * len(METRICS)]
        for m in models:
            mm = t[t.model == m]
            out.append(f"| {m} | " + " | ".join(fmt(mm[k].mean(), k) for k, _ in METRICS) + " |")
        out.append("")
    out += ["## Paired Wilcoxon over nights", "", "Mean difference (a − b) and p; p ≤ 0.05 in bold.", ""]
    for (a, b), t in paired.groupby(["a", "b"], sort=False):
        out += [f"### {a} − {b}", "", "| Cohort | " + " | ".join(lab for _, lab in METRICS) + " |",
                "|:---|" + "---:|" * len(METRICS)]
        for cohort in COHORTS:
            cells = []
            for k, _ in METRICS:
                r = t[(t.dataset == cohort) & (t.metric == k)].iloc[0]
                p = "p < 0.001" if r.wilcoxon_p < 0.001 else f"p = {r.wilcoxon_p:.3f}"
                s = f"{fmt(r.mean_diff, k)} ({p})"
                cells.append(f"**{s}**" if r.wilcoxon_p <= 0.05 else s)
            out.append(f"| {cohort} (n = {int(r.n)}) | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def history_check() -> list[str]:
    h = json.loads((LOCKED / "diagnostics/history_seed_0.json").read_text())
    k = np.asarray(h["eval_kappa"])
    best = int(k.argmax()) + 1
    lines = [f"lr-locked history: {len(k)} epochs, eval κ peaks at {k.max():.4f} at epoch {best} (index {best - 1} + 1)"]
    for e in LOCKED_EPOCHS:
        lines.append(f"  epoch {e}: eval κ {k[e - 1]:.4f}" + ("" if (LOCKED / f'diagnostics/epoch_{e}_seed_0.pisces').exists()
                                                            else " (no checkpoint)"))
    return lines


def weights_equal(a, b) -> float:
    import jax
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max())
               for x, y in zip(jax.tree_util.tree_leaves(a.params), jax.tree_util.tree_leaves(b.params)))


def main(out: Path) -> None:
    from arcasleep.models import get_family
    from arcasleep_bidoze.adult import AdultPredictor, StackedModel
    paths, notes = model_paths()
    fam = get_family("bidoze")
    loaded = {p: fam.load_checkpoint(p) for ps in paths.values() for p in ps}
    stacks = {name: StackedModel([loaded[p] for p in ps]) for name, ps in paths.items()}
    spec = json.loads((g.F2_BUNDLE / "composite.json").read_text())
    divisor = np.array([1.0] + [float(spec["stage_divisor"][k]) for k in ("light", "deep", "rem")])
    ctx = AdultPredictor(g.F2_BUNDLE).ensemble
    logger = g.make_logger()

    nights = pd.concat([score_cached(logger, stacks, ctx, divisor), score_eesm(stacks, divisor)])
    nights["night"] = nights.night.astype(str)
    out.mkdir(parents=True, exist_ok=True)
    nights.to_csv(out / "per_night.csv", index=False)
    models = list(paths)
    pairs = [(LOCKED_NAME.format(160), RELEASE_S0), (LOCKED_NAME.format(160), LOCKED_NAME.format(35))]
    paired = paired_tests(nights, pairs)
    paired.to_csv(out / "paired.csv", index=False)

    print("\n== checks ==")
    grid = pd.read_csv(g.R / "model-grid-2026-10-23/per_night.csv", dtype={"night": str})
    grid = grid[grid.model == g.F2X3]
    for cohort in COHORTS:
        a = nights[(nights.model == RELEASE_ENS) & (nights.dataset == cohort)].set_index("night")
        b = grid[grid.dataset == cohort].set_index("night")
        common = a.index.intersection(b.index)
        diffs = {k: float((a.loc[common, k] - b.loc[common, k]).abs().max())
                 for k in ("kappa_sleep_wake", "kappa_4stage", "wake_specificity_pct", "tst_error_min")}
        print(f"release ensemble vs grid, {cohort}: {len(common)}/{len(a)} nights in common (grid has {len(b)}); max diffs {diffs}")
    for line in history_check():
        print(line)
    for note in notes:
        print(note)
    final = fam.load_checkpoint(LOCKED / "model_outputs/model_final_seed_0.pisces")
    print("lr-locked model_final vs epoch 160 max |weight diff|:", weights_equal(final, loaded[paths[LOCKED_NAME.format(160)][0]]))
    final = fam.load_checkpoint(DROP / "model_outputs/model_final_seed_0.pisces")
    print("LR-drop model_final vs epoch 160 max |weight diff|:", weights_equal(final, loaded[paths[DROP_NAME][0]]))
    (out / "tables.md").write_text(tables(nights, paired, models) + "\n")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]))

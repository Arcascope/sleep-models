"""Score the strand-fusion fits and BiDoze-G x3 on the Weaver children each draw did not train on.

Usage (repository root, after ``export PYTHONPATH=core/src:bidoze/src``):
    python bidoze/experiments/strand_fusion/analysis/weaver_children.py bidoze/results/strand-fusion-2026-10-21 predict
    python bidoze/experiments/strand_fusion/analysis/weaver_children.py bidoze/results/strand-fusion-2026-10-21 score

Held-out children of draw d: the children of the release's revised-Weaver nights (``matched``, 202
recordings, 201 children) with no recording in the draw's fit or eval list (``split.json``). ``predict``
writes ``<run>/weaver-children/predictions/<unit>/draw<d>/<subject>.npz`` and ``leakage.json``;
``score`` writes per_recording.csv, per_draw.csv, paired.csv and tables.md.

Decodings: ``argmax`` is the argmax over wake/light/deep/REM (no stage divisor); ``divisor`` is the
adult hybrid decoding with the draw's stage divisor (secondary). Epochs without a stage label and gap
epochs are excluded as in ``card_metrics.subject_metrics``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path[:0] = [str(HERE.parent), str(EXP / "adult_devices")]

import data as D  # noqa: E402
from arcasleep.data.cache import discover_cached_subjects, participant_id  # noqa: E402

MATCHED = D.RESULTS / "bidoze-adult-devices-0/setup/arms.json"
FEATURES = Path("/home/eric/Engineering/Work/autopisces/sleep-models/feature_cache")
DRAWS = range(5)
FIT_UNITS = {"F0": "final3", "F1": "final3_mixed", "F2s0": "final3_fuse", "F2s1": "final3_fuse",
             "F2s2": "final3_fuse", "F3": "final3_mixed_fuse", "F4": "final3_large"}
G_SEEDS = range(6)
STG = ["wake", "light", "deep", "rem"]
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity",
           *[f"recall_{s}" for s in STG], "tst_bias_min"]
DECODINGS = {"argmax": "argmax", "divisor": "draw divisor (secondary)"}


def fit_dir(run: Path, unit: str, draw: int) -> Path:
    arm = FIT_UNITS[unit]
    seed = int(unit[3]) if unit.startswith("F2s") else 0
    return run / arm / f"draw{draw}_seed{seed}"


def heldout(run: Path):
    """``{draw: (held-out ids, fit/eval ids)}`` and the leakage record, checked against draws.json."""
    matched = json.loads(MATCHED.read_text())["matched"]
    assert len(matched) == 202 and len({participant_id(s) for s in matched}) == 201
    record, out = {}, {}
    for d in DRAWS:
        sets = []
        for unit in FIT_UNITS:
            s = json.loads((fit_dir(run, unit, d) / "split.json").read_text())
            sets.append(sorted(r["subject_id"] for k in ("fit", "validation") for r in s[k] if r["dataset"] == "weaverDraw"))
        assert all(x == sets[0] for x in sets), "fits of a draw differ in their Weaver recordings"
        trained = sets[0]
        for name in (D.SOURCE_RUN, D.G3_RUN):
            listed = json.loads((name / "setup/draws.json").read_text())[f"draw{d}"]["weaverDraw"]
            assert sorted(listed) == trained, f"split.json differs from {name.name} draws.json"
        assert set(trained) <= set(matched)
        out_ids = sorted(set(matched) - set(trained))
        leak = set(out_ids) & set(trained)
        pleak = {participant_id(s) for s in out_ids} & {participant_id(s) for s in trained}
        assert not leak and not pleak
        record[f"draw{d}"] = dict(trained_recordings=len(trained), trained_children=len({participant_id(s) for s in trained}),
                                  heldout_recordings=len(out_ids), heldout_children=len({participant_id(s) for s in out_ids}),
                                  recording_overlap=len(leak), child_overlap=len(pleak))
        out[d] = out_ids
    return out, record


def g_path(run: Path, seed: int, draw: int) -> Path:
    import g3_predictions as G
    return G.model_path(run, seed, draw)


def predict(run: Path):
    import jax
    import jax.numpy as jnp
    from arcasleep_bidoze.model import BiDozeNetModel
    from fusion_model import StrandFusionModel
    ids, record = heldout(run)
    out = run / "weaver-children"
    (out / "leakage.json").write_text(json.dumps(record, indent=2))
    for d in DRAWS:
        files = {s: FEATURES / "12Hz_log_PSD_aligned" / f"weaver_revised_subject_{s}_features.npz" for s in ids[d]}
        missing = [s for s, p in files.items() if not p.exists()]
        assert not missing, f"missing cache files {missing[:3]}"
        batches = {s: D.recording_batch(p) for s, p in files.items()}
        units = {}
        for unit in FIT_UNITS:
            m = StrandFusionModel.load_research(fit_dir(run, unit, d))
            f = jax.jit(lambda p, x, m=m: m.module.apply({"params": p}, x, jnp.ones(x.shape[:2]), return_outputs=True))
            units[unit] = (m, f)
        for seed in G_SEEDS:
            m = BiDozeNetModel.loadz(g_path(run, seed, d))
            units[f"G{seed}"] = (m, jax.jit(lambda x, m=m: jax.nn.softmax(
                m.module.apply({"params": m.params}, x, jnp.ones(x.shape[:2]), train=False), axis=-1)))
        for unit, (m, f) in units.items():
            target = out / "predictions" / unit / f"draw{d}"
            target.mkdir(parents=True, exist_ok=True)
            for s, (x, labels) in batches.items():
                path = target / f"{s}.npz"
                if path.exists():
                    continue
                n = len(labels)
                if unit.startswith("G"):
                    D.save_recording(path, labels, np.asarray(f(x))[0, :n][None])
                else:
                    o = f(m.params, x)
                    fusion = np.exp(np.asarray(o["log_head"]))[0, :n] if m.spec.fusion else None
                    D.save_recording(path, labels, np.asarray(o["strand_probabilities"])[:, 0, :n], fusion)
            print("done", unit, d, flush=True)
        del units


def load(run: Path, unit: str, d: int, s: str):
    z = np.load(run / "weaver-children/predictions" / unit / f"draw{d}" / f"{s}.npz")
    return z["labels"], z["strand_probabilities"], z["fusion_probabilities"] if "fusion_probabilities" in z.files else None


def probabilities(run: Path, name: str, d: int, s: str):
    """``(labels, (n, 5) probabilities)`` of one model on one recording."""
    def one(unit, head):
        y, strands, fusion = load(run, unit, d, s)
        return y, (fusion if head == "fusion" else strands.mean(0))
    spec = {"F0": [("F0", "strand")], "F1": [("F1", "strand")], "F3": [("F3", "fusion")],
            "F3 strand mean": [("F3", "strand")], "F4": [("F4", "strand")],
            "F2 x3": [(f"F2s{k}", "fusion") for k in range(3)],
            "F2 x3 strand mean": [(f"F2s{k}", "strand") for k in range(3)],
            "G x3 (seeds 0-2)": [(f"G{k}", "strand") for k in range(3)],
            "G x3 (seeds 3-5)": [(f"G{k}", "strand") for k in range(3, 6)],
            **{f"F2 seed {k}": [(f"F2s{k}", "fusion")] for k in range(3)}}[name]
    got = [one(*x) for x in spec]
    return got[0][0], np.mean([g[1] for g in got], axis=0)


MODELS = ["F2 seed 0", "F2 seed 1", "F2 seed 2", "F2 x3", "F2 x3 strand mean", "G x3 (seeds 0-2)", "G x3 (seeds 3-5)",
          "F0", "F1", "F3", "F3 strand mean", "F4"]
REFS = ["G x3 (seeds 0-2)", "G x3 (seeds 3-5)"]


def score(run: Path):
    from arcasleep.training.config import load_metrics_config
    from arcasleep_bidoze.adult import hybrid_calls
    from card_metrics import subject_metrics
    from pisces_lite.metrics.logger import MetricsLogger
    import arcasleep.evaluation  # noqa: F401  (registers the metrics)
    logger = MetricsLogger(load_metrics_config(EXP.parent / "config/metrics.json")[0])
    ids, _ = heldout(run)
    rows = []
    for d in DRAWS:
        divisor = np.load(fit_dir(run, "F0", d) / "stage_divisor.npy")
        for name in MODELS:
            for s in ids[d]:
                y, p = probabilities(run, name, d, s)
                for dec in DECODINGS:
                    calls = p[:, :4].argmax(1) if dec == "argmax" else hybrid_calls(p, p, divisor)
                    m = subject_metrics(logger, y, p, calls)
                    scored = (y >= 0) & (y < 4)
                    row = dict(model=name, decoding=dec, draw=d, subject_id=s, child=participant_id(s),
                               **{k: float(m[k]) for k in METRICS[:4]})
                    for i, st in enumerate(STG):
                        on = scored & (y == i)
                        row[f"recall_{st}"] = float(np.mean(calls[on] == i)) if on.any() else np.nan
                    row["tst_bias_min"] = 0.5 * float(np.sum((calls[scored] > 0) & (calls[scored] < 4)) - np.sum(y[scored] > 0))
                    rows.append(row)
        print("scored draw", d, flush=True)
    out = run / "weaver-children"
    rec = pd.DataFrame(rows)
    rec.to_csv(out / "per_recording.csv", index=False)
    per_draw = rec.groupby(["model", "decoding", "draw"], sort=False).agg(
        n_recordings=("subject_id", "count"), **{m: (m, "mean") for m in METRICS}).reset_index()
    per_draw.to_csv(out / "per_draw.csv", index=False)
    paired = []
    pairs = [(m, r) for r in REFS for m in MODELS if m != r] + [(REFS[1], REFS[0])]
    pairs = [(a, b) for a, b in pairs if not (a == REFS[1] and b == REFS[0]) or True]
    seen = set()
    for dec in DECODINGS:
        t = per_draw[per_draw.decoding == dec].set_index(["model", "draw"])
        for a, b in pairs:
            if (a, b, dec) in seen:
                continue
            seen.add((a, b, dec))
            for m in METRICS:
                diff = (t.loc[a][m] - t.loc[b][m]).to_numpy()
                n = len(diff)
                half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
                paired.append(dict(model=a, reference=b, decoding=dec, metric=m, n_draws=n, mean_diff=diff.mean(),
                                   sd=diff.std(ddof=1), ci_low=diff.mean() - half, ci_high=diff.mean() + half,
                                   draws_higher=int((diff > 0).sum())))
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)
    write_tables(out, per_draw, paired, json.loads((out / "leakage.json").read_text()))


def fmt(x, digits=3):
    return f"{x:.{digits}f}".replace("-", "−")


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + ":---|" * len(header),
                      *["| " + " | ".join(r) + " |" for r in rows]])


def write_tables(out: Path, per_draw, paired, leak):
    short = dict(kappa_4stage="κ 4-stage", kappa_sleep_wake="κ sleep/wake", sleep_sensitivity="sleep sens",
                 wake_specificity="wake spec", recall_wake="wake recall", recall_light="light recall",
                 recall_deep="deep recall", recall_rem="REM recall", tst_bias_min="TST bias (min)")
    lines = ["# Weaver children not used to train each draw", "",
             "## Held-out counts and leakage check", "",
             md(["draw", "trained recordings", "trained children", "held-out recordings", "held-out children",
                 "recordings in both", "children in both"],
                [[k[4:], *[str(v[x]) for x in ("trained_recordings", "trained_children", "heldout_recordings",
                                              "heldout_children", "recording_overlap", "child_overlap")]]
                 for k, v in leak.items()]), ""]
    for dec, label in DECODINGS.items():
        t = per_draw[per_draw.decoding == dec].groupby("model", sort=False)[METRICS].mean()
        lines += [f"## Mean over the 5 draws, decoding: {label}", "",
                  "Each draw's value is the mean over its held-out recordings.", "",
                  md(["model", *short.values()], [[m, *[fmt(t.loc[m][k], 1 if k == "tst_bias_min" else 3) for k in short]] for m in MODELS]), ""]
    for dec, label in DECODINGS.items():
        for ref in REFS + [None]:
            sub = paired[(paired.decoding == dec)]
            sub = sub[(sub.reference == ref)] if ref else sub[(sub.model == REFS[1]) & (sub.reference == REFS[0])]
            title = f"minus {ref}" if ref else f"{REFS[1]} minus {REFS[0]}"
            lines += [f"## Paired differences over draws ({title}), decoding: {label}", "",
                      "Mean ± SD over 5 draws [paired 95% t-interval] (draws higher of 5).", ""]
            rows = []
            for a in sub.model.unique():
                r = [a]
                for k in short:
                    x = sub[(sub.model == a) & (sub.metric == k)].iloc[0]
                    dg = 1 if k == "tst_bias_min" else 3
                    r.append(f"{fmt(x.mean_diff, dg)} ± {fmt(x.sd, dg)} [{fmt(x.ci_low, dg)}, {fmt(x.ci_high, dg)}] ({x.draws_higher})")
                rows.append(r)
            lines += [md(["model", *short.values()], rows), ""]
    (out / "tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run = Path(sys.argv[1])
    {"predict": predict, "score": score}[sys.argv[2]](run)

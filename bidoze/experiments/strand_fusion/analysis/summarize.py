"""Tables of the strand-fusion arms against BiDoze-G x3 from a run directory.

Usage (repository root): JAX_PLATFORMS=cpu PYTHONPATH=core/src:bidoze/src python \
    bidoze/experiments/strand_fusion/analysis/summarize.py bidoze/results/strand-fusion-2026-10-21

Reads ``<run>/<arm>/draw<d>_seed0/`` (run.py) and ``<run>/g3/draw<d>/`` (g3_predictions.py); writes
``<run>/analysis/``: per_draw.csv, summary.csv (means and paired differences against G x3 on the
same draw), fits.csv (weights, GPU time), tables.md.

Decodings: ``orig`` is the draw's stage divisor; ``prior`` refits the deep and REM divisors on the
adult eval-share outputs of the method being scored so the called light/deep/REM shares match
BIDsleep's (``strand_gated_fusion/fuse.py``). Methods: ``output`` is the arm's output (the mean of
strand probabilities, or the fusion head for fused arms); ``strand_mean`` is the mean of the strand
heads of a fused arm. G x3 is the mean of its three seeds' probabilities.

Units: ``g3`` (G seeds 0-2), ``g3b`` (G seeds 3-5), the arms F0-F4 (seed 0), ``final3_fuse_s1`` and
``final3_fuse_s2`` (F2 seeds 1, 2) and ``final3_fuse_x3`` (F2 x3: the mean of the probabilities of F2
seeds 0-2 on each draw, 3 x the weights of one). Replication tables (``replication.csv``) pair each
F2 seed with G x3 (seeds 0-2) on the same draw, pooling draws x seeds, and pair F2 x3 with G seeds 3-5.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "strand_gated_fusion"))
sys.path.insert(0, str(HERE.parent))

import fuse  # noqa: E402  (refit_divisors and the scoring it uses)
from fusion_model import ARMS  # noqa: E402

scoring = fuse.scoring
COH = {"testSleepAccel": "SleepAccel", "testClinical": "Clinical"}
STG = ["wake", "light", "deep", "rem"]
METRICS = ["kappa_4stage", "kappa_sleep_wake", "sleep_sensitivity", "wake_specificity"] + [f"recall_{s}" for s in STG]
SHORT = dict(kappa_4stage="κ4", kappa_sleep_wake="κ sw", sleep_sensitivity="sens", wake_specificity="spec",
             recall_wake="wake", recall_light="light", recall_deep="deep", recall_rem="REM")
DECODINGS = ["orig", "prior"]
REFERENCE = "g3"
REPLICATE = "g3b"
F2 = "final3_fuse"
F2_SEEDS = {F2: 0, "final3_fuse_s1": 1, "final3_fuse_s2": 2}
F2_X3 = "final3_fuse_x3"
UNITS = [REFERENCE, REPLICATE, *ARMS, "final3_fuse_s1", "final3_fuse_s2", F2_X3]
LABEL = {REFERENCE: "G x3 (seeds 0-2)", REPLICATE: "G x3 (seeds 3-5)",
         **{a: f"{v[0]} {a}" for a, v in ARMS.items()},
         "final3_fuse_s1": "F2 seed 1", "final3_fuse_s2": "F2 seed 2", F2_X3: "F2 x3"}
LABEL[F2] = "F2 final3_fuse (seed 0)"


def unit_dirs(run: Path, unit: str, d: int) -> list[Path]:
    if unit in (REFERENCE, REPLICATE):
        return [run / unit / f"draw{d}"]
    if unit == F2_X3:
        return [run / F2 / f"draw{d}_seed{k}" for k in F2_SEEDS.values()]
    if unit in F2_SEEDS:
        return [run / F2 / f"draw{d}_seed{F2_SEEDS[unit]}"]
    return [run / unit / f"draw{d}_seed0"]


def load(directory: Path, role: str):
    """``[(dataset, subject, labels, strand probabilities (S, n, 5), fusion probabilities or None)]``."""
    rows = []
    for f in sorted((directory / "predictions").glob(f"{role}_*.npz")):
        ds, sid = f.stem.removeprefix(f"{role}_").split("_", 1)
        z = np.load(f)
        rows.append((ds, sid, z["labels"], z["strand_probabilities"],
                     z["fusion_probabilities"] if "fusion_probabilities" in z.files else None))
    return rows


def methods_of(rows):
    out = {"output": [r[4] if r[4] is not None else r[3].mean(0) for r in rows]}
    if rows and rows[0][4] is not None:
        out["strand_mean"] = [r[3].mean(0) for r in rows]
    return out


def pi_adult():
    found = fuse.discover_cached_subjects(["bidsleep"], scoring.LABELS / scoring.CACHE_NAME)["bidsleep"]
    share = np.asarray(scoring.night_counts([x.path for x in found]), float).sum(0)
    share = share / share.sum()
    return np.concatenate([[0.0], share[1:4]]) if len(share) >= 4 else share


def mean_over_dirs(per_dir):
    """Per method, the mean over directories of each recording's probabilities."""
    return {m: [np.mean(ps, axis=0) for ps in zip(*[x[m] for x in per_dir])] for m in per_dir[0]}


def score_fit(directories: list, name: str, draw: int, logger, pi):
    divisor = np.load(directories[0] / "stage_divisor.npy")
    evs = [load(x, "eval") for x in directories]
    tes = [load(x, "test") for x in directories]
    for group in (evs, tes):
        assert all([r[:2] for r in x] == [r[:2] for r in group[0]] for x in group), "recordings differ between fits"
    ev, te = evs[0], tes[0]
    ev_m, te_m = mean_over_dirs([methods_of(x) for x in evs]), mean_over_dirs([methods_of(x) for x in tes])
    out = []
    for method in te_m:
        refit = fuse.refit_divisors(ev_m[method], [r[2] for r in ev], divisor, pi)["prior"]
        for dec, div in (("orig", divisor), ("prior", refit)):
            for (ds, sid, y, _, _), p in zip(te, te_m[method]):
                calls = scoring.hybrid_calls(p, p, div)
                m = scoring.subject_metrics(logger, y, p, calls)
                rec = {}
                for st in range(4):
                    on = y == st
                    rec[f"hits{st}"], rec[f"total{st}"] = int((calls[on] == st).sum()), int(on.sum())
                out.append(dict(arm=name, draw=draw, method=method, decoding=dec, dataset=ds, subject_id=sid,
                                **{k: float(m[k]) for k in METRICS[:4]}, **rec))
    return out


def md(frame):
    def cell(x):
        if isinstance(x, float):
            return f"{x:.3f}".replace("-", "−")
        return str(x)
    head = "| " + " | ".join(frame.columns) + " |\n|" + ":---|" * len(frame.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(x) for x in r) + " |" for r in frame.to_numpy())


def compare(agg, pairs, metrics=METRICS):
    """Paired differences per (method, decoding, cohort, metric): ``pairs`` is ``[(unit, reference unit)]``;
    each pair contributes one difference per draw both have. Returns mean, SD and the number higher."""
    rows = []
    for method in ("output", "strand_mean"):
        for dec in DECODINGS:
            for coh in COH.values():
                for m in metrics:
                    diffs = []
                    for unit, ref in pairs:
                        x = agg[(agg.arm == unit) & (agg.method == method) & (agg.decoding == dec) & (agg.cohort == coh)]
                        # the reference is the unit's output (G x3 has no other method)
                        r = agg[(agg.arm == ref) & (agg.method == "output") & (agg.decoding == dec) & (agg.cohort == coh)]
                        d = (x.set_index("draw")[m] - r.set_index("draw")[m]).dropna()
                        diffs.extend(d.tolist())
                    if diffs:
                        rows.append(dict(method=method, decoding=dec, cohort=coh, metric=m, n=len(diffs),
                                         mean_diff=np.mean(diffs), sd=np.std(diffs, ddof=1) if len(diffs) > 1 else np.nan,
                                         n_higher=int(sum(x > 0 for x in diffs))))
    return pd.DataFrame(rows)


def main():
    run = Path(sys.argv[1])
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    logger = scoring.MetricsLogger(scoring.load_metrics_config(fuse.ROOT / "bidoze/config/metrics.json")[0])
    pi = pi_adult()
    rows, info = [], []
    for name in UNITS:
        for d in range(5):
            dirs = unit_dirs(run, name, d)
            if not all((x / "complete.json").exists() for x in dirs):
                continue
            cache = out / "cache" / f"{name}_draw{d}.csv"
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                rows.append(pd.read_csv(cache))
            else:
                df = pd.DataFrame(score_fit(dirs, name, d, logger, pi))
                df.to_csv(cache, index=False)
                rows.append(df)
            done = [json.loads((x / "complete.json").read_text()) for x in dirs]
            seconds = [x.get("train_seconds") for x in done]
            info.append(dict(arm=name, draw=d, parameters=sum(x["parameters"] for x in done),
                             train_seconds=None if None in seconds else sum(seconds),
                             time_capped=any(x.get("time_capped", False) for x in done)))
    df = pd.concat(rows)
    df["cohort"] = df.dataset.map(COH)
    keys = ["arm", "draw", "method", "decoding", "cohort"]
    agg = df.groupby(keys).agg(**{m: (m, "mean") for m in METRICS[:4]},
                               **{f"hits{i}": (f"hits{i}", "sum") for i in range(4)},
                               **{f"total{i}": (f"total{i}", "sum") for i in range(4)}).reset_index()
    for i, s in enumerate(STG):
        agg[f"recall_{s}"] = agg[f"hits{i}"] / agg[f"total{i}"]
    agg = agg.drop(columns=[f"hits{i}" for i in range(4)] + [f"total{i}" for i in range(4)])
    agg.to_csv(out / "per_draw.csv", index=False)

    ref = agg[agg.arm == REFERENCE].set_index(["draw", "decoding", "cohort"])[METRICS]
    summary = []
    for (arm, method, dec, coh), g in agg.groupby(["arm", "method", "decoding", "cohort"]):
        g = g.set_index("draw")
        r = dict(arm=arm, method=method, decoding=dec, cohort=coh, n_draws=len(g))
        for m in METRICS:
            r[m] = g[m].mean()
            if arm != REFERENCE:
                x = ref.xs((dec, coh), level=(1, 2))[m].reindex(g.index)
                d = (g[m] - x).dropna()
                r[f"{m}_minus_g3"], r[f"{m}_minus_g3_sd"] = d.mean(), d.std()
                r["n_paired"] = len(d)
        summary.append(r)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    comparisons = {
        "F2 seeds 0-2 vs G x3 (seeds 0-2), draws x seeds": [(u, REFERENCE) for u in F2_SEEDS],
        "F2 x3 vs G x3 (seeds 3-5), draws": [(F2_X3, REPLICATE)],
        "F2 x3 vs G x3 (seeds 0-2), draws": [(F2_X3, REFERENCE)],
        "G x3 (seeds 3-5) vs G x3 (seeds 0-2), draws": [(REPLICATE, REFERENCE)],
    }
    replication = pd.concat([compare(agg, p).assign(comparison=c) for c, p in comparisons.items()])
    replication.to_csv(out / "replication.csv", index=False)

    info = pd.DataFrame(info)
    info.to_csv(out / "fits.csv", index=False)
    hours = info.groupby("arm").agg(weights=("parameters", "first"), n_draws=("draw", "count"),
                                    gpu_minutes_per_draw=("train_seconds", lambda s: s.mean() / 60),
                                    gpu_minutes_total=("train_seconds", lambda s: s.sum() / 60)).reset_index()
    hours["arm"] = hours.arm.map(LABEL)

    lines = ["# Strand fusion against BiDoze-G x3", "",
             "Cells: mean over draws; for rows other than G x3 (seeds 0-2), the paired difference to G x3 "
             "(seeds 0-2) on the same draw as mean ± SD over draws (negative numbers use −). n = draws. "
             "G fits of seeds 3-5 on draws 0-1 and G seeds 0-2 have no recorded GPU time here.", "",
             "## Weights and training GPU time", "", md(hours), ""]
    for dec in DECODINGS:
        for coh in COH.values():
            lines += [f"## {coh}, decoding `{dec}`", ""]
            table = []
            s = summary[(summary.decoding == dec) & (summary.cohort == coh)]
            for arm in UNITS:
                for method in ("output", "strand_mean"):
                    x = s[(s.arm == arm) & (s.method == method)]
                    if x.empty:
                        continue
                    x = x.iloc[0]
                    row = {"arm": LABEL[arm] + ("" if method == "output" else " (strand mean)"), "n": int(x.n_draws)}
                    for m in METRICS:
                        text = f"{x[m]:.3f}"
                        if arm != REFERENCE:
                            text += f" ({x[m + '_minus_g3']:+.3f} ± {x[m + '_minus_g3_sd']:.3f})"
                        row[SHORT[m]] = text.replace("-", "−")
                    table.append(row)
            lines += [md(pd.DataFrame(table)), ""]
    lines += ["## Replication", "",
              "Mean difference ± SD over pairs, with the number of pairs where the first is higher (k/n). "
              "Method `output` is the fusion head (F2) or the mean (G); `strand_mean` is the mean of F2's strand heads.", ""]
    for dec in DECODINGS:
        for coh in COH.values():
            lines += [f"### {coh}, decoding `{dec}`", ""]
            table = []
            r = replication[(replication.decoding == dec) & (replication.cohort == coh)]
            for (comparison, method), g in r.groupby(["comparison", "method"], sort=False):
                row = {"comparison": comparison + ("" if method == "output" else " (F2 strand mean)"), "n": int(g.n.max())}
                for _, x in g.iterrows():
                    row[SHORT[x.metric]] = (f"{x.mean_diff:+.3f} ± {x.sd:.3f} ({int(x.n_higher)}/{int(x.n)})").replace("-", "−")
                table.append(row)
            lines += [md(pd.DataFrame(table)), ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Run the fixed-point reference on Weaver Garmin nights and compare it with the float features.

    python -m arcasleep_watchtree.port_run reference --out results/port-2026-10-27 [--limit N]
    python -m arcasleep_watchtree.port_run compare   --out results/port-2026-10-27 \\
        --features results/features-2026-10-27

``reference`` replays each night's 25 Hz stream (rounded to integer milli-g, one heart-rate
reading per second) through :class:`~arcasleep_watchtree.fixed.Engine` and saves its feature
integers of every epoch that reached its 45-min delay. ``compare`` measures agreement with
``features.py`` and the 4-stage kappa change on out-of-fold models.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from arcasleep_watchtree import cv
from arcasleep_watchtree import export as ex
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, load_night, subject_ids
from arcasleep_watchtree.constants import SAMPLES_PER_EPOCH
from arcasleep_watchtree.definition import CONFIG_DIR, TreeSpec
from arcasleep_watchtree.fixed import Engine
from arcasleep_watchtree.portspec import FEATURES, NA, SCALE

_STATE: dict = {}
COHORT = "weaver_garmin"


def _init() -> None:
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset

    _STATE["data_set"] = load_dataset(DATA_ROOT / COHORTS[COHORT].directory)


def stream_of(night) -> dict:
    """Integer inputs of a night: samples in mg, valid flags, one heart-rate bpm per second."""
    acc = np.rint(night.acc25 * 1000.0).astype(np.int64)
    n_sec = acc.shape[0] // 25
    hr = np.full(n_sec, -1, np.int64)
    t = np.asarray(night.hr_t, float)
    ok = np.isfinite(night.hr_bpm) & (t >= 0) & (t < n_sec)
    sec = t[ok].astype(int)
    hr[sec] = np.rint(night.hr_bpm[ok]).astype(np.int64)  # the last reading in a second wins
    return {"acc": acc, "valid": night.valid25, "hr": hr}


def run_stream(acc, valid, hr, model=None) -> Engine:
    eng = Engine(model)
    n_sec = acc.shape[0] // 25
    xs, ys, zs = acc[:, 0].tolist(), acc[:, 1].tolist(), acc[:, 2].tolist()
    hrl, vl = hr.tolist(), valid.tolist()
    for b in range(n_sec):
        s = b * 25
        eng.batch(xs[s : s + 25], ys[s : s + 25], zs[s : s + 25], 0, hrl[b] if hrl[b] >= 0 else None, vl[s : s + 25])
    return eng


def _one(args):
    subject, reported = args
    try:
        night = load_night(COHORTS[COHORT], subject, reported, _STATE["data_set"])
        s = stream_of(night)
        eng = run_stream(s["acc"], s["valid"], s["hr"])
    except Exception as exc:  # noqa: BLE001
        return reported, type(exc).__name__ + str(exc)[:80]
    epochs = np.array([r["epoch"] for r in eng.out], np.int32)
    feats = np.array([r["features"] for r in eng.out], np.int64)
    return reported, (epochs, feats, night.labels.size)


def reference(out: Path, limit: int | None, workers: int) -> None:
    ids, reported = subject_ids(COHORTS[COHORT])
    ids = ids[:limit] if limit else ids
    ctx = multiprocessing.get_context("spawn")
    result, failed = {}, {}
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init) as pool:
        for rep, payload in pool.map(_one, [(i, reported[i]) for i in ids]):
            if isinstance(payload, str):
                failed[rep] = payload
            else:
                result[rep] = payload
            print(len(result) + len(failed), "nights", flush=True)
    keys, arrays = [], {}
    for rep, (epochs, feats, n_epochs) in result.items():
        arrays[f"{rep}__epochs"] = epochs
        arrays[f"{rep}__features"] = feats.astype(np.int32)
        arrays[f"{rep}__n_epochs"] = np.array([n_epochs])
        keys.append(rep)
    np.savez_compressed(out / f"reference-{COHORT}.npz", **arrays)
    (out / f"reference-{COHORT}.json").write_text(json.dumps({"nights": keys, "failed": failed}, indent=1))
    print("done", len(keys), "failed", failed)


# ----------------------------------------------------------------------------------- compare


def compare(out: Path, features_dir: Path) -> None:
    ref = np.load(out / f"reference-{COHORT}.npz")
    nights = json.loads((out / f"reference-{COHORT}.json").read_text())["nights"]
    table = pd.read_parquet(features_dir / f"features-{COHORT}.parquet")
    table = table[table.subject.isin(nights)]
    model = json.loads((out / "model.json").read_text())
    scale = np.array(model["scale"], float)
    edges, real = [np.array(e) for e in model["edges"]], [np.array(e) for e in model["real_edges"]]
    folds = json.loads((out / "folds.json").read_text())

    rows = []
    for sub, g in table.groupby("subject"):
        g = g.sort_values("epoch")
        epochs = ref[f"{sub}__epochs"]
        feats = ref[f"{sub}__features"].astype(np.int64)
        x = g[FEATURES].to_numpy(np.float64)[epochs]
        keep = epochs >= 240
        rows.append(
            pd.DataFrame(
                {
                    "subject": sub, "epoch": epochs[keep], "y4": g.y4.to_numpy()[epochs[keep]],
                    **{f"ref_{n}": feats[keep, i] for i, n in enumerate(FEATURES)},
                    **{f"flt_{n}": x[keep, i] for i, n in enumerate(FEATURES)},
                }
            )
        )
    both = pd.concat(rows, ignore_index=True)
    ref_i = both[[f"ref_{n}" for n in FEATURES]].to_numpy()
    flt = both[[f"flt_{n}" for n in FEATURES]].to_numpy()
    ref_f = np.where(ref_i == NA, np.nan, ref_i / scale)

    per_feature = []
    for i, n in enumerate(FEATURES):
        a, b = flt[:, i], ref_f[:, i]
        both_ok = np.isfinite(a) & np.isfinite(b)
        sd = float(np.std(a[both_ok])) if both_ok.any() else np.nan
        err = np.abs(a[both_ok] - b[both_ok])
        corr = float(np.corrcoef(a[both_ok], b[both_ok])[0, 1]) if both_ok.sum() > 2 and sd > 0 else np.nan
        per_feature.append(
            {
                "feature": n,
                "unit": 1.0 / scale[i],
                "nan_mismatch_share": float(np.mean(np.isfinite(a) != np.isfinite(b))),
                "corr": corr,
                "max_abs_err": float(err.max()) if err.size else np.nan,
                "max_abs_err_in_sd": float(err.max() / sd) if err.size and sd > 0 else np.nan,
                "mean_abs_err_in_sd": float(err.mean() / sd) if err.size and sd > 0 else np.nan,
                "bin_mismatch_share": np.nan,
            }
        )
    # both sides use the same integer edges, so a differing bin is a difference of the features
    ints_float = np.where(np.isnan(flt), NA, np.rint(flt * scale))
    flt_bins = ex.bin_matrix(ints_float, edges)[:, : len(FEATURES)]
    ref_bins = ex.bin_matrix(ref_i, edges)[:, : len(FEATURES)]
    diff = flt_bins != ref_bins
    for i, row in enumerate(per_feature):
        row["bin_mismatch_share"] = float(diff[:, i].mean())
    pd.DataFrame(per_feature).to_csv(out / "feature_agreement.csv", index=False)
    summary = {
        "nights": len(nights),
        "epochs_compared": int(len(both)),
        "share_epochs_any_bin_differs": float(diff.any(axis=1).mean()),
        "mean_features_differing_per_epoch": float(diff.sum(axis=1).mean()),
        "share_epochs_two_or_more_bins_differ": float((diff.sum(axis=1) >= 2).mean()),
    }

    # kappa on the same epochs, out of fold
    from arcasleep_watchtree.cv import person_of

    both["person"] = [person_of(COHORT, s) for s in both.subject]
    both["fold"] = both.person.map(folds)
    scored = (both.y4 >= 0) & both.fold.notna()
    pred = {k: np.full(len(both), -1) for k in ("lightgbm_float", "export_float_features", "export_reference")}
    for k in range(5):
        sel = (both.fold == k).to_numpy()
        if not sel.any():
            continue
        booster = lgb.Booster(model_file=str(out / "folds" / f"fold{k}.txt"))
        pred["lightgbm_float"][sel] = booster.predict(flt[sel].astype(np.float32)).argmax(1)
        trees, real_k, edges_k = ex.build_trees(booster, FEATURES)
        blob = ex.pack(trees, ex.leaf_scale(trees))
        feat, thr, leaf = ex.unpack(blob, len(trees))
        pred["export_float_features"][sel] = ex.run_trees(ex.bin_matrix(ints_float[sel], edges_k), feat, thr, leaf).argmax(1)
        pred["export_reference"][sel] = ex.run_trees(ex.bin_matrix(ref_i[sel], edges_k), feat, thr, leaf).argmax(1)
    kap = {}
    for name, p in pred.items():
        t = both[scored].assign(pred=p[scored.to_numpy()])
        per = t.groupby("subject").apply(lambda g: cv._kappa(g.y4.to_numpy(), g.pred.to_numpy()), include_groups=False)
        kap[name] = per
        summary[f"kappa4_{name}"] = float(per.mean())
        summary[f"kappa4_pooled_{name}"] = float(cv._kappa(t.y4, t.pred))
        sw = t.groupby("subject").apply(lambda g: cv._kappa(g.y4.to_numpy() > 0, g.pred.to_numpy() > 0), include_groups=False)
        summary[f"kappa_sw_{name}"] = float(sw.mean())
    summary["scored_epochs"] = int(scored.sum())
    summary["scored_nights"] = int(kap["export_reference"].size)
    d = kap["export_reference"] - kap["export_float_features"]
    summary["kappa4_change_reference_minus_float_features_mean"] = float(d.mean())
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    summary["kappa4_change_ci95"] = [float(d.mean() - half), float(d.mean() + half)]
    from scipy import stats

    summary["kappa4_change_wilcoxon_p"] = float(stats.wilcoxon(d.dropna()).pvalue)
    summary["pred_agreement_reference_vs_float_features"] = float(
        (pred["export_reference"][scored.to_numpy()] == pred["export_float_features"][scored.to_numpy()]).mean()
    )
    (out / "feature_agreement_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def costs(out: Path, max_rounds: int) -> None:
    """Heap, CPU and ops of the stager with the numbers measured on the Instinct 3 (mlbench, treebench)."""
    from arcasleep_watchtree.cv import ops_per_epoch
    from arcasleep_watchtree.fixed import Engine
    from arcasleep_watchtree.costs import WATCH_OPS_PER_S

    model = json.loads((out / "model.json").read_text())
    n_trees = model["n_trees"]
    per_number, per_byte = 5.064, 1.064  # heap bytes per Array<Number> element and per ByteArray byte
    rings = sum(Engine(None).size.values())
    n_edges = sum(len(e) for e in model["edges"])
    windows = len([r for r in Engine(None).rows if r["kind"] == "win"])
    heap = {
        "trees_bytes": n_trees * model["stride"] * per_byte,
        "edges_and_tables": (n_edges + 2 * len(model["edges"])) * per_number,
        "rings": rings * per_number,
        "window_state_and_misc": (windows * 3 + 87 + 72 + 200) * per_number + 256 * per_byte,
    }
    heap["total_static"] = sum(heap.values())
    ops = ops_per_epoch(FEATURES, max_rounds)
    tree_s = n_trees / 1886.0  # 4-bit-threshold trees per second, treebench on the watch
    summary = {
        "n_trees": n_trees,
        "tree_bytes_resource": n_trees * model["stride"],
        "ring_elements": rings,
        "n_edges": n_edges,
        "heap_bytes_estimate": heap,
        "heap_total_with_app_baseline_9300": heap["total_static"] + 9300,
        "watch_total_mem": 126944,
        "cost_model_feature_ops_per_epoch": ops["feature_ops"],
        "cost_model_feature_cpu_percent": ops["feature_cpu_percent"],
        "trees_seconds_per_epoch_measured_rate": tree_s,
        "trees_ops_equivalent_per_epoch": tree_s * WATCH_OPS_PER_S,
        "total_ops_per_epoch": ops["feature_ops"] + tree_s * WATCH_OPS_PER_S,
        "total_cpu_percent": 100 * (ops["feature_ops"] + tree_s * WATCH_OPS_PER_S) / (WATCH_OPS_PER_S * 30),
    }
    (out / "costs.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["reference", "compare", "costs"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--training-config", type=Path, default=CONFIG_DIR / "training.json")
    args = parser.parse_args()
    if args.step == "costs":
        costs(args.out, TreeSpec.load(training_config=args.training_config).max_rounds)
    elif args.step == "reference":
        reference(args.out, args.limit, args.workers)
    else:
        compare(args.out, args.features)


if __name__ == "__main__":
    main()

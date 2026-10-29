"""Cross-validation, leave-one-cohort-out, leak probes, quantisation and ablations.

    python -m arcasleep_watchtree.cv --features results/features-2026-10-27 \\
        --uncropped results/cv-2026-10-27/features --out results/cv-2026-10-27 \\
        --model-config config/model.json --training-config config/training.json

Complete-context epochs: the forward window of ``FORWARD_MAX_MIN`` minutes lies fully inside the
recorded stream and at least ``BACK_EPOCHS`` epochs (2 h) of stream precede the epoch, so every
backward window and the heart-rate baseline are warm. Training and scoring use only these
epochs (and only epochs with a PSG stage), except in the setups named ``*_all``.

Metrics are computed per night and averaged over nights; pooled-epoch kappa is for reference.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from arcasleep_watchtree.data.loaders import participant_of
from arcasleep_tree.features import constants as C
from arcasleep_tree.features.costs import BYTES_PER_INT16, TREE_OPS_PER_EPOCH, WATCH_OPS_PER_S
from arcasleep_tree.features.table import (
    EPOCHS_PER_MIN,
    WINDOWS,
    feature_names,
    stream_costs,
    window_costs,
    window_name,
)
from arcasleep_tree.model.definition import CONFIG_DIR, TreeSpec, WatchTree

COHORT_ORDER = ["wearanize", "aauwss", "sleepaccel", "weaver_garmin", "weaver_fitbit"]
FWD_EPOCHS = C.FORWARD_MAX_MIN * EPOCHS_PER_MIN
BACK_EPOCHS = 120 * EPOCHS_PER_MIN
MIN_NIGHT_EPOCHS = 120
ALL = feature_names()
WINDOWED = [window_name(s, st, d, m) for s, st, d, ms in WINDOWS for m in ms]


# ---------------------------------------------------------------------------------- data


def person_of(cohort: str, subject: str) -> str:
    """The participant of a source night id; feature tables carry their own ``person`` column."""
    return participant_of(cohort, subject)


def load_tables(features_dir: Path, uncropped_dir: Path) -> dict[str, pd.DataFrame]:
    """Tables by name; ``wearanize`` is the uncropped build, ``wearanize_cropped`` the old one."""
    tables = {}
    for name in COHORT_ORDER[1:]:
        tables[name] = pd.read_parquet(features_dir / f"features-{name}.parquet")
    tables["wearanize_cropped"] = pd.read_parquet(features_dir / "features-wearanize.parquet")
    tables["wearanize"] = pd.read_parquet(uncropped_dir / "features-wearanize_uncropped.parquet")
    for name, table in tables.items():
        if "in_psg" not in table:
            table["in_psg"] = True
        cohort = "wearanize" if name.startswith("wearanize") else name
        n_rows = table.groupby("subject").epoch.transform("size").to_numpy()
        table["ctx"] = (table.epoch.to_numpy() >= BACK_EPOCHS) & (
            table.epoch.to_numpy() + FWD_EPOCHS <= n_rows - 1
        )
        if "person" not in table:
            table["person"] = [person_of(cohort, s) for s in table.subject]
        table["cohort"] = cohort
        table["scored"] = table.y4.to_numpy() >= 0
        table["row_epoch"] = table.epoch.to_numpy()
        # Stream coverage of each window: the position-derived signal the probe looks for.
        k = table.epoch.to_numpy()
        for m in sorted({m for _, _, _, ms in WINDOWS for m in ms}):
            n = m * EPOCHS_PER_MIN
            table[f"cov_b{m}"] = (np.minimum(n, k + 1) / n).astype(np.float32)
            table[f"cov_f{m}"] = (np.minimum(n, n_rows - 1 - k) / n).astype(np.float32)
        table["warm"] = (k >= BACK_EPOCHS).astype(np.float32)
        for col in WINDOWED:
            table[f"nan_{col}"] = np.isnan(table[col].to_numpy()).astype(np.float32)
    return tables


def select(table: pd.DataFrame, context: bool) -> pd.DataFrame:
    keep = table.scored & (table.ctx if context else True)
    return table[keep].reset_index(drop=True)


def kept_nights(table: pd.DataFrame) -> list[str]:
    """Nights with at least ``MIN_NIGHT_EPOCHS`` complete-context scored epochs."""
    counts = table[table.scored & table.ctx].groupby("subject").size()
    return sorted(counts[counts >= MIN_NIGHT_EPOCHS].index)


def person_folds(persons: list[str], k: int = 5, seed: int = 0) -> dict[str, int]:
    order = sorted(persons)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return {p: i % k for i, p in enumerate(order)}


# -------------------------------------------------------------------------------- feature sets


def groups() -> dict[str, list[str]]:
    hr = [f for f in ALL if f.startswith("hr")]
    breath = [f for f in ALL if f.startswith("breath")]
    forward = [f for f in ALL if re.search(r"_f\d+$", f) or f == "tnb_min"]

    def minutes(f: str) -> int:
        m = re.search(r"_[bf](\d+)$", f)
        return int(m[1]) if m else 0

    long = [f for f in ALL if minutes(f) > 15 or f in ("tsb_min", "tnb_min", "hr_base")]
    long += [f for f in ALL if f.startswith("hr_rel")]
    drop = lambda names: [f for f in ALL if f not in set(names)]  # noqa: E731
    return {
        "with_hr": ALL,
        "no_hr": drop(hr),
        "no_forward": drop(forward),
        "no_breathing": drop(breath),
        "windows_le15": drop(long),
    }


def ops_per_epoch(features: list[str], max_rounds: int) -> dict[str, float]:
    """On-watch ops per 30-s epoch for a feature set; trees are added at ``4 * max_rounds``."""
    kept = set(features)
    has = lambda prefix: any(f.startswith(prefix) for f in kept)  # noqa: E731
    total = 0.0
    for line in stream_costs():
        if line.family == "heart rate":
            if "baseline" in line.what:
                if "hr_base" in kept or has("hr_rel"):
                    total += line.ops_per_epoch
            elif has("hr"):
                total += line.ops_per_epoch
        elif line.family == "breathing":
            total += line.ops_per_epoch if has("breath") else 0
        elif line.family == "bouts":
            total += 4 + 2  # jerk floor and bout test, needed by the act windows
            total += 3 if "tsb_min" in kept else 0
            total += 90 if "tnb_min" in kept else 0
        else:
            total += line.ops_per_epoch
    for line in window_costs():
        if line.what in kept:
            total += line.ops_per_epoch
    reach: dict[str, list[int]] = {}
    for signal, stat, direction, minutes in WINDOWS:
        for m in minutes:
            if window_name(signal, stat, direction, m) in kept:
                slot = reach.setdefault(signal, [0, 0])
                slot[0 if direction == "b" else 1] = max(slot[0 if direction == "b" else 1], m * EPOCHS_PER_MIN)
    ring_bytes = sum(sum(v) for v in reach.values()) * BYTES_PER_INT16
    trees = TREE_OPS_PER_EPOCH * (4 * max_rounds) / 524
    return {
        "feature_ops": total,
        "feature_cpu_percent": 100 * total / (WATCH_OPS_PER_S * C.EPOCH_S),
        "tree_ops": trees,
        "total_ops": total + trees,
        "total_cpu_percent": 100 * (total + trees) / (WATCH_OPS_PER_S * C.EPOCH_S),
        "n_features": len(features),
        "ring_bytes": ring_bytes,
    }


# -------------------------------------------------------------------------------- fitting


def quantise_leaves(booster) -> float:
    """Round every leaf value to int8 with one scale for the whole model; returns the scale.

    Trees past ``best_iteration`` are left alone and not used by :meth:`WatchTree.predict`.
    """
    values = []
    per_tree = _leaf_counts(booster)
    for tree, count in enumerate(per_tree):
        for leaf in range(count):
            values.append((tree, leaf, booster.get_leaf_output(tree, leaf)))
    top = max(abs(v) for _, _, v in values)
    scale = top / 127.0 if top > 0 else 1.0
    for tree, leaf, value in values:
        booster.set_leaf_output(tree, leaf, float(np.clip(np.rint(value / scale), -127, 127) * scale))
    return scale


def _leaf_counts(booster) -> list[int]:
    info = booster.dump_model()["tree_info"]
    return [t["num_leaves"] for t in info]


def run_folds(
    spec: TreeSpec,
    table: pd.DataFrame,
    features: list[str],
    folds: dict[str, int],
    *,
    max_bin: int = 255,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold predictions of one table (rows already selected), by person fold."""
    fold = table.person.map(folds).to_numpy()
    pred = np.full(len(table), -1)
    rounds = []
    for k in sorted(set(folds.values())):
        test = fold == k
        if not test.any() or test.all():
            continue
        model = WatchTree(spec, features, max_bin, seed).fit(table[~test])
        pred[test] = model.predict(table[test])
        rounds.append(model.best_iteration)
    out = table[["cohort", "subject", "person", "epoch", "y4"]].copy()
    out["pred"] = pred
    return out, {"mean_rounds": float(np.mean(rounds))}


def run_both(spec, table, features, folds, max_bin=255, seed=0):
    """Float and int8-leaf predictions from the same fitted models."""
    fold = table.person.map(folds).to_numpy()
    plain = np.full(len(table), -1)
    quant = np.full(len(table), -1)
    rounds = []
    for k in sorted(set(folds.values())):
        test = fold == k
        if not test.any() or test.all():
            continue
        model = WatchTree(spec, features, max_bin, seed).fit(table[~test])
        plain[test] = model.predict(table[test])
        quantise_leaves(model.booster)
        quant[test] = model.predict(table[test])
        rounds.append(model.best_iteration)
    base = table[["cohort", "subject", "person", "epoch", "y4"]]
    return base.assign(pred=plain), base.assign(pred=quant), float(np.mean(rounds))


def run_loco(spec, tables, held_out, features, *, exclude=(), seed=0):
    """Train on every other cohort's complete-context epochs; predict the held-out cohort."""
    train = pd.concat(
        [select(tables[c], True) for c in COHORT_ORDER if c != held_out and c not in exclude],
        ignore_index=True,
    )
    test = select(tables[held_out], True)
    model = WatchTree(spec, features, 255, seed).fit(train)
    out = test[["cohort", "subject", "person", "epoch", "y4"]].copy()
    out["pred"] = model.predict(test)
    return out, float(model.best_iteration)


# -------------------------------------------------------------------------------- metrics


def _kappa(a, b) -> float:
    if len(np.unique(a)) < 2 and len(np.unique(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def per_night(pred: pd.DataFrame, nights: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for subject, g in pred.groupby("subject"):
        if nights is not None and subject not in nights:
            continue
        y, p = g.y4.to_numpy(), g.pred.to_numpy()
        rows.append(
            {
                "subject": subject,
                "n_epochs": len(g),
                "kappa4": _kappa(y, p),
                "kappa_sw": _kappa(y > 0, p > 0),
                "deep_pred": float(np.mean(p == 2)),
                "deep_psg": float(np.mean(y == 2)),
                "rem_pred": float(np.mean(p == 3)),
                "rem_psg": float(np.mean(y == 3)),
                "tst_pred_min": float(np.sum(p > 0)) * C.EPOCH_S / 60,
                "tst_psg_min": float(np.sum(y > 0)) * C.EPOCH_S / 60,
            }
        )
    frame = pd.DataFrame(rows)
    frame["tst_err_min"] = frame.tst_pred_min - frame.tst_psg_min
    return frame


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan, np.nan, np.nan)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def summarise(nights: pd.DataFrame, pred: pd.DataFrame) -> dict:
    mean, lo, hi = ci95(nights.kappa4)
    return {
        "unit": "night",
        "n_nights": int(len(nights)),
        "n_epochs": int(len(pred)),
        "kappa4_mean": mean,
        "kappa4_lo": lo,
        "kappa4_hi": hi,
        "kappa_sw_mean": float(np.nanmean(nights.kappa_sw)),
        "deep_pred_mean": float(nights.deep_pred.mean()),
        "deep_psg_mean": float(nights.deep_psg.mean()),
        "rem_pred_mean": float(nights.rem_pred.mean()),
        "rem_psg_mean": float(nights.rem_psg.mean()),
        "tst_err_min_mean": float(nights.tst_err_min.mean()),
        "tst_abs_err_min_mean": float(nights.tst_err_min.abs().mean()),
        "pooled_kappa4": _kappa(pred.y4, pred.pred),
        "pooled_kappa_sw": _kappa(pred.y4 > 0, pred.pred > 0),
    }


# -------------------------------------------------------------------------------- probes


def probe_features(kind: str) -> list[str]:
    if kind == "nan":
        return [f"nan_{c}" for c in WINDOWED] + ["warm"]
    return sorted(c for c in ALL_COVERAGE) + ["warm"]


ALL_COVERAGE = [f"cov_{d}{m}" for d in "bf" for m in sorted({m for _, _, _, ms in WINDOWS for m in ms})]


# -------------------------------------------------------------------------------- driver


class Collector:
    """Per-night rows and summaries of every experiment, written as they accumulate."""

    def __init__(self, out: Path):
        self.out = out
        self.nights: list[pd.DataFrame] = []
        self.summary: list[dict] = []
        self.oof: list[pd.DataFrame] = []
        self.extra: dict[tuple, dict] = {}

    def add(self, experiment, cohort, config, pred, nights=None, keep_oof=False, **extra):
        pred = pred[pred.pred >= 0]
        table = per_night(pred, nights)
        pred = pred[pred.subject.isin(table.subject)]
        for key, value in (("experiment", experiment), ("cohort", cohort), ("config", config)):
            table.insert(0, key, value)
        self.nights.append(table)
        row = {"experiment": experiment, "cohort": cohort, "config": config}
        self.summary.append({**row, **summarise(table, pred), **extra})
        if keep_oof:
            self.oof.append(pred.assign(experiment=experiment, config=config))
        print(experiment, cohort, config, f"n={len(table)}", f"k4={self.summary[-1]['kappa4_mean']:.3f}", flush=True)

    def write(self):
        if not self.nights:
            return
        pd.concat(self.nights, ignore_index=True).to_csv(self.out / "per_night.csv", index=False)
        pd.DataFrame(self.summary).to_csv(self.out / "summary.csv", index=False)
        if self.oof:
            pd.concat(self.oof, ignore_index=True).to_parquet(self.out / "oof_predictions.parquet")


def counts_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, t in tables.items():
        kept = kept_nights(t)
        scored = t[t.scored]
        ctx = t[t.scored & t.ctx & t.subject.isin(kept)]
        rows.append(
            {
                "table": name,
                "nights_built": t.subject.nunique(),
                "scored_epochs_all": len(scored),
                "nights_kept": len(kept),
                "scored_epochs_kept": len(ctx),
                "share_of_scored_kept": len(ctx) / len(scored),
                "persons_kept": ctx.person.nunique(),
            }
        )
    return pd.DataFrame(rows)


def run_all(
    spec: TreeSpec, features_dir: Path, uncropped_dir: Path, out: Path, stages: set[str]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tables = load_tables(features_dir, uncropped_dir)
    counts_table(tables).to_csv(out / "counts.csv", index=False)
    nights = {name: kept_nights(t) for name, t in tables.items()}
    for name in tables:
        tables[name] = tables[name][tables[name].subject.isin(nights[name])].reset_index(drop=True)
    sets = groups()
    col = Collector(out)

    if "cv" in stages:
        for cohort in COHORT_ORDER:
            table = select(tables[cohort], True)
            folds = person_folds(sorted(table.person.unique()))
            for bins in (255, 15):
                plain, quant, rounds = run_both(spec, table, ALL, folds, max_bin=bins)
                suffix = "" if bins == 255 else "_bin15"
                col.add("cv_within", cohort, f"with_hr{suffix}", plain, keep_oof=bins == 255, rounds=rounds)
                col.add("cv_within", cohort, f"with_hr{suffix}_int8", quant, rounds=rounds)
            for config in ("no_hr",) + (
                ("no_forward", "no_breathing", "windows_le15") if cohort == "wearanize" else ()
            ):
                pred, info = run_folds(spec, table, sets[config], folds)
                col.add("cv_within", cohort, config, pred, keep_oof=config == "no_hr", rounds=info["mean_rounds"])
            col.write()

    if "loco" in stages:
        ctx_tables = tables
        for held in COHORT_ORDER:
            variants = {"": ()}
            if held.startswith("weaver"):
                sibling = "weaver_fitbit" if held == "weaver_garmin" else "weaver_garmin"
                variants["_no_sibling"] = (sibling,)
            for tag, exclude in variants.items():
                for config in ("with_hr", "no_hr"):
                    pred, rounds = run_loco(spec, ctx_tables, held, sets[config], exclude=exclude)
                    col.add("loco", held, config + tag, pred, keep_oof=held == "wearanize", rounds=rounds)
            col.write()

    if "leak" in stages:
        for cohort in COHORT_ORDER:
            setups = {"cropped_all": (tables["wearanize_cropped"] if cohort == "wearanize" else tables[cohort], False),
                      "ctx": (tables[cohort], True)}
            if cohort == "wearanize":
                setups["cropped_ctx"] = (tables["wearanize_cropped"], True)
                setups["fixed_all"] = (tables["wearanize"], False)
            persons = sorted(select(tables[cohort], True).person.unique())
            folds = person_folds(persons)
            common = nights[cohort]
            for setup, (table, context) in setups.items():
                sel = select(table, context)
                sel = sel[sel.subject.isin(common) & sel.person.isin(folds)].reset_index(drop=True)
                for label, feats in (("model", ALL), ("probe_nan", probe_features("nan")),
                                     ("probe_coverage", probe_features("cov"))):
                    pred, _ = run_folds(spec, sel, feats, folds)
                    col.add("leak", cohort, f"{setup}/{label}", pred)
            col.write()
    if "regions" in stages:
        leak_regions(spec, tables, nights["wearanize"], out)
    col.write()


def leak_regions(spec, tables, common, out: Path) -> None:
    """Wearanize+ kappa by position in the PSG span, with cropped and uncropped features.

    Regions are the first ``BACK_EPOCHS`` epochs, the last ``FWD_EPOCHS`` epochs and the middle
    (the epochs that are complete-context on the cropped stream). All setups use the same folds
    and the same nights; the fixed model is trained and scored on every scored PSG epoch.
    """
    folds = person_folds(sorted(select(tables["wearanize"], True).person.unique()))
    setups = {
        "cropped features, trained on all scored epochs": (tables["wearanize_cropped"], False),
        "cropped features, trained on complete-context epochs": (tables["wearanize_cropped"], True),
        "uncropped features, trained on all scored epochs": (tables["wearanize"], False),
    }
    rows = []
    for name, (table, context) in setups.items():
        sel = select(table, context)
        sel = sel[sel.subject.isin(common)].reset_index(drop=True)
        pred, _ = run_folds(spec, sel, ALL, folds)
        start = table[table.in_psg].groupby("subject").epoch.min()
        length = table[table.in_psg].groupby("subject").size()
        pos = pred.epoch.to_numpy() - pred.subject.map(start).to_numpy()
        n = pred.subject.map(length).to_numpy()
        region = np.where(pos < BACK_EPOCHS, "first 2 h", np.where(pos >= n - FWD_EPOCHS, "last 45 min", "middle"))
        for r in ("first 2 h", "middle", "last 45 min"):
            p = pred[region == r]
            if len(p) == 0:
                continue
            rows.append({
                "setup": name, "region": r, "n_epochs": len(p), "n_nights": p.subject.nunique(),
                "pooled_kappa4": _kappa(p.y4, p.pred), "pooled_kappa_sw": _kappa(p.y4 > 0, p.pred > 0),
                "wake_pred": float(np.mean(p.pred == 0)), "wake_psg": float(np.mean(p.y4 == 0)),
            })
        print("regions", name, flush=True)
    pd.DataFrame(rows).to_csv(out / "leak_regions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--uncropped", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stages", default="cv,loco,leak")
    parser.add_argument("--model-config", type=Path, default=CONFIG_DIR / "model.json")
    parser.add_argument("--training-config", type=Path, default=CONFIG_DIR / "training.json")
    args = parser.parse_args()
    spec = TreeSpec.load(args.model_config, args.training_config)
    run_all(spec, args.features, args.uncropped, args.out, set(args.stages.split(",")))


if __name__ == "__main__":
    main()

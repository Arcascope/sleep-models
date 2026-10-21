"""Gap checks for TriDoze candidates: do models call gap on sleep epochs that have accelerometer data?

Every model is one or more checkpoints whose softmax outputs are averaged. Calls are the raw
5-way argmax (no stage divisor, no GGIR filter). An epoch "has data" when its gap row is < 1.

Checks, per model:
1. Gap on sleep with data (SleepAccel and SleepAccel-Clinical test adults of ``--draw``, nights
   as cached): share of PSG sleep epochs (light/deep/REM) with data whose argmax is gap.
   Gate: ≤ 0.1%.
2. Late-context shift test (2026-10-15 §2, ``position.py``): each night as cached (context epoch 0)
   and shifted (1,700 absent epochs before the night: the recording's median spectrum, gap row 1,
   built by ``epoch_padding``). Reports the share of PSG sleep called sleep (argmax light, deep or
   REM) and called gap in both and the change. Gate: sleep called sleep changes by ≥ −1.0 point,
   and gap on PSG sleep with data ≤ 0.1% when shifted.
3. Foundry Apple labels (``--foundry``): the 40 windows at context epoch 0; Apple sleep epochs with
   data whose argmax is gap, pooled and per person/device. Gate: ≤ 0.1% pooled and in every group.
   Also "full" (window as scored) against "cut" (start 240 epochs before Apple's first sleep
   epoch) sleep called sleep on Eric's and Olivia's 27 scored windows.

Output, in ``--out``:
- ``<model>_draw<d>.recordings.csv``  per recording and condition (written once; an existing file
  is skipped, so a rerun only does missing models);
- ``gap_checks.csv``  one row per model, cohort and condition, aggregated from every
  ``*.recordings.csv`` in ``--out``: pooled shares in percent, per-recording means, and the gate
  results (``gate_pass`` 1/0, blank where the row has no gate).

Usage: python gap_checks.py --models independent_draw0=ckpt0.pisces,ckpt1.pisces,ckpt2.pisces
           --draw 0 --out DIR [--no-foundry] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.data.batching import epoch_padding, pad_recording_input, recording_input
from arcasleep.data.constants import GAP_LABEL
from arcasleep.training.driver import softmax
from pisces_lite.utils import pad_or_truncate

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
BASELINE = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results/bidoze-joint-ensemble-2026-10-15")
FOUNDRY = BASELINE.parent / "foundry-apple-stages-2026-10-15"
CACHE_NAME = "12Hz_log_PSD_aligned"
GAP = 4
SLEEP_LABELS = (1, 2, 3, 5)
SHIFT = 1700
LEAD = 240
X_LENGTH = Y_LENGTH = 3000
COHORTS = {"testSleepAccel": "SleepAccel", "testClinical": "SleepAccel-Clinical"}
GAP_GATE_PCT = 0.1
SLEEP_CHANGE_GATE_POINTS = -1.0


def load_model(path: Path):
    from arcasleep_bidoze.model import BiDozeNetModel
    return BiDozeNetModel.loadz(path)


def build_inputs(cache_files: list[tuple[str, str, Path]], foundry_cut: set[str]) -> list[dict]:
    """One entry per recording: its conditions' padded inputs and padded cache labels.

    ``cache_files`` holds (group, recording, npz path); the cached/shifted conditions apply to adults
    and the full/cut conditions to Foundry (``group`` starts with "Foundry"; ``foundry_cut`` names the
    windows that also get the cut).
    """
    out = []
    for group, name, path in cache_files:
        z = np.load(path, allow_pickle=True)
        y = z["y"][0]
        x = recording_input(z["X"][0], y.shape[0], 1)
        if group == "FoundryApple":
            placed = {"full": (x, y)}
            if name in foundry_cut:
                start = max(int(np.argmax(y > 0)) - LEAD, 0)
                placed["cut"] = (x[start:], y[start:])
        else:
            lead = np.full(SHIFT, GAP_LABEL, dtype=y.dtype)
            placed = {"cached": (x, y), "shifted": (np.concatenate([epoch_padding(x, SHIFT), x]), np.concatenate([lead, y]))}
        out.append({"group": group, "recording": name, "conditions": {
            c: (pad_recording_input(xc, X_LENGTH, 1),
                pad_or_truncate(yc, target_length=Y_LENGTH, pad_value=GAP_LABEL, axis=0))
            for c, (xc, yc) in placed.items()}})
    return out


def checkpoint_outputs(path: Path, recordings: list[dict]) -> list[dict[str, np.ndarray]]:
    """One checkpoint's softmax output per recording and condition, (3000, 5) each."""
    model = load_model(path)
    out = []
    for r in recordings:
        names = list(r["conditions"])
        batch = np.stack([r["conditions"][names[min(i, len(names) - 1)]][0] for i in range(2)])
        proba = softmax(np.asarray(model.infer(batch)))
        out.append({c: proba[i] for i, c in enumerate(names)})
    return out


def predict(model_paths: list[Path], recordings: list[dict], cache: dict) -> list[dict[str, np.ndarray]]:
    """Mean softmax output of the checkpoints; ``cache`` keeps each checkpoint's output for other models."""
    for path in model_paths:
        if path not in cache:
            cache[path] = checkpoint_outputs(path, recordings)
    return [{c: np.mean([cache[p][i][c] for p in model_paths], axis=0) for c in r["conditions"]}
            for i, r in enumerate(recordings)]


def recording_rows(model: str, draw: int, recordings: list[dict], probas: list[dict]) -> list[dict]:
    rows = []
    for r, per_condition in zip(recordings, probas):
        for condition, (x, y) in r["conditions"].items():
            call = per_condition[condition].argmax(-1)
            has_data = x[:, -1, 0] < 1.0
            sleep = np.isin(y, SLEEP_LABELS)
            sleep_data = sleep & has_data
            rows.append({
                "model": model, "draw": draw, "group": r["group"], "recording": r["recording"],
                "condition": condition,
                "sleep_epochs": int(sleep.sum()), "sleep_with_data_epochs": int(sleep_data.sum()),
                "gap_on_sleep_with_data_epochs": int((call[sleep_data] == GAP).sum()),
                "gap_on_sleep_epochs": int((call[sleep] == GAP).sum()),
                "sleep_called_sleep_epochs": int(np.isin(call[sleep], (1, 2, 3)).sum()),
                "first_sleep_epoch": int(np.argmax(sleep)) if sleep.any() else -1,
            })
    return rows


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled and per-recording-mean shares per model, group and condition, with the gates."""
    out = []

    def shares(g: pd.DataFrame) -> dict:
        data = g.sleep_with_data_epochs.sum()
        sleep = g.sleep_epochs.sum()
        per = g[g.sleep_epochs > 0]
        return {"recordings": len(g), "sleep_epochs": int(sleep), "sleep_with_data_epochs": int(data),
                "gap_on_sleep_with_data_pct": 100 * g.gap_on_sleep_with_data_epochs.sum() / max(data, 1),
                "gap_on_sleep_pct": 100 * g.gap_on_sleep_epochs.sum() / max(sleep, 1),
                "sleep_called_sleep_pct": 100 * g.sleep_called_sleep_epochs.sum() / max(sleep, 1),
                "gap_on_sleep_recording_mean_pct": 100 * (per.gap_on_sleep_epochs / per.sleep_epochs).mean(),
                "sleep_called_sleep_recording_mean_pct": 100 * (per.sleep_called_sleep_epochs / per.sleep_epochs).mean()}

    for (model, draw), g in rows.groupby(["model", "draw"], sort=False):
        adults = g[g.group != "FoundryApple"]
        for cohort, label in COHORTS.items():
            sub = adults[adults.group == label]
            if sub.empty:
                continue
            by = {c: shares(sub[sub.condition == c]) for c in ("cached", "shifted")}
            for c, s in by.items():
                out.append({"model": model, "draw": draw, "cohort": label, "condition": c, **s,
                            "gate": "gap on sleep with data ≤ 0.1%", "gate_pass": int(s["gap_on_sleep_with_data_pct"] <= GAP_GATE_PCT)})
            change = {k: by["shifted"][k] - by["cached"][k] for k in
                      ("gap_on_sleep_with_data_pct", "gap_on_sleep_pct", "sleep_called_sleep_pct",
                       "sleep_called_sleep_recording_mean_pct", "gap_on_sleep_recording_mean_pct")}
            out.append({"model": model, "draw": draw, "cohort": label, "condition": "shifted − cached",
                        "recordings": by["cached"]["recordings"], **change,
                        "gate": "sleep called sleep change ≥ −1.0 point",
                        "gate_pass": int(change["sleep_called_sleep_pct"] >= SLEEP_CHANGE_GATE_POINTS)})
        foundry = g[g.group == "FoundryApple"].copy()
        if foundry.empty:
            continue
        foundry["person_device"] = foundry.recording.str.extract(r"^([A-Za-z]+-[A-Za-z]+)-\d{4}")[0]
        full = foundry[foundry.condition == "full"]
        pooled = shares(full)
        out.append({"model": model, "draw": draw, "cohort": "FoundryApple", "condition": "full, all windows",
                    **pooled, "gate": "gap on Apple sleep with data ≤ 0.1%",
                    "gate_pass": int(pooled["gap_on_sleep_with_data_pct"] <= GAP_GATE_PCT)})
        for pd_name, gg in full.groupby("person_device"):
            s = shares(gg)
            out.append({"model": model, "draw": draw, "cohort": "FoundryApple", "condition": f"full, {pd_name}", **s,
                        "gate": "gap on Apple sleep with data ≤ 0.1%",
                        "gate_pass": int(s["gap_on_sleep_with_data_pct"] <= GAP_GATE_PCT)})
        cut_names = set(foundry[foundry.condition == "cut"].recording)
        if cut_names:
            fc = {c: shares(foundry[(foundry.condition == c) & foundry.recording.isin(cut_names)]) for c in ("full", "cut")}
            for c, s in fc.items():
                out.append({"model": model, "draw": draw, "cohort": "FoundryApple", "condition": f"{c}, scored windows", **s})
            out.append({"model": model, "draw": draw, "cohort": "FoundryApple", "condition": "cut − full, scored windows",
                        "recordings": fc["full"]["recordings"],
                        **{k: fc["cut"][k] - fc["full"][k] for k in ("sleep_called_sleep_pct", "gap_on_sleep_with_data_pct",
                                                                    "sleep_called_sleep_recording_mean_pct")}})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=path[,path...] (softmax outputs averaged)")
    ap.add_argument("--draw", type=int, default=0, help="draw whose test adults and caches are used")
    ap.add_argument("--foundry", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=None, help="at most N recordings per cohort and for Foundry (testing)")
    ap.add_argument("--baseline-run", type=Path, default=BASELINE)
    ap.add_argument("--foundry-run", type=Path, default=FOUNDRY)
    ap.add_argument("--out", type=Path, default=RESULTS / "tridoze-2026-10-16/gap_checks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    models = {}
    for spec in args.models:
        name, _, paths = spec.partition("=")
        models[name] = [Path(p) for p in paths.split(",")]
    todo = {n: p for n, p in models.items() if not (args.out / f"{n}_draw{args.draw}.recordings.csv").exists()}
    for n in models.keys() - todo.keys():
        print(f"skip {n}: {n}_draw{args.draw}.recordings.csv exists")

    if todo:
        draws = json.loads((args.baseline_run / "setup/draws.json").read_text())[f"draw{args.draw}"]
        cache = args.baseline_run / f"setup/cache/draw{args.draw}" / CACHE_NAME
        files = []
        for dataset, label in COHORTS.items():
            ids = draws[dataset][: args.limit]
            files += [(label, sid, cache / f"{dataset}_subject_{sid}_features.npz") for sid in ids]
        foundry_cut: set[str] = set()
        if args.foundry:
            scored = json.loads((args.foundry_run / "evaluate/seed0/evaluation.json").read_text())["subjects"]["FoundryApple"]
            foundry_cut = {n for n in scored if re.match(r"(Eric|Olivia)-", n)}
            windows = sorted(p.name.split("_subject_")[1].removesuffix("_features.npz")
                             for p in (args.foundry_run / "feature_cache" / CACHE_NAME).glob("FoundryApple_subject_*_features.npz"))
            if args.limit:
                windows = windows[: args.limit]
                foundry_cut &= set(windows)
            files += [("FoundryApple", n, args.foundry_run / "feature_cache" / CACHE_NAME / f"FoundryApple_subject_{n}_features.npz")
                      for n in windows]
        recordings = build_inputs(files, foundry_cut)
        print(f"{len(recordings)} recordings", flush=True)
        cache: dict = {}
        for name, paths in todo.items():
            probas = predict(paths, recordings, cache)
            pd.DataFrame(recording_rows(name, args.draw, recordings, probas)).to_csv(
                args.out / f"{name}_draw{args.draw}.recordings.csv", index=False)
            print(f"{name}: done", flush=True)

    rows = pd.concat([pd.read_csv(f) for f in sorted(args.out.glob("*.recordings.csv"))], ignore_index=True)
    table = aggregate(rows)
    table.to_csv(args.out / "gap_checks.csv", index=False, float_format="%.4f")
    show = ["model", "draw", "cohort", "condition", "recordings", "gap_on_sleep_with_data_pct", "sleep_called_sleep_pct", "gate_pass"]
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.max_colwidth", 40):
        print(table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

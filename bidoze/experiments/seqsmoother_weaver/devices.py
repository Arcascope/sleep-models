"""Evaluate devices separately using the same child-held-out Actigraph checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare import FEATURES, inventory, process
from report import METHODS, bootstrap, summarize
from train import MODEL, dataset, predict_dataset, sequence_ends

ERROR_PREFIXES = ("Insufficient increasing timestamps", "No eligible windows")


def prepare_device(job):
    try:
        return process(job)
    except ValueError as error:
        if not str(error).startswith(ERROR_PREFIXES):
            raise
        return dict(filename=job[0].name, device=job[1], status="unusable", reason=str(error))


def recording_rows(path, fold, model_dir):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata"]))
        arrays = {key:d[key] for key in d.files if key != "metadata"}
    if meta["reference_source"] != "kubios_none_mean_hr":
        raise ValueError("Optical-reference recordings belong to the separate secondary study")
    if meta["participant"] not in fold["test"]:
        raise ValueError("Attempt to score a child with a checkpoint that did not hold them out")
    if meta["max_seconds"] is not None:
        raise ValueError("Truncated recording in device study")
    ends = sequence_ends(arrays["endpoint"])
    ends = ends[arrays["eligible"][ends]]
    if not len(ends):
        return [], {**meta, "status":"unscorable", "reason":"No labeled endpoint with full sequence context"}
    learned = {}
    config = json.loads((model_dir / "best.json").read_text())
    if config["fold"] != fold or config["feature_names"] != FEATURES:
        raise ValueError("Checkpoint fold/feature contract mismatch")
    if meta["nightbeat"]:
        r = dict(path=path, meta=meta, arrays=arrays)
        dset = dataset([r], fold["test"], 1)
        template = MODEL.init(jax.random.PRNGKey(0), jnp.zeros((1, 40, 26)))["params"]
        params = serialization.from_bytes(template, (model_dir / "best.msgpack").read_bytes())
        with np.load(model_dir / "best.normalization.npz") as norm:
            if norm["feature_names"].tolist() != FEATURES:
                raise ValueError("Normalization feature order mismatch")
            predictions = predict_dataset(params, dset, norm["mean"], norm["std"])
        learned = {end:float(pred) for (_, end), pred in zip(dset["info"], predictions)}
    rows = []
    for end in ends:
        row = dict(fold=fold["fold"], participant=meta["participant"], night=meta["night"],
            recording=path.stem, device=meta["device"], reference_source=meta["reference_source"],
            native_hz=meta["native_hz"], endpoint=float(arrays["endpoint"][end]),
            reference=float(arrays["reference"][end]), quality=float(arrays["reference_quality"][end]),
            psg_stage=str(arrays["psg_stage"][end]), motion=float(arrays["motion"][end]),
            clarity=float(arrays["peak_wide_clarity"][end]), training_mean=config["training_mean"],
            seqsmoother=learned.get(end, np.nan))
        row.update({m:float(arrays[m][end]) for m in METHODS if m not in ("seqsmoother", "training_mean")})
        rows.append(row)
    return rows, {**meta, "status":"scored", "model_windows":len(learned),
                  "evaluation_windows":len(rows), "model_training_device":"actigraph"}


def per_recording(frame):
    rows = []
    for keys, part in frame.groupby(["device", "reference_source", "participant", "night", "recording"]):
        for method in METHODS:
            valid = np.isfinite(part[method])
            if not valid.any():
                continue
            error = (part.loc[valid, method] - part.loc[valid, "reference"]).abs()
            rows.append(dict(zip(["device", "reference_source", "participant", "night", "recording"], keys),
                method=method, windows=int(valid.sum()), eligible_windows=len(part),
                availability=float(valid.mean()), mae=float(error.mean())))
    return pd.DataFrame(rows)


def paired_devices(frame, rng):
    """Separate device-vs-Actigraph contrasts on identical child/night/endpoints."""
    baseline = frame[frame.device == "actigraph"]
    rows, audit = [], []
    for device, part in frame[frame.device != "actigraph"].groupby("device"):
        pair = part.merge(baseline, on=["participant", "night", "endpoint", "reference_source"],
            suffixes=("_device", "_actigraph"), how="inner", validate="one_to_one")
        if pair.empty:
            audit.append(dict(device=device, common_windows=0, reference_disagreements=0))
            continue
        agree = np.isclose(pair.reference_device, pair.reference_actigraph, atol=1e-6, rtol=0)
        audit.append(dict(device=device, common_windows=len(pair),
            reference_disagreements=int((~agree).sum())))
        if not agree.all():
            raise ValueError("References disagree at matched timestamps for " + device)
        for method in METHODS:
            selected = np.isfinite(pair[method + "_device"]) & np.isfinite(pair[method + "_actigraph"])
            p = pair.loc[selected].copy()
            if p.empty:
                continue
            p["device_error"] = (p[method + "_device"] - p.reference_device).abs()
            p["actigraph_error"] = (p[method + "_actigraph"] - p.reference_device).abs()
            per_night = p.groupby(["participant", "night"])[["device_error", "actigraph_error"]].mean()
            per_child = per_night.groupby("participant").mean()
            delta = per_child.device_error - per_child.actigraph_error
            lo, hi = bootstrap(delta, rng)
            rows.append(dict(device=device, method=method, participants=len(per_child),
                nights=len(per_night), matched_windows=len(p),
                device_macro_mae=float(per_child.device_error.mean()),
                actigraph_macro_mae=float(per_child.actigraph_error.mean()),
                difference_device_minus_actigraph=float(delta.mean()), ci_low=lo, ci_high=hi,
                comparison="Same child, night, reference, endpoint, and method availability"))
    return pd.DataFrame(rows), audit


def report(frame, audit, inv, output, protocol, preparation, full_nightbeat):
    frame.to_csv(output / "predictions_by_device.csv.gz", index=False)
    recordings = per_recording(frame)
    recordings.to_csv(output / "recordings_by_device.csv", index=False)
    participants = recordings.groupby(["device", "reference_source", "participant", "method"], as_index=False).agg(
        mae=("mae", "mean"), recordings=("recording", "nunique"), windows=("windows", "sum"))
    participants.to_csv(output / "participants_by_device.csv", index=False)
    rng = np.random.default_rng(20261007)
    summaries = []
    for (device, source), part in frame.groupby(["device", "reference_source"]):
        masks = [("all", np.ones(len(part), bool)),
                 ("psg_sleep", part.psg_stage.isin(["L", "D", "R", "N1", "N2", "N3", "REM"])),
                 ("high_reference_quality", part.quality >= 0.95),
                 ("nightbeat_retained", np.isfinite(part.nightbeat))]
        for subset, mask in masks:
            if mask.any():
                for row in summarize(part, mask, subset, rng):
                    row.update(device=device, reference_source=source,
                        model_training_device="actigraph" if row["method"] == "seqsmoother" else "",
                        evaluation="held-out Actigraph" if device == "actigraph" else "held-out device transfer")
                    summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary_by_device.csv", index=False)
    paired, pair_audit = paired_devices(frame, rng)
    paired.to_csv(output / "paired_device_vs_actigraph.csv", index=False)
    (output / "recording_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "paired_audit.json").write_text(json.dumps(pair_audit, indent=2) + "\n")
    expected = inv[inv.participant.isin([child for f in protocol["folds"] for child in f["test"]])]
    expected = expected[expected.device.isin(["actigraph", "apple", "fitbit", "garmin"])]
    availability = []
    for device, part in expected.groupby("device"):
        actual = frame[frame.device == device]
        availability.append(dict(device=device, header_children=part.participant.nunique(),
            header_recordings=len(part), scored_children=actual.participant.nunique(),
            scored_recordings=actual.recording.nunique(), eligible_windows=len(actual)))
    pd.DataFrame(availability).to_csv(output / "availability_by_device.csv", index=False)
    status = dict(status="complete" if full_nightbeat else "baselines_complete_model_transfer_pending",
        model_training_device="actigraph", test_children=len({child for fold in protocol["folds"] for child in fold["test"]}),
        aggregation="Within each device/reference: mean epoch absolute error per recording; mean recording MAE per child; equal-child macro mean",
        reference="Kubios None Mean HR; provenance remains unconfirmed",
        device_counts=availability, preparation=preparation,
        axivity="Separate optical-reference cohort; excluded from this Kubios comparison",
        comparison="Raw device summaries use different available children; paired table controls child/night/timestamp")
    (output / "status.json").write_text(json.dumps(status, indent=2, default=int) + "\n")
    display = summary[summary.subset == "all"]
    display = display[display.method.isin(["peak_narrow", "peak_wide", "peak_median", "nightbeat", "training_mean", "seqsmoother"])]
    text = "Status: " + status["status"] + "\n\n" + status["aggregation"] + "\n"
    text += "Neural checkpoints trained on Actigraph only; other devices are transfer evaluations.\n"
    text += "Kubios reference provenance remains unconfirmed.\n\n"
    text += display[["device", "method", "participants", "windows", "availability", "participant_macro_mae"]].to_string(index=False) + "\n"
    (output / "report.txt").write_text(text)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    methods = ["peak_narrow", "peak_wide", "training_mean", "seqsmoother", "nightbeat"]
    devices = ["actigraph", "apple", "fitbit", "garmin"]
    for i, method in enumerate(methods):
        part = display[display.method == method].set_index("device").reindex(devices)
        axes[0].bar(np.arange(len(devices)) + (i - 2) * 0.15, part.participant_macro_mae,
                    width=0.15, label=method)
    axes[0].set_xticks(np.arange(len(devices)), devices)
    axes[0].set(ylabel="Participant-macro MAE (bpm)", title="Separate device summaries; available children differ")
    axes[0].legend(fontsize=8)
    for method in ["peak_narrow", "peak_wide", "seqsmoother"]:
        p = paired[paired.method == method]
        if len(p):
            axes[1].plot(p.device, p.difference_device_minus_actigraph, "o-", label=method)
    axes[1].axhline(0, color="grey", linewidth=1)
    axes[1].set(ylabel="MAE difference from Actigraph (bpm)",
                title="Matched children, nights, and timestamps")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "device_comparison.png", dpi=170)
    plt.close(fig)
    print(text, flush=True)


def secondary_axivity(root, output):
    """Separate optical-reference cohort; no model or Kubios comparisons."""
    rows, audit = [], []
    for path in sorted((root / "axivity-secondary/cache").glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
            ends = np.flatnonzero(d["eligible"])
            audit.append({**meta, "scored_windows":len(ends),
                          "context":"All eligible baseline windows; no neural context constraint"})
            for end in ends:
                row = dict(participant=meta["participant"], night=meta["night"], recording=path.stem,
                    device=meta["device"], reference_source=meta["reference_source"],
                    endpoint=float(d["endpoint"][end]), reference=float(d["reference"][end]))
                row.update({method:float(d[method][end]) if method in
                    ("peak_narrow", "peak_wide", "peak_median", "peak_screened") else np.nan
                    for method in METHODS})
                rows.append(row)
    if rows:
        frame = pd.DataFrame(rows)
        out = []
        rng = np.random.default_rng(20261007)
        for (device, source), part in frame.groupby(["device", "reference_source"]):
            for row in summarize(part, np.ones(len(part), bool), "all_baseline_eligible", rng):
                row.update(device=device, reference_source=source,
                           evaluation="Separate optical-reference cohort, not comparable to Kubios pilot")
                out.append(row)
        pd.DataFrame(out).to_csv(output / "axivity_secondary_summary.csv", index=False)
        per_recording(frame).to_csv(output / "axivity_secondary_recordings.csv", index=False)
    (output / "axivity_secondary_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    root = args.run.resolve()
    output = root / "devices"
    features_dir = output / ("baseline_features" if args.baselines_only else "features")
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "cache").mkdir(exist_ok=True)
    protocol = json.loads((root / "splits.json").read_text())
    inv = inventory(args.data, output)
    ids = [child for f in protocol["folds"] for child in f["test"]]
    chosen = inv[inv.participant.isin(ids) & inv.device.isin(["apple", "fitbit", "garmin"]) & inv.kubios_reference]
    jobs = [(args.data / "raw" / r.filename, r.device, features_dir, root / "vendor/Nightbeat",
             not args.baselines_only, None, False) for r in chosen.itertuples()]
    if not args.report_only:
        with ProcessPoolExecutor(args.workers) as pool:
            preparation = list(pool.map(prepare_device, jobs))
        (features_dir / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    else:
        preparation = json.loads((features_dir / "preparation.json").read_text())
    fold_by_child = {child:fold for fold in protocol["folds"] for child in fold["test"]}
    rows, audit = [], []
    paths = list((features_dir / "cache").glob("*.npz"))
    paths += list((root / "pilot/cache").glob("*.npz"))
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
        child = meta["participant"]
        if child not in fold_by_child:
            continue
        fold = fold_by_child[child]
        recording, detail = recording_rows(path, fold, root / "models" / f"fold{fold['fold']:02d}")
        rows.extend(recording)
        audit.append(detail)
    frame = pd.DataFrame(rows)
    report(frame, audit, inv, output, protocol, preparation, not args.baselines_only)
    secondary_axivity(root, output)


if __name__ == "__main__":
    main()

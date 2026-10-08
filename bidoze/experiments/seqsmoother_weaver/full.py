"""Resume full-cohort GPU cross-validation and update the requested research log."""
from __future__ import annotations
import argparse
import errno
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from devices import prepare_device
from prepare import inventory
from train import sequence_ends

HERE = Path(__file__).resolve().parent
TITLE = "## Side quest: deep learning for heart rate?"


def grouped_folds(ids, seed=20261007):
    ids = list(np.random.default_rng(seed).permutation(sorted(ids)))
    folds = []
    for k, test in enumerate(np.array_split(ids, 5)):
        remaining = [child for child in ids if child not in set(test)]
        shuffled = list(np.random.default_rng(seed + k + 1).permutation(remaining))
        nval = max(1, round(len(shuffled) * .2))
        folds.append(dict(fold=k, train=shuffled[nval:], validation=shuffled[:nval], test=test.tolist()))
    return folds


def update_log(path, body):
    text = path.read_text()
    if TITLE in text:
        before, after = text.split(TITLE, 1)
        next_section = after.find("\n## ")
        tail = after[next_section:] if next_section >= 0 else ""
        text = before.rstrip() + "\n\n" + TITLE + "\n\n" + body.rstrip() + "\n" + tail
    else:
        text = text.rstrip() + "\n\n" + TITLE + "\n\n" + body.rstrip() + "\n"
    path.write_text(text)


def findings(root, stage, log):
    if log is None:
        return
    import pandas as pd
    body = ("[Rohan et al., *Deep Learning for Sleep Heart Rate Estimation from Accelerometers*]"
            "(https://arxiv.org/pdf/2610.06823v1) propose SeqSmoother, a temporal transformer "
            "using accelerometer spectra and intermediate Nightbeat anchors. We tested whether "
            "this correction improves on simple peak finding in revised Weaver.\n\n")
    body += f"**Status: {stage}.** Full-cohort output directory: `{root.name}/`.\n\n"
    body += ("The full study uses five participant-grouped folds. All nights and devices of one child "
             "stay in the same fold. Each child is tested once; 20% of the remaining children are "
             "validation and the rest train. SeqSmoother models train on Actigraph only; Apple, Fitbit "
             "and Garmin are transfer evaluations. Normalization and the constant-HR control use "
             "training participants only. Training uses the selected JAX backend.\n\n")
    body += ("For each device separately, MAE is the mean absolute error per night, then the mean "
             "night MAE per participant, then the equal-participant mean. Device contrasts use "
             "the same child, night, timestamp and reference. Missing context and unavailable "
             "estimates are audited; header availability is not usable coverage.\n\n")
    body += ("This is an external methodological replication, not the paper's dataset or exact "
             "implementation. Windows are 60 s with a 15 s step; the model uses 40-window context. "
             "The narrow peak search is 54–78 bpm and the wide search is 30–120 bpm. The inferred "
             "106,625-parameter architecture and training stride of 40 are departures; validation "
             "and test stride are 1. Reference HR is Kubios None Mean HR, restricted to 30–120 bpm; "
             "its ECG provenance remains unconfirmed. The full CV includes the 38 children used in "
             "the earlier pilot, whose results we had already inspected. Each child is still excluded "
             "from training and validation of the model used to predict their HR. We did not reserve "
             "a separate group of previously unexamined children for a final test.\n\n")
    if (root / "splits.json").exists():
        protocol = json.loads((root / "splits.json").read_text())
        body += f"Cohort: {protocol['candidate_participants']} header-eligible children; {len(protocol['pilot_participants'])} have scorable Actigraph context. Excluded children and recordings are listed in `splits.json` and `pilot/preparation.json`.\n\n"
    summary = root / "devices/summary_by_device.csv"
    if summary.exists():
        table = pd.read_csv(summary)
        table = table[table.subset.eq("all")]
    elif (root / "report/summary.csv").exists():
        table = pd.read_csv(root / "report/summary.csv")
        table = table[table.subset.eq("completed_model_folds")]
        table["device"] = "actigraph"
    else:
        table = None
    if table is not None and len(table):
        body += "### Full-cohort results\n\nThe **constant-HR baseline** predicts the same HR for every test window: the mean reference HR over valid targets in that fold's training sequences. It uses no accelerometer measurements at prediction time. **SeqSmoother** predicts HR from accelerometer features and temporal context; its MAE is the error of those predictions. Both columns use the participant-macro MAE defined above.\n\n| Device | Participants | Narrow peak MAE | Wide peak MAE | Constant-HR baseline MAE | SeqSmoother MAE |\n|:---|---:|---:|---:|---:|---:|\n"
        for device, part in table.groupby("device"):
            by = part.set_index("method")
            values = [f"{by.loc[m, 'participant_macro_mae']:.2f}" if m in by.index else "pending" for m in ["peak_narrow", "peak_wide", "training_mean", "seqsmoother"]]
            body += "| " + device + " | " + str(int(part.participants.max())) + " | " + " | ".join(values) + " |\n"
        body += "\nMAE is in bpm. Participant counts differ by device; compare devices with the matched table, not these raw rows.\n\n"
        paired = root / "devices/paired_device_vs_actigraph.csv"
        if paired.exists():
            q = pd.read_csv(paired)
            body += "| Device vs Actigraph | Method | Participants | Matched MAE difference | 95% participant-bootstrap CI |\n|:---|:---|---:|---:|:---|\n"
            for row in q[q.method.isin(["peak_narrow", "peak_wide", "seqsmoother"])].itertuples():
                body += f"| {row.device} | {row.method} | {row.participants} | {row.difference_device_minus_actigraph:.2f} | {row.ci_low:.2f} to {row.ci_high:.2f} |\n"
            body += "\nNegative differences mean lower error than Actigraph on identical intervals. These transfer results do not test models trained separately on each device.\n\n"
        body += "Audit and uncertainty: `report/summary.csv`, `report/paired.csv`, `devices/availability_by_device.csv`, `devices/recording_audit.json`, and `devices/paired_device_vs_actigraph.csv` under the full-cohort artifact directory.\n\n"
    body += ("### Earlier pilot: completed, 13 held-out participants\n\n"
             "The pilot selected 38 children and tested 13, with 29 training and 8 validation "
             "children per fold. This was a small pilot, not the full Weaver cohort. Actigraph "
             "MAE was 7.91 bpm for SeqSmoother, 10.11 for a training-mean constant, 20.24 for narrow "
             "peak finding and 37.66 for wide peak finding. Nightbeat MAE was 26.58 at 17.3% availability; "
             "SeqSmoother MAE on those same retained intervals was 8.89.\n\n"
             "Apple and Fitbit peak finding had lower error than Actigraph on matched children "
             "and timestamps: narrow-peak differences −4.80 and −4.00 bpm. The Actigraph-trained "
             "neural model instead had higher matched error on Apple and Fitbit: +1.26 and +1.78 bpm. "
             "The small pilot therefore shows device sensitivity and a neural improvement over "
             "these particular baselines; it does not establish the paper's accuracy or prove a "
             "general advantage over our October 6 peak-finding procedure. That procedure's "
             "reported median absolute errors are not directly comparable with mean absolute error.\n\n"
             "Axivity remains a separate optical-reference cohort: four children, narrow peak "
             "MAE 19.81 bpm and wide peak MAE 27.63. It is excluded from the Kubios comparison. "
             "Pilot artifacts: `bidoze/results/seqsmoother-weaver-2026-10-07/`.\n")
    update_log(log, body)


def reuse_cache(source, destination):
    """Reuse a completed cache, including when output is on a different filesystem."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--log", type=Path, help="Optional existing Markdown research log to update")
    ap.add_argument("--platform", choices=("cpu", "cuda"), default="cuda",
                    help="JAX backend for training and model inference (default: cuda)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "pilot/cache"
    cache.mkdir(parents=True, exist_ok=True)
    vendor = a.pilot / "vendor/Nightbeat"
    (root / "vendor").mkdir(exist_ok=True)
    if not (root / "vendor/Nightbeat").exists():
        (root / "vendor/Nightbeat").symlink_to(vendor.resolve(), target_is_directory=True)
    device_cache = root / "devices/features/cache"
    device_cache.mkdir(parents=True, exist_ok=True)
    for path in (a.pilot / "devices/features/cache").glob("*.npz"):
        dest = device_cache / path.name
        if not dest.exists():
            reuse_cache(path, dest)
    inv = inventory(a.data, root)
    selected = inv[inv.device.eq("actigraph") & inv.kubios_reference & inv.psg]
    candidates = sorted(selected.participant.unique())
    findings(root, "full-cohort feature preparation running; pilot results below", a.log)
    started = time.monotonic()
    for path in (a.pilot / "pilot/cache").glob("*.npz"):
        dest = cache / path.name
        if not dest.exists():
            reuse_cache(path, dest)
    jobs = [(a.data / "raw" / r.filename, r.device, root / "pilot", vendor, True, None, False) for r in selected.itertuples()]
    with ProcessPoolExecutor(a.workers) as pool:
        audits = list(pool.map(prepare_device, jobs))
    (root / "pilot/preparation.json").write_text(json.dumps(audits, indent=2) + "\n")
    usable, exclusions = set(), []
    for p in sorted(cache.glob("*.npz")):
        with np.load(p, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
            ends = sequence_ends(d["endpoint"])
            if d["eligible"][ends].any():
                usable.add(meta["participant"])
            else:
                exclusions.append(dict(recording=p.stem, reason="No labeled endpoint with 40-window context"))
    protocol = dict(seed=20261007, candidate_participants=len(candidates), pilot_participants=sorted(usable),
                    excluded_participants=sorted(set(candidates)-usable), exclusions=exclusions,
                    folds=grouped_folds(usable), selection="All header-eligible Actigraph/Kubios/PSG children; folds frozen after signal/label/context eligibility, before model outcomes")
    split = root / "splits.json"
    if split.exists() and json.loads(split.read_text()) != protocol:
        raise ValueError("Existing frozen splits differ")
    split.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Full cohort: {len(candidates)} header-eligible, {len(usable)} scorable participants", flush=True)
    env = dict(os.environ, JAX_PLATFORMS=a.platform, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    expected_platform = "gpu" if a.platform == "cuda" else "cpu"
    check = f"import jax; print(jax.devices()); assert jax.devices()[0].platform == {expected_platform!r}"
    subprocess.run([sys.executable, "-c", check], env=env, check=True)
    report = [sys.executable, str(HERE/"report.py"), "--cache", str(cache), "--splits", str(split), "--folds-dir", str(root/"models"), "--output", str(root/"report")]
    for fold in protocol["folds"]:
        subprocess.run([sys.executable, str(HERE/"train.py"), "--cache", str(cache), "--splits", str(split), "--output", str(root/"models"), "--folds", str(fold["fold"])], env=env, check=True)
        subprocess.run(report, env=env, check=True)
        findings(root, f"GPU training: {fold['fold']+1}/5 folds complete; device transfer pending", a.log)
    findings(root, "Actigraph cross-validation complete; device transfer running", a.log)
    subprocess.run([sys.executable, str(HERE/"devices.py"), "--data", str(a.data), "--run", str(root), "--workers", str(a.workers)], env=env, check=True)
    findings(root, "full-cohort Actigraph cross-validation and device transfer complete", a.log)
    (root/"pipeline_complete.json").write_text(json.dumps(dict(status="complete", elapsed_seconds=time.monotonic()-started, participants=len(usable), platform=a.platform)) + "\n")


if __name__ == "__main__":
    main()

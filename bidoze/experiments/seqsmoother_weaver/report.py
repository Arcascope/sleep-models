"""Report participant-level HR error, availability, paired differences, and coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from train import sequence_ends

METHODS = ["peak_narrow", "peak_wide", "peak_median", "peak_screened",
           "nb_anchor", "nightbeat", "anchor_filled", "anchor_median", "training_mean", "seqsmoother"]


def child_errors(frame, method, mask):
    d = frame.loc[mask, ["participant", "recording", "reference", method]].copy()
    d["error"] = (d[method] - d.reference).abs()
    return d.groupby(["participant", "recording"]).error.mean().groupby("participant").mean()


def centered_r(frame, method, mask):
    d = frame.loc[mask, ["participant", "reference", method]].copy()
    if len(d) < 3:
        return np.nan
    x = d[method] - d.groupby("participant")[method].transform("mean")
    y = d.reference - d.groupby("participant").reference.transform("mean")
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap(values, rng):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return [np.nan, np.nan]
    means = x[rng.integers(0, len(x), size=(5000, len(x)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(frame, mask, subset, rng):
    rows = []
    denominator = int(mask.sum())
    for method in METHODS:
        if method not in frame:
            continue
        selected = mask & np.isfinite(frame[method])
        if not selected.any():
            continue
        values = child_errors(frame, method, selected)
        low, high = bootstrap(values, rng)
        error = (frame.loc[selected, method] - frame.loc[selected, "reference"]).abs()
        rows.append(dict(subset=subset, method=method, participants=len(values),
            windows=int(selected.sum()), eligible_windows=denominator,
            availability=float(selected.sum() / denominator),
            participant_macro_mae=float(values.mean()), mae_ci_low=low, mae_ci_high=high,
            pooled_mae=float(error.mean()), pooled_median_abs_error=float(error.median()),
            within_5_bpm=float((error <= 5).mean()), centered_r=centered_r(frame, method, selected)))
    return rows


def baseline_frame(cache, ids):
    records = []
    for path in sorted(cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
            if meta["participant"] not in ids:
                continue
            ends = sequence_ends(a["endpoint"])
            for end in ends[a["eligible"][ends]]:
                row = dict(participant=meta["participant"], recording=path.stem,
                    device=meta["device"], endpoint=float(a["endpoint"][end]),
                    reference=float(a["reference"][end]), quality=float(a["reference_quality"][end]),
                    psg_stage=str(a["psg_stage"][end]), motion=float(a["motion"][end]),
                    clarity=float(a["peak_wide_clarity"][end]), subharmonic=float(a["X"][end, 5]))
                row.update({m:float(a[m][end]) for m in METHODS if m not in ("seqsmoother", "training_mean")})
                records.append(row)
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits = json.loads(args.splits.read_text())
    test_ids = [child for fold in splits["folds"] for child in fold["test"]]
    frame = baseline_frame(args.cache, test_ids)
    completions, learned = [], []
    for fold in splits["folds"]:
        directory = args.folds_dir / f"fold{fold['fold']:02d}"
        if (directory / "complete.json").exists():
            complete = json.loads((directory / "complete.json").read_text())
            if complete["smoke"]:
                continue
            completions.append(complete)
            learned.append(pd.read_csv(directory / "test_predictions.csv.gz"))
    expected = len(splits["folds"])
    status = dict(expected_folds=expected, completed_folds=len(completions),
        baseline_participants=int(frame.participant.nunique()) if not frame.empty else 0,
        status="complete" if len(completions) == expected else "incomplete",
        reference_provenance="Kubios source unconfirmed; provisional external replication",
        finished_folds=completions)
    (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    if frame.empty:
        print(json.dumps(status), flush=True)
        return
    if learned:
        model_frame = pd.concat(learned, ignore_index=True)
        frame = frame.merge(model_frame[["participant", "recording", "endpoint", "seqsmoother", "training_mean"]],
            on=["participant", "recording", "endpoint"], how="left", validate="one_to_one")
    frame.to_csv(args.output / "predictions.csv.gz", index=False)
    rng = np.random.default_rng(20261007)
    all_mask = np.ones(len(frame), dtype=bool)
    rows = summarize(frame, all_mask, "all_baseline_eligible", rng)
    masks = [("nightbeat_retained", np.isfinite(frame.nightbeat)),
             ("psg_sleep", frame.psg_stage.isin(["L", "D", "R", "N1", "N2", "N3", "REM"])),
             ("high_reference_quality", frame.quality >= 0.95)]
    if "seqsmoother" in frame:
        learned_mask = np.isfinite(frame.seqsmoother)
        masks += [("completed_model_folds", learned_mask),
                  ("model_and_nightbeat_matched", learned_mask & np.isfinite(frame.nightbeat)),
                  ("model_and_nightbeat_rejected", learned_mask & np.isfinite(frame.nb_anchor) & ~np.isfinite(frame.nightbeat))]
    for name, mask in masks:
        if mask.any():
            rows += summarize(frame, mask, name, rng)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    subject_rows = []
    for method in METHODS:
        if method in frame:
            values = child_errors(frame, method, np.isfinite(frame[method]))
            subject_rows += [dict(participant=child, method=method, mae=float(value)) for child, value in values.items()]
    pd.DataFrame(subject_rows).to_csv(args.output / "participants.csv", index=False)
    paired = []
    if "seqsmoother" in frame:
        for method in METHODS:
            if method == "seqsmoother" or method not in frame:
                continue
            mask = np.isfinite(frame.seqsmoother) & np.isfinite(frame[method])
            if not mask.any():
                continue
            left = child_errors(frame, "seqsmoother", mask)
            right = child_errors(frame, method, mask)
            diff = left - right
            low, high = bootstrap(diff, rng)
            p = (float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0) if len(diff) >= 2 else np.nan
            paired.append(dict(comparator=method, participants=len(diff), matched_windows=int(mask.sum()),
                seqsmoother_macro_mae=float(left.mean()), comparator_macro_mae=float(right.mean()),
                difference_seq_minus_comparator=float(diff.mean()), ci_low=low, ci_high=high,
                wilcoxon_p=p))
    pd.DataFrame(paired).to_csv(args.output / "paired.csv", index=False)
    # Use one accelerometer-only ranking for all methods, avoiding target-derived gating.
    coverage_frame = frame.loc[np.isfinite(frame.seqsmoother)].copy() if learned else frame
    score = coverage_frame.clarity / (coverage_frame.motion + 1e-6)
    coverage_rows = []
    for fraction in [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]:
        masks_by_recording = []
        for recording, part in coverage_frame.groupby("recording"):
            n = max(1, int(np.floor(len(part) * fraction)))
            masks_by_recording.extend(score.loc[part.index].nlargest(n).index)
        selected = coverage_frame.index.isin(masks_by_recording)
        for row in summarize(coverage_frame, selected, "acc_quality", rng):
            row["requested_fraction"] = fraction
            row["availability_on_full_grid"] = row["windows"] / len(coverage_frame)
            coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output / "coverage.csv", index=False)
    positive = np.abs(frame.nb_anchor - 2 * frame.reference) <= 5
    anchor = np.isfinite(frame.nb_anchor) & np.isfinite(frame.subharmonic)
    harmonic = dict(anchor_windows=int(anchor.sum()), candidate_windows=int((positive & anchor).sum()),
        positive_participants=int(frame.loc[positive & anchor, "participant"].nunique()),
        interpretation="pooled diagnostic on overlapping windows; no independent-window inference")
    if len(np.unique(positive[anchor])) == 2:
        harmonic["pooled_auroc"] = float(roc_auc_score(positive[anchor], frame.loc[anchor, "subharmonic"]))
    (args.output / "harmonic_diagnostic.json").write_text(json.dumps(harmonic, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ["peak_wide", "peak_median", "nightbeat", "anchor_median", "seqsmoother"]:
        part = coverage[coverage.method == method].sort_values("availability_on_full_grid")
        if len(part):
            ax.plot(100 * part.availability_on_full_grid, part.participant_macro_mae, "o-", label=method)
    ax.set(xlabel="Estimates available (% of eligible evaluation windows)",
           ylabel="Participant-macro MAE (bpm)",
           title=f"Weaver Actigraph / Kubios proxy reference — {len(completions)}/{expected} model folds complete")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "error_coverage.png", dpi=170)
    plt.close(fig)
    main_subset = "completed_model_folds" if learned else "all_baseline_eligible"
    display = summary[summary.subset == main_subset][["method", "participants", "windows", "availability", "participant_macro_mae"]]
    text = "Status: " + status["status"] + "\n\nReference provenance remains unconfirmed.\n\n" + display.to_string(index=False) + "\n"
    (args.output / "report.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()

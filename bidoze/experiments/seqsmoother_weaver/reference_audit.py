"""Audit alternative Weaver HR references on participants selected by frozen splits."""

import argparse
from pathlib import Path
import collections
import json
import numpy as np
import pandas as pd
from prepare import header


def audit(data, predictions, splits, out):
    out.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    for p in (data / "raw").glob("*.csv"):
        h = header(p)
        counts.update(k for k in h if any(t in k for t in ["heartrate", "hr", "kubios"]))
    (out / "weaver_hr_column_headers.json").write_text(
        json.dumps(dict(sorted(counts.items())), indent=2) + "\n"
    )
    old = json.loads(splits.read_text())
    ids = {child for fold in old["folds"] for child in fold["test"]}
    pred = pd.read_csv(predictions)
    pred = pred[pred.participant.isin(ids)]
    rows, source_rows = [], []
    for p in sorted((data / "raw").glob("*.csv")):
        child, night = p.stem.rsplit("_n", 1)
        if child not in ids:
            continue
        h = header(p)
        refs = {
            "kubios_none_mean_hr": ("kubiostime", "kubiosnonemeanhr"),
            "kubios_medium_mean_hr": ("kubiostime", "kubiosmediummeanhr"),
        }
        refs.update(
            {
                d + "_optical_hr": (d + "time", d + "heartrate")
                for d in ["apple", "fitbit", "garmin", "pixel"]
                if d + "heartrate" in h
            }
        )
        keys = {k for pair in refs.values() for k in pair if k in h}
        if "fitabase_hr" in h:
            keys.add("fitabase_hr")
        d = pd.read_csv(p, usecols=[h[k] for k in keys], low_memory=False).rename(
            columns={v: k for k, v in h.items()}
        )
        parts = pred[(pred.participant.eq(child)) & pred.night.eq(int(night))]
        if parts.empty:
            continue
        values = {}
        for source, (tc, hc) in refs.items():
            if tc not in d or hc not in d:
                continue
            s = (
                d[[tc, hc]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .sort_values(tc)
                .drop_duplicates(tc)
            )
            t, y = s[tc].to_numpy(), s[hc].to_numpy()
            source_rows.append(
                dict(
                    participant=child,
                    night=int(night),
                    source=source,
                    finite_samples=len(y),
                    median_hr=float(np.median(y)) if len(y) else None,
                    median_interval=float(np.median(np.diff(t))) if len(t) > 1 else None,
                )
            )
            values[source] = (t, y)
        for device, part in parts.groupby("device"):
            endpoints = part.endpoint.to_numpy()
            y0 = part.reference.to_numpy()
            for source, (t, y) in values.items():
                bounds = np.searchsorted(t, np.column_stack([endpoints - 60, endpoints]))
                for reduction in ["mean", "median"]:
                    reference = np.array(
                        [getattr(np, reduction)(y[a:b]) if b > a else np.nan for a, b in bounds]
                    )
                    valid = np.isfinite(reference) & (reference >= 30) & (reference <= 120)
                    if not valid.any():
                        continue
                    delta = np.abs(reference[valid] - y0[valid])
                    for method in ["peak_narrow", "peak_wide", "seqsmoother"]:
                        estimate = part[method].to_numpy()
                        e = np.abs(estimate[valid] - reference[valid])
                        rows.append(
                            dict(
                                participant=child,
                                night=int(night),
                                device=device,
                                reference_source=source,
                                reduction=reduction,
                                method=method,
                                windows=int(valid.sum()),
                                mae=float(e.mean()),
                                median_abs_error=float(np.median(e)),
                                reference_difference_mean=float(delta.mean()),
                                reference_difference_median=float(np.median(delta)),
                            )
                        )
        print("Audited", p.stem, flush=True)
    if not rows:
        raise ValueError("No matching predictions and finite HR references for the supplied splits")
    recordings = pd.DataFrame(rows)
    recordings.to_csv(out / "weaver_reference_recordings.csv", index=False)
    children = recordings.groupby(
        ["participant", "device", "reference_source", "reduction", "method"], as_index=False
    ).agg(
        mae=("mae", "mean"),
        reference_difference_mean=("reference_difference_mean", "mean"),
        windows=("windows", "sum"),
    )
    summary = children.groupby(
        ["device", "reference_source", "reduction", "method"], as_index=False
    ).agg(
        participants=("participant", "nunique"),
        participant_macro_mae=("mae", "mean"),
        reference_difference_mean=("reference_difference_mean", "mean"),
        windows=("windows", "sum"),
    )
    summary.to_csv(out / "weaver_reference_summary.csv", index=False)
    (out / "weaver_reference_samples.json").write_text(json.dumps(source_rows, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Revised Weaver dataset directory containing raw/*.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Device predictions CSV, including participant, night and endpoint",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="Frozen splits JSON; evaluate the union of its test participants",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for HR header, sample, recording and summary audits",
    )
    args = parser.parse_args()
    audit(args.data, args.predictions, args.splits, args.output)


if __name__ == "__main__":
    main()

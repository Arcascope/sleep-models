"""Compare conditioned Weaver errors with saved SleepAccel peak predictions."""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from train import sequence_ends


def analyze(weaver_run, old_epochs, output):
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    paths = list((weaver_run / "pilot/cache").glob("*.npz"))
    paths += list((weaver_run / "devices/features/cache").glob("*.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
            ends = sequence_ends(a["endpoint"])
            ends = ends[a["eligible"][ends]]
            if not len(ends):
                continue
            motion = a["motion"]
            clarity = a["peak_narrow_clarity"]
            f = pd.DataFrame(
                {
                    "participant": meta["participant"],
                    "recording": path.stem,
                    "device": meta["device"],
                    "reference": a["reference"][ends],
                    "peak_narrow": a["peak_narrow"][ends],
                    "peak_wide": a["peak_wide"][ends],
                    "still_clear": ((motion < np.median(motion)) & (clarity > np.median(clarity)))[
                        ends
                    ],
                }
            )
            frames.append(f)
    if not frames:
        raise ValueError("No scorable caches in the supplied Weaver run")
    d = pd.concat(frames, ignore_index=True)
    rows = []
    for device, part in d.groupby("device"):
        for name, mask in [
            ("all", np.ones(len(part), bool)),
            ("in_band", part.reference.between(54, 78)),
            ("still_clear", part.still_clear),
            ("still_clear_in_band", part.still_clear & part.reference.between(54, 78)),
        ]:
            p = part.loc[mask].copy()
            for method in ["peak_narrow", "peak_wide"]:
                p["error"] = (p[method] - p.reference).abs()
                rows.append(
                    dict(
                        dataset="Weaver",
                        device=device,
                        subset=name,
                        method=method,
                        participants=p.participant.nunique(),
                        windows=len(p),
                        participant_macro_mae=p.groupby(["participant", "recording"])
                        .error.mean()
                        .groupby("participant")
                        .mean()
                        .mean(),
                        pooled_median_abs_error=p.error.median(),
                        pooled_mae=p.error.mean(),
                        reference_median=p.reference.median(),
                        reference_outside_band=(~p.reference.between(54, 78)).mean(),
                    )
                )
    old = pd.read_csv(old_epochs, dtype={"subject_id": str})
    old = old[old.cohort.eq("SleepAccel")].dropna(subset=["hr_est", "watch_hr"])
    for name, mask in [
        ("all", np.ones(len(old), bool)),
        ("in_band", old.watch_hr.between(54, 78)),
        ("still_clear", old.still_clear),
        ("still_clear_in_band", old.still_clear & old.watch_hr.between(54, 78)),
    ]:
        p = old.loc[mask].copy()
        e = (p.hr_est - p.watch_hr).abs()
        rows.append(
            dict(
                dataset="SleepAccel_Oct6",
                device="apple",
                subset=name,
                method="peak50_narrow",
                participants=p.subject_id.nunique(),
                windows=len(p),
                participant_macro_mae=e.groupby(p.subject_id).mean().mean(),
                pooled_median_abs_error=e.median(),
                pooled_mae=e.mean(),
                reference_median=p.watch_hr.median(),
                reference_outside_band=(~p.watch_hr.between(54, 78)).mean(),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(output / "conditioned_error_comparison.csv", index=False)
    print(
        out[
            (out.device.eq("actigraph"))
            | (out.dataset.eq("SleepAccel_Oct6"))
            | ((out.device.eq("apple")) & out.method.eq("peak_narrow"))
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weaver-run",
        type=Path,
        required=True,
        help="Completed Weaver run containing pilot/cache and devices/features/cache",
    )
    parser.add_argument(
        "--old-epochs",
        type=Path,
        required=True,
        help="Earlier hr-epochs.csv.gz containing SleepAccel predictions",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for conditioned_error_comparison.csv"
    )
    args = parser.parse_args()
    analyze(args.weaver_run, args.old_epochs, args.output)


if __name__ == "__main__":
    main()

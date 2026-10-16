#!/usr/bin/env python3
"""Freeze the transplant manifest from features and PSG labels only, before any inference.

Writes into ``<out>``:

* ``eligibility.csv``: per SENSE recording (one person each), targets under the
  prediction-CSV observation proxy (``is_gap == 0``) and under the feature mask (gap row
  == 0, every frame present), PSG sleep and wake epochs in the targets, distant observed
  hours, and donor blocks.
* ``targets.csv``: one row per target.
* ``assignments.npz``: for each target, its slots and, per draw, the donor recording and
  the wake and sleep donor blocks for every slot.
* ``joins.csv`` / ``reuse.csv``: standardized feature jumps at every join, and donor-block
  reuse, per dose, carrier and draw.

No model output is read: the prediction CSV supplies only the PSG, ``is_gap`` and
``is_spt`` columns.

    python manifest.py --cache <SENSE feature cache> --predictions <release seed0 CSVs> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (BUFFER, DOSES, DRAWS, MIN_REMOTE, SLEEP_STAGES, SLOT, TARGET, donor_blocks,
                    eligible_starts, load_recording, remote_order, slots)

NAMES = [f"sen{i:03d}" for i in range(3, 16)]


def carrier_rows(donor, blocks, slot_list):
    """Positions and donor rows for every slot, each slot taking the first len(slot)
    epochs of its block."""
    pos = np.concatenate(slot_list)
    rows = np.concatenate([donor.x[b:b + len(s)] for b, s in zip(blocks, slot_list)])
    return pos, rows


def natural_jumps(rec, scale=None):
    """Jumps between consecutive observed epochs of the unmodified recording: their
    (median, IQR), or, given that scale, the standardized jumps."""
    spec = rec.x[:rec.n, :-1].reshape(rec.n, -1)
    ok = rec.observed[1:] & rec.observed[:-1]
    j = np.linalg.norm(np.diff(spec, axis=0), axis=1)[ok]
    if scale is not None:
        return (j - scale[0]) / scale[1]
    q1, med, q3 = np.percentile(j, [25, 50, 75])
    return med, q3 - q1


def join_z(rec, pos, rows, source, scale):
    """Standardized jumps at every boundary between epochs from different sources."""
    x = rec.x[:rec.n, :-1].reshape(rec.n, -1).copy()
    src = np.full(rec.n, -1)
    x[pos] = rows[:, :-1].reshape(len(pos), -1)
    src[pos] = source
    t = np.arange(1, rec.n)
    at = t[(src[t] != src[t - 1]) & rec.observed[t] & rec.observed[t - 1]]
    j = np.linalg.norm(x[at] - x[at - 1], axis=1)
    return (j - scale[0]) / scale[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs = {n: load_recording(args.cache, args.predictions, n) for n in NAMES}
    pools = {n: donor_blocks(r) for n, r in recs.items()}

    elig, targets = [], []
    for n, r in recs.items():
        proxy = eligible_starts(r, r.csv_observed)
        starts = eligible_starts(r, r.observed)
        remote = [len(remote_order(r, s)) / 120 for s in starts]
        stage = np.concatenate([r.psg[s:s + TARGET] for s in starts]) if starts else np.array([])
        elig.append({
            "recording": n, "epochs": r.n,
            "partial_epochs": int((r.csv_observed & ~r.observed).sum()),
            "targets_csv_proxy": len(proxy), "targets": len(starts),
            "targets_60min": len(starts), "targets_6h": len(starts), "targets_all": len(starts),
            "target_sleep_epochs": int(np.isin(stage, SLEEP_STAGES).sum()),
            "target_wake_epochs": int((stage == "wake").sum()),
            "remote_min_hours": round(min(remote), 3) if remote else np.nan,
            "remote_max_hours": round(max(remote), 3) if remote else np.nan,
            "donor_wake_blocks": len(pools[n]["wake"]), "donor_sleep_blocks": len(pools[n]["sleep"]),
        })
        for s in starts:
            st = r.psg[s:s + TARGET]
            targets.append({
                "target": f"{n}_{s}", "recording": n, "start": s,
                "sleep_epochs": int(np.isin(st, SLEEP_STAGES).sum()),
                "wake_epochs": int((st == "wake").sum()),
                "spt_epochs": int(r.is_spt[s:s + TARGET].sum()),
                "remote_epochs": len(remote_order(r, s)),
            })
    elig = pd.DataFrame(elig)
    targets = pd.DataFrame(targets)
    elig.to_csv(args.out / "eligibility.csv", index=False)
    targets.to_csv(args.out / "targets.csv", index=False)

    assign, joins, reuse, pooled = {}, [], [], {}
    scales = {n: natural_jumps(r) for n, r in recs.items()}
    for ti, t in targets.iterrows():
        r = recs[t.recording]
        slot_list = slots(r, t.start)
        order = remote_order(r, t.start)
        assert sum(len(s) for s in slot_list) == len(order)
        donors_used = []
        entry = {"slot_len": np.array([len(s) for s in slot_list]),
                 "slot_pos": np.concatenate(slot_list)}
        for k in range(DRAWS):
            rng = np.random.default_rng([k, NAMES.index(t.recording), int(t.start)])
            others = [n for n in NAMES if n != t.recording and n not in donors_used]
            donor = others[rng.integers(len(others))]
            donors_used.append(donor)
            entry[f"donor{k}"] = np.array(donor)
            for carrier in ("wake", "sleep"):
                blocks = rng.choice(pools[donor][carrier], size=len(slot_list), replace=True)
                entry[f"{carrier}{k}"] = blocks
                pos, rows = carrier_rows(recs[donor], blocks, slot_list)
                source = np.repeat(np.arange(len(slot_list)), [len(s) for s in slot_list])
                for dose, d in DOSES.items():
                    chosen = set(order[:d].tolist()) if d is not None else None
                    keep = np.ones(len(pos), bool) if chosen is None else np.isin(pos, list(chosen))
                    z = join_z(r, pos[keep], rows[keep], source[keep], scales[t.recording])
                    pooled.setdefault((dose, carrier), []).append(z)
                    joins.append({"target": t.target, "recording": t.recording, "dose": dose,
                                  "carrier": carrier, "draw": k, "joins": len(z),
                                  "z_median": float(np.median(z)), "z_q1": float(np.percentile(z, 25)),
                                  "z_q3": float(np.percentile(z, 75))})
                    used = blocks[np.unique(source[keep])]
                    _, counts = np.unique(used, return_counts=True)
                    reuse.append({"target": t.target, "dose": dose, "carrier": carrier, "draw": k,
                                  "slots": len(used), "distinct_blocks": len(counts),
                                  "max_uses": int(counts.max())})
        assign[t.target] = entry
    np.savez_compressed(args.out / "assignments.npz",
                        **{f"{tg}/{k}": v for tg, e in assign.items() for k, v in e.items()})
    pd.DataFrame(joins).to_csv(args.out / "joins.csv", index=False, float_format="%.4f")
    pd.DataFrame(reuse).to_csv(args.out / "reuse.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(elig.to_string(index=False))
    print(f"\n{len(targets)} targets in {targets.recording.nunique()} recordings; "
          f"{targets.sleep_epochs.sum()} PSG sleep epochs, {targets.wake_epochs.sum()} PSG wake epochs")
    natural = np.concatenate([natural_jumps(r, scales[r.name]) for r in recs.values()])
    rows = [{"dose": "none", "carrier": "natural", "joins": len(natural),
             **dict(zip(["z_q1", "z_median", "z_q3", "z_p95"], np.percentile(natural, [25, 50, 75, 95])))}]
    for (dose, carrier), zs in pooled.items():
        z = np.concatenate(zs)
        rows.append({"dose": dose, "carrier": carrier, "joins": len(z),
                     **dict(zip(["z_q1", "z_median", "z_q3", "z_p95"], np.percentile(z, [25, 50, 75, 95])))})
    pooled_df = pd.DataFrame(rows)
    pooled_df.to_csv(args.out / "joins_pooled.csv", index=False, float_format="%.3f")
    print(pooled_df.round(2).to_string(index=False))
    (args.out / "manifest.json").write_text(json.dumps({
        "target_epochs": TARGET, "buffer_epochs": BUFFER, "slot_epochs": SLOT,
        "min_remote_epochs": MIN_REMOTE, "doses": DOSES, "draws": DRAWS,
        "targets": len(targets), "recordings": NAMES,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""On-watch cost per arm from features.py window_costs / stream_costs, restricted to the arm's windows and the offering's columns.

CPU: ops per epoch (window updates + per-sample stream work + trees at the shipped 4 x 130 rounds scale) over the Instinct's
23,640 ops/s. Memory: ring buffer epochs per series (longest backward window + longest forward window kept for that series)
x 2.1 bytes (int16 ByteArray), as costs.ring_epochs. Streams and series not used by an offering are left out:
no breathing in either; no heart-rate stream or hr windows in the accelerometer-only tree. The bout detector's forward scan
costs F epochs instead of 90; the baseline ring holds 12 block means at B = 120 (blocks kept at shorter B: see baseline_shape).
"""
import re, sys
import pandas as pd
from context_common import *
from arcasleep_watchtree import costs as K
from arcasleep_watchtree.features import WINDOWS, window_costs, stream_costs, window_name, EPOCHS_PER_MIN
from context_features import baseline_shape

rows = []
for b, f in ARMS:
    for cfg in ("hr", "nohr"):
        feats = set(arm_features(CFGS[cfg], b, f))
        wc = [l for l in window_costs() if l.what in feats]
        win_ops = sum(l.ops_per_epoch for l in wc)
        reach = {}
        for signal, stat, d, minutes in WINDOWS:
            for m in minutes:
                if window_name(signal, stat, d, m) in feats:
                    r = reach.setdefault(signal, [0, 0])
                    r[0 if d == "b" else 1] = max(r[0 if d == "b" else 1], m * EPOCHS_PER_MIN)
        if "tnb_min" in feats:  # the bout indicator is kept forward for the next-bout scan
            r = reach.setdefault("act", [0, 0]); r[1] = max(r[1], f * EPOCHS_PER_MIN)
        if "tsb_min" in feats:
            reach.setdefault("act", [0, 0])
        ring_epochs = sum(x + y for x, y in reach.values())
        stream = 0.0
        for l in stream_costs():
            if l.family == "breathing":
                continue
            if cfg == "nohr" and l.family == "heart rate":
                continue
            if l.family == "bouts":
                stream += 4 + 2 + 3 + f * EPOCHS_PER_MIN
            else:
                stream += l.ops_per_epoch
        total = stream + win_ops
        cpu = 100 * (total + K.TREE_OPS_PER_EPOCH * (4 * 130) / 524) / (K.WATCH_OPS_PER_S * 30)
        be, nb, lowest, _ = baseline_shape(b)
        base_vals = nb * (2 if cfg == "hr" else 1)  # jerk floor (+ heart-rate baseline) block means
        rows.append({"arm": arm_name(b, f), "B_min": b, "F_min": f, "offering": cfg, "n_features": len(feats),
                     "scoring_latency_min": f, "ring_epochs": ring_epochs, "ring_bytes": round((ring_epochs + base_vals) * K.BYTES_PER_INT16),
                     "window_ops_per_epoch": win_ops, "total_ops_per_epoch": round(total), "cpu_percent": round(cpu, 2)})
out = pd.DataFrame(rows)
out.to_csv(EXP / "watch_cost.csv", index=False)
print(out.to_string())

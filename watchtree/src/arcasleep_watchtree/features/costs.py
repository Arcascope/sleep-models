"""On-watch cost of the features: ops per epoch by family, CPU share and ring-buffer memory.

    python -m arcasleep_watchtree.features.costs

An op is one multiply-add, add or compare; the Instinct 3 runs 23,640 multiply-adds per second
at 100% CPU (int8 ``ByteArray`` dot product), so the CPU share is ops per epoch over
``23,640 * 30``.
"""

from __future__ import annotations

from collections import defaultdict

from arcasleep_watchtree.features import constants as C
from arcasleep_watchtree.features.table import EPOCHS_PER_MIN, WINDOWS, stream_costs, window_costs

WATCH_OPS_PER_S = 23_640
#: Trees: 524 8-bit depth-6 trees take 0.18 s per prediction.
TREE_OPS_PER_EPOCH = 0.18 * WATCH_OPS_PER_S
#: Heap bytes per element of a ``ByteArray`` holding int16 values.
BYTES_PER_INT16 = 2.1


def ring_epochs() -> dict[str, int]:
    """Epochs each series keeps: its longest backward window plus its forward reach."""
    reach: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for signal, _, direction, minutes in WINDOWS:
        slot = 0 if direction == "b" else 1
        reach[signal][slot] = max(reach[signal][slot], max(minutes) * EPOCHS_PER_MIN)
    return {signal: back + forward for signal, (back, forward) in reach.items()}


def summary() -> dict:
    by_family: dict[str, float] = defaultdict(float)
    for line in stream_costs() + window_costs():
        by_family[line.family] += line.ops_per_epoch
    total = sum(by_family.values())
    rings = ring_epochs()
    return {
        "ops_per_epoch_by_family": dict(by_family),
        "features_ops_per_epoch": total,
        "features_cpu_percent": 100 * total / (WATCH_OPS_PER_S * C.EPOCH_S),
        "trees_ops_per_epoch": TREE_OPS_PER_EPOCH,
        "total_cpu_percent": 100 * (total + TREE_OPS_PER_EPOCH) / (WATCH_OPS_PER_S * C.EPOCH_S),
        "ring_epochs": rings,
        "ring_bytes_int16": sum(rings.values()) * BYTES_PER_INT16,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), indent=1))

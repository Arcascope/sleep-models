"""On-watch cost of the features: ops per epoch by family, CPU share and ring-buffer memory.

    python -m arcasleep_watchtree.features.costs watchtree/config/inputs.json

An op is one multiply-add, add or compare; the Instinct 3 runs 23,640 multiply-adds per second
at 100% CPU (int8 ``ByteArray`` dot product), so the CPU share is ops per epoch over
``23,640 * 30``.
"""

from __future__ import annotations

from collections import defaultdict

from arcasleep_tree.features import constants as C
from arcasleep_tree.features.config import FeatureConfig
from arcasleep_tree.features.table import stream_costs, window_costs

WATCH_OPS_PER_S = 23_640
#: Trees: 524 8-bit depth-6 trees take 0.18 s per prediction.
TREE_OPS_PER_EPOCH = 0.18 * WATCH_OPS_PER_S
#: Heap bytes per element of a ``ByteArray`` holding int16 values.
BYTES_PER_INT16 = 2.1


def ring_epochs(config: FeatureConfig) -> dict[str, int]:
    """Epochs each series keeps: its longest backward window plus its forward reach."""
    reach: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for w in config.windows:
        slot = 0 if w.direction == "b" else 1
        reach[w.signal][slot] = max(reach[w.signal][slot], w.epochs)
    return {signal: back + forward for signal, (back, forward) in reach.items()}


def summary(config: FeatureConfig) -> dict:
    by_family: dict[str, float] = defaultdict(float)
    for line in stream_costs() + window_costs(config):
        by_family[line.family] += line.ops_per_epoch
    total = sum(by_family.values())
    rings = ring_epochs(config)
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
    import sys

    print(json.dumps(summary(FeatureConfig.from_json(sys.argv[1])), indent=1))

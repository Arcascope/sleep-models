"""A streaming low baseline of a per-epoch series.

The baseline at epoch ``k`` is the mean of the lowest 3 of the last 12 completed 10-min block
means: a stand-in for a low percentile of the last 2 h that needs a ring of 12 values and sorts
only those 12. A block with fewer than 5 epochs of data is skipped. Before the first block
completes the baseline is the mean of the epochs so far. A block completed at the end of epoch
``k`` first counts at epoch ``k + 1``.
"""

from __future__ import annotations

import numpy as np

from arcasleep_watchtree.features import constants as C


def low_baseline(hr: np.ndarray) -> np.ndarray:
    """The baseline of ``hr`` (NaN epochs are skipped) at every epoch."""
    n_epochs = hr.size
    base = np.full(n_epochs, np.nan)
    blocks: list[float] = []
    block_sum, block_count, partial_sum, partial_count = 0.0, 0, 0.0, 0
    for k in range(n_epochs):
        if np.isfinite(hr[k]):
            block_sum += hr[k]
            block_count += 1
            partial_sum += hr[k]
            partial_count += 1
        if blocks:
            recent = blocks[-C.BASELINE_BLOCKS :]
            base[k] = float(np.mean(sorted(recent)[: C.BASELINE_LOWEST]))
        elif partial_count:
            base[k] = partial_sum / partial_count
        if (k + 1) % C.BLOCK_EPOCHS == 0:
            if block_count >= C.BLOCK_MIN_EPOCHS:
                blocks.append(block_sum / block_count)
            block_sum, block_count = 0.0, 0
    return base

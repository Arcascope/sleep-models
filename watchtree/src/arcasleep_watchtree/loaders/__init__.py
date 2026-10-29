"""Per-dataset readers of original downloads into pisces-lite ``DataSetObject``s; see :mod:`.common`."""

from __future__ import annotations

from pathlib import Path

from pisces_lite.datasets.data_set_object import DataSetObject

from arcasleep_watchtree.loaders import aauwss, foundry, sleepaccel, wearanize, weaver

#: Dataset directory name -> the module that reads it.
LOADERS = {
    "Wearanize": wearanize,
    "AAUWSS": aauwss,
    "SleepAccel": sleepaccel,
    "SleepAccel-Clinical": sleepaccel,
    "weaver-2026-10-revised": weaver,
    "Foundry": foundry,
}


def load(root: Path | str) -> DataSetObject:
    root = Path(root)
    module = LOADERS.get(root.name)
    if module is None:
        raise ValueError(f"{root}: no WatchTree loader for {root.name!r}; known: {sorted(LOADERS)}")
    return module.load(root)

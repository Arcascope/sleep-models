"""``arcasleep process`` with the jerk divided by the time step.

pisces-lite computes the jerk with ``senpy.compute_jerk(..., use_diff=True)``, which is
``‖a[i] − a[i−1]‖``. Here that call also passes ``divide_by_dt=True``, giving
``‖a[i] − a[i−1]‖ / (t[i] − t[i−1])``; this needs a senpy build with that argument (senpy branch
``jerk-divide-dt``). The cache prefix comes from ``config/inputs.json`` as usual.

Usage: python process_jerk_dt.py <arcasleep process arguments>
"""
from __future__ import annotations

import inspect
import sys

import senpy

from arcasleep.cli import main

_COMPUTE_JERK = senpy.compute_jerk
if "divide_by_dt" not in inspect.signature(_COMPUTE_JERK).parameters:
    raise SystemExit("this senpy has no compute_jerk(divide_by_dt=...); build senpy branch jerk-divide-dt")


def _dt_jerk(timestamps, x, y, z, ts_unit="s", use_diff=True):
    return _COMPUTE_JERK(timestamps, x, y, z, ts_unit=ts_unit, use_diff=use_diff, divide_by_dt=True)


if __name__ == "__main__":
    senpy.compute_jerk = _dt_jerk
    sys.argv = ["arcasleep", "process", *sys.argv[1:]]
    sys.exit(main())

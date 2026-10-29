"""Empatica E4 ``ACC.csv`` and ``HR.csv`` files.

Each file starts with two header rows, the Unix start time and the rate, then one sample per row.
The files carry no per-sample timestamps, so sample ``i`` is placed at ``start + i / rate``; this
is a limit of the format.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from arcasleep_watchtree.loaders.common import accel_frame, heart_rate_frame

ACC_FILE = "ACC.csv"
HR_FILE = "HR.csv"
HEADER_ROWS = 2
UNITS_PER_G = 64.0


@dataclass(frozen=True)
class E4:
    acc_start: float
    acc_rate: float
    #: Raw counts, ``(n, 3)``.
    acc: np.ndarray
    hr_start: float
    hr_rate: float
    hr: np.ndarray


def read_e4(read: Callable[[str], bytes]) -> E4:
    """``read(file_name)`` returns the bytes of one file of the recording."""
    acc = pd.read_csv(io.BytesIO(read(ACC_FILE)), header=None).to_numpy(float)
    hr = pd.read_csv(io.BytesIO(read(HR_FILE)), header=None).to_numpy(float)[:, 0]
    return E4(acc[0, 0], acc[1, 0], acc[HEADER_ROWS:], hr[0], hr[1], hr[HEADER_ROWS:])


def accelerometer(e4: E4, start: float) -> pd.DataFrame:
    """The whole stream in g, with sample 0 at ``start``."""
    return accel_frame(start + np.arange(e4.acc.shape[0]) / e4.acc_rate, e4.acc / UNITS_PER_G)


def heart_rate(e4: E4, start: float) -> pd.DataFrame:
    """Heart rate on the clock where accelerometer sample 0 is at ``start``."""
    t = start + (e4.hr_start - e4.acc_start) + np.arange(e4.hr.size) / e4.hr_rate
    return heart_rate_frame(t, e4.hr)

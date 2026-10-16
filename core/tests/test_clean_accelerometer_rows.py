"""Which raw accelerometer rows ``clean_accelerometer_rows`` keeps."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pisces_lite")

from arcasleep.processing import MAX_ABS_G, clean_accelerometer_rows  # noqa: E402


def test_rows_beyond_the_range_are_dropped() -> None:
    accel = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.1, MAX_ABS_G, -MAX_ABS_G, 0.0],
        [0.2, 3e20, 0.0, 1.0],
        [0.3, 0.0, -17.0, 1.0],
        [0.4, np.nan, 0.0, 1.0],
        [0.5, 0.1, 0.2, 0.9],
    ])
    np.testing.assert_array_equal(clean_accelerometer_rows(accel)[:, 0], [0.0, 0.1, 0.5])


def test_timestamps_are_not_range_checked() -> None:
    accel = np.array([[1e9, 0.0, 0.0, 1.0], [1e9 + 0.1, 0.0, 0.0, 1.0]])
    assert clean_accelerometer_rows(accel).shape == (2, 4)

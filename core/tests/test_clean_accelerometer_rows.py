"""Which raw accelerometer rows ``clean_accelerometer_rows`` keeps."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pisces_lite")

from arcasleep.processing import (  # noqa: E402
    MAX_ABS_G,
    FeatureExtractionError,
    clean_accelerometer_rows,
)


def still_rows(n: int) -> np.ndarray:
    return np.column_stack([np.arange(n) * 0.04, np.zeros(n), np.zeros(n), np.ones(n)])


def test_rows_beyond_the_range_are_dropped_with_a_warning() -> None:
    accel = still_rows(5000)
    accel[1, 1:] = [MAX_ABS_G, -MAX_ABS_G, 0.0]
    accel[2, 1] = 3e20
    accel[3, 2] = -17.0
    accel[4, 1] = np.nan
    with pytest.warns(RuntimeWarning):
        clean = clean_accelerometer_rows(accel)
    assert clean.shape[0] == 4997
    assert not np.isin(accel[2:5, 0], clean[:, 0]).any()
    assert accel[1, 0] in clean[:, 0]


@pytest.mark.parametrize("gravity", [9.81, 0.0])
def test_input_not_in_g_with_gravity_is_refused(gravity: float) -> None:
    accel = still_rows(1000)
    accel[:, 3] = gravity
    with pytest.raises(FeatureExtractionError):
        clean_accelerometer_rows(accel)


def test_timestamps_are_not_range_checked() -> None:
    accel = np.array([[1e9, 0.0, 0.0, 1.0], [1e9 + 0.1, 0.0, 0.0, 1.0]])
    assert clean_accelerometer_rows(accel).shape == (2, 4)

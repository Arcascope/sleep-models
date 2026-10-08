import numpy as np
from sleepaccel import peaks50


def test_rectified_jerk_peak_matches_heart_rate_on_complete_60s_window():
    # Rectification doubles the frequency of a sinusoidal coordinate derivative.
    t = np.arange(0, 61, .02)
    raw = np.column_stack([t, np.sin(2*np.pi*.55*t), np.zeros_like(t), np.zeros_like(t)])
    result = peaks50(raw, np.array([60.])).iloc[0]
    assert abs(result.peak50_narrow - 66) < .1
    assert abs(result.peak50_wide - 66) < .1

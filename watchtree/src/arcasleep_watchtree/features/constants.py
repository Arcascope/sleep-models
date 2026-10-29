"""Rates, filters and thresholds shared by every feature.

Each constant is a number the watch app carries; changing one changes the
features, so a trained model is tied to these values.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter

EPOCH_S = 30
#: Accelerometer rate the Instinct delivers and every cohort is brought to.
FS = 25
SAMPLES_PER_EPOCH = EPOCH_S * FS
#: A gap in the raw stream longer than this is not interpolated; the last value is held and the
#: samples are marked invalid.
GAP_S = 1.0
#: An epoch with a smaller fraction of valid 25 Hz samples has no movement or angle features.
MIN_VALID_FRACTION = 0.5

#: Stage 1: 4-pole Butterworth low-pass at 1.5 Hz on the 25 Hz stream, then keep every 5th output.
DECIMATE_1 = 5
FS_1 = FS // DECIMATE_1
LP1_HZ = 1.5
#: Stage 2 (breathing path): 4-pole Butterworth low-pass at 0.8 Hz on the 5 Hz stream, keep
#: every 2nd output.
DECIMATE_2 = 2
FS_2 = FS_1 / DECIMATE_2
LP2_HZ = 0.8
SOS_1 = butter(4, LP1_HZ, fs=FS, output="sos")
SOS_2 = butter(4, LP2_HZ, fs=FS_1, output="sos")
#: 1-pole high-pass on the 2.5 Hz stream, cut-off about 0.04 Hz:
#: y[n] = HP_A * (y[n-1] + x[n] - x[n-1]).
HP_A = 1.0 / (1.0 + 2.0 * np.pi * 0.04 / FS_2)

#: Movement.
JERK_SECOND = FS
#: An epoch is a high-activity bout when its mean jerk is at least this multiple of the night's
#: jerk floor (see :func:`~arcasleep_watchtree.features.baseline.low_baseline`).
BOUT_RATIO = 3.0
#: Angle: mean of each 5-s block of the 5 Hz stream, in degrees; a change under this is "still".
ANGLE_BLOCK_SAMPLES = 25
ANGLE_BLOCK_S = ANGLE_BLOCK_SAMPLES / FS_1
STILL_DEG = 5.0

#: Breathing: Goertzel bins at 2.5 Hz over one epoch (75 samples), bin k at k * 2.5 / 75 Hz.
BREATH_SAMPLES = int(EPOCH_S * FS_2)
BREATH_BINS = np.arange(3, 19)
BREATH_HZ = BREATH_BINS * FS_2 / BREATH_SAMPLES
#: An epoch is still for breathing when every 5-s angle change in it is under this.
BREATH_STILL_DEG = 2.0

#: Low baseline of a per-epoch series: blocks of this many epochs (10 min); the mean of the lowest 3
#: of the last 12 completed blocks (about the lowest quartile of the last 2 h).
BLOCK_EPOCHS = 20
BASELINE_BLOCKS = 12
BASELINE_LOWEST = 3
BLOCK_MIN_EPOCHS = 5

#: Heart rate.
HR_MIN_BPM = 30.0
HR_MAX_BPM = 200.0

#: Time since the last bout is capped here; time to the next bout is capped at the forward reach.
TSB_CAP_MIN = 240
FORWARD_MAX_MIN = 45

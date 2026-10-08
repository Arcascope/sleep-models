import numpy as np
import pandas as pd

from align import epoch_times
from readout import hr_features, subject_metrics


def test_epoch_times_recovers_a_shift():
    rng = np.random.default_rng(0)
    stages = rng.choice(["wake", "light", "deep", "rem"], size=400)
    psg = pd.DataFrame({"time": 600.0 + 30.0 * np.arange(400), "stage": stages})
    pred = pd.Series(list(stages[3:]) + [None] * 3)
    times, shift, agreement = epoch_times(pred, psg)
    assert shift == 3 and agreement == 1.0 and times[0] == 690.0


def test_hr_features_and_metrics_build():
    night = pd.DataFrame({"hr": np.r_[60 + np.sin(np.arange(200) / 7), [np.nan] * 10]})
    f = hr_features(night, "hr")
    assert np.isfinite(f.to_numpy()).all() and f.hr_missing.sum() == 10
    y = np.array([0, 1, 2, 3, 1, 2, -1])
    m = subject_metrics(y, np.array([0, 1, 2, 3, 2, 2, 1]))
    assert 0 < m["kappa_4stage"] < 1

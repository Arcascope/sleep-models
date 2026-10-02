from __future__ import annotations

import numpy as np

from arcasleep.data.constants import GAP_LABEL
from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion
from arcasleep.training.balanced import selection_confusion


def test_a_scored_epoch_predicted_as_gap_counts_as_a_miss() -> None:
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    right = y_true.copy()
    one_gap = right.copy()
    one_gap[2] = 4

    assert cohen_kappa_from_confusion(selection_confusion(y_true, right, 4)) == 1.0
    assert cohen_kappa_from_confusion(selection_confusion(y_true, one_gap, 4)) < 1.0
    assert selection_confusion(y_true, one_gap, 4)[2, 4] == 1


def test_unscored_and_gap_truth_stay_out_of_the_selection_confusion() -> None:
    y_true = np.array([0, -1, GAP_LABEL, 4, 1])
    y_pred = np.array([0, 0, 4, 4, 1])

    assert selection_confusion(y_true, y_pred, 4).sum() == 2

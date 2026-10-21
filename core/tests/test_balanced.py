from __future__ import annotations

import numpy as np
import pytest

from pisces_lite.cv import SubjectRef

from arcasleep.data.cache import participant_id
from arcasleep.data.constants import GAP_LABEL
from arcasleep.evaluation.staging_metrics import cohen_kappa_from_confusion
from arcasleep.training.balanced import (
    Patience,
    Plateau,
    PlateauDecision,
    selection_confusion,
    split_train_eval,
)


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


def _run(plateau: Plateau, kappas: list[float]) -> list[PlateauDecision]:
    decisions = []
    for kappa in kappas:
        decisions.append(plateau.update(kappa))
        if decisions[-1].stop:
            break
    return decisions


def test_patience_in_epochs_drops_and_stops_after_that_many_epochs_without_a_best() -> None:
    plateau = Plateau(Patience(lr_drop=2, early_stop=5, unit="epochs"), steps_per_epoch=7)

    decisions = _run(plateau, [0.1, 0.3] + [0.2] * 10)

    assert [d.drop_lr for d in decisions] == [False, False, False, True, False, True, False]
    assert len(decisions) == 7 and decisions[-1].stop


def test_patience_in_steps_does_not_depend_on_the_epoch_size() -> None:
    patience = Patience(lr_drop=40, early_stop=100, unit="steps")
    kappas = [0.1, 0.3] + [0.2] * 200

    small = _run(Plateau(patience, steps_per_epoch=10), kappas)
    large = _run(Plateau(patience, steps_per_epoch=50), kappas)

    assert len(small) == 2 + 10
    assert len(large) == 2 + 2


def test_one_high_evaluation_does_not_reset_a_smoothed_patience() -> None:
    kappas = [0.2, 0.2, 0.2, 0.6] + [0.2] * 20
    raw = Plateau(Patience(lr_drop=50, early_stop=5, unit="steps"), steps_per_epoch=1)
    smoothed = Plateau(
        Patience(lr_drop=50, early_stop=5, unit="steps", smoothing_steps=4), steps_per_epoch=1
    )

    raw_decisions = _run(raw, kappas)
    smoothed_decisions = _run(smoothed, kappas)

    assert raw.best_step == 4
    assert smoothed.best_smoothed == pytest.approx(0.3)
    assert len(smoothed_decisions) < len(kappas)
    assert raw_decisions[-1].stop and smoothed_decisions[-1].stop


def test_a_high_first_evaluation_does_not_set_the_bar_before_the_window_fills() -> None:
    kappas = [0.3, 0.1, 0.2, 0.2] + [0.22] * 3 + [0.1] * 20
    plateau = Plateau(Patience(lr_drop=50, early_stop=6, smoothing_steps=4, unit="steps"), 1)

    _run(plateau, kappas)

    assert plateau.best_step >= 4
    assert plateau.best_smoothed < 0.3


def test_an_evaluation_without_a_kappa_is_left_out_of_the_smoothing() -> None:
    plateau = Plateau(Patience(lr_drop=5, early_stop=9, smoothing_steps=3, unit="steps"), 1)

    decisions = _run(plateau, [0.2, float("nan"), 0.4])

    assert decisions[-1].smoothed_kappa == pytest.approx(0.3)


def test_patience_counts_only_from_the_first_kappa_above_the_start() -> None:
    kappas = [0.05] * 10 + [0.4] + [0.3] * 10
    counted = Plateau(Patience(lr_drop=50, early_stop=3, unit="steps"), 1)
    started = Plateau(Patience(lr_drop=50, early_stop=3, unit="steps", start_kappa=0.3), 1)

    counted_decisions = _run(counted, kappas)
    started_decisions = _run(started, kappas)

    assert len(counted_decisions) == 4 and counted_decisions[-1].stop
    assert started.best_step == 11
    assert len(started_decisions) == 14 and started_decisions[-1].stop


@pytest.mark.parametrize(
    "kwargs",
    [{"unit": "minutes"}, {"lr_drop": 0}, {"early_stop": 0}, {"smoothing_steps": -1}],
)
def test_an_impossible_patience_is_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Patience(**{"lr_drop": 1, "early_stop": 2, **kwargs})


def test_the_eval_holdout_keeps_every_recording_of_a_participant_on_one_side() -> None:
    refs = [
        SubjectRef(dataset, f"{child}_{recording}")
        for child in [f"child{i:02d}" for i in range(20)]
        for dataset, recording in [("devices", "apple_n1"), ("devices", "fitbit_n1"), ("nights", "n2")]
    ] + [SubjectRef("single", f"solo{i:02d}") for i in range(10)]

    for seed in range(5):
        fit, held_out = split_train_eval(refs, 0.3, seed)
        fit_participants = {participant_id(r.subject_id) for r in fit}
        eval_participants = {participant_id(r.subject_id) for r in held_out}
        assert fit_participants.isdisjoint(eval_participants)
        assert len(eval_participants) == 9
        assert sorted(fit + held_out) == sorted(refs)


def test_the_eval_holdout_does_not_depend_on_the_order_of_the_recordings() -> None:
    refs = sorted(
        SubjectRef(dataset, f"p{participant:02d}_{night}")
        for dataset, participants in [("cohort_b", range(0, 12)), ("cohort_a", range(8, 20))]
        for participant in participants
        for night in ("n1", "n2")
    )
    rng = np.random.default_rng(0)

    for seed in range(5):
        split = split_train_eval(refs, 0.3, seed)
        for _ in range(5):
            permuted = [refs[i] for i in rng.permutation(len(refs))]
            assert split_train_eval(permuted, 0.3, seed) == split

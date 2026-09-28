import numpy as np
import pytest

# No senpy guard: pisces_lite.proc resolves its pipeline lazily, so importing
# prepare for a label helper does not pull in the compiled backend.
from autofish.data.prepare import MIN_SCORED_MINUTES, scored_minutes


def test_scored_minutes_counts_only_staged_epochs() -> None:
    # 0=Wake, 1..5 sleep stages, -1 PSG_MASK, -2 PAD: 4 scored epochs at 30s.
    labels = np.array([0, 1, 5, 3, -1, -1, -2])
    assert scored_minutes(labels) == pytest.approx(2.0)


def test_scored_minutes_is_zero_for_a_wholly_unscored_record() -> None:
    assert scored_minutes(np.full(37, -1)) == 0.0


def test_default_threshold_separates_a_short_night_from_a_full_one() -> None:
    # 119.5 min of staging was the closest real exclusion in the weaver cohort.
    assert scored_minutes(np.zeros(239)) < MIN_SCORED_MINUTES
    assert scored_minutes(np.zeros(240)) >= MIN_SCORED_MINUTES

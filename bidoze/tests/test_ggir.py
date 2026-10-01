"""The GGIR-style sleep filter: z-angle, SIB, sub-recording splits, SPT, naps."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bidoze import ggir
from bidoze.predict import _apply_ggir_filter, _summarize


def test_z_angle_matches_ggir_formula() -> None:
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([1.0, -1.0, 0.0])

    angle = ggir.z_angle(x, y, z)

    assert angle[0] == pytest.approx(45.0)
    assert angle[1] == pytest.approx(-45.0)
    assert angle[2] == pytest.approx(0.0)


def test_sub_recording_bounds_split_on_a_long_lapse() -> None:
    timestamps = np.array([0.0, 1.0, 2.0, 4000.0, 4001.0])

    assert ggir.sub_recording_bounds(timestamps) == [(0, 3), (3, 5)]


def test_van_hees_sib_flags_a_quiet_stretch_but_not_a_restless_one() -> None:
    quiet = np.zeros(600)
    restless = np.tile([0.0, 10.0], 300)

    assert ggir.van_hees_sib(quiet).all()
    assert not ggir.van_hees_sib(restless).any()


def test_rolling_median_extends_edges() -> None:
    values = np.array([0.0, 10.0, 0.0, 0.0, 10.0, 0.0])

    median = ggir.rolling_median(values, 3)

    np.testing.assert_allclose(median, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(ggir.rolling_median(np.array([1.0, 2.0, 3.0]), 3), [1, 2, 3])


def _night_and_day(seconds: int, fs: float = 1.0) -> np.ndarray:
    """One quiet night followed by a restless day, on a 1 Hz clock."""
    t = np.arange(seconds) / fs
    night = t < seconds / 3
    z = np.where(night, 0.0, np.tile([0.0, 10.0], seconds)[:seconds])
    accel = np.column_stack([t, np.ones(seconds), np.zeros(seconds), z])
    return accel


def test_spt_runs_and_covers_the_quiet_block_on_a_long_recording() -> None:
    accel = _night_and_day(12 * 3600)
    epoch_start_s = accel[0, 0] + 30.0 * np.arange(accel.shape[0] // 30)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    assert result.spt_ran
    assert len(result.spt_windows) == 1
    early = epoch_start_s < accel.shape[0] / 3
    assert result.is_spt[early].mean() > 0.9
    assert not result.is_spt[~early].any()


def test_a_recording_below_the_minimum_span_is_all_spt() -> None:
    accel = _night_and_day(8 * 3600)
    epoch_start_s = accel[0, 0] + 30.0 * np.arange(accel.shape[0] // 30)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    assert not result.spt_ran
    assert result.is_spt.all()


def test_a_short_sub_recording_is_all_spt_but_the_lapse_before_it_is_not() -> None:
    long_day = [("restless", 3.0), ("quiet", 4.0), ("restless", 3.0)]
    accel = _recording([*long_day, ("lapse", 2.0), ("restless", 8.0)])
    epoch_start_s = _epochs(accel)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    hours = (epoch_start_s - accel[0, 0]) / 3600.0
    assert result.spt_ran
    assert not result.is_spt[(hours >= 10.0) & (hours < 12.0)].any()
    assert result.is_spt[(hours >= 12.0) & (hours < 19.9)].all()


def test_two_nights_get_one_main_window_each() -> None:
    # 48 h so it is binned into two 24 h stretches; quiet the start of each.
    accel = _night_and_day(48 * 3600)
    restless = accel[:, 0] % (24 * 3600)
    accel[:, 3] = np.where(restless < 6 * 3600, 0.0, np.resize([0.0, 10.0], accel.shape[0]))
    epoch_start_s = accel[0, 0] + 30.0 * np.arange(accel.shape[0] // 30)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    assert result.spt_ran
    assert len(result.spt_windows) == 2


def _recording(segments: list[tuple[str, float]], fs: float = 10.0) -> np.ndarray:
    """Concatenate ``("quiet" | "restless" | "lapse", hours)`` segments at ``fs``.

    Restless flips the z-angle every 5 s bin, so it survives the 5 s pooling;
    a lapse is simply absent samples.
    """
    parts, clock = [], 0.0
    for kind, hours in segments:
        seconds = hours * 3600.0
        if kind != "lapse":
            t = clock + np.arange(int(seconds * fs)) / fs
            z = np.zeros_like(t) if kind == "quiet" else 10.0 * (np.floor(t / 5.0) % 2)
            parts.append(np.column_stack([t, np.ones_like(t), np.zeros_like(t), z]))
        clock += seconds
    return np.concatenate(parts)


def _epochs(accel: np.ndarray) -> np.ndarray:
    return accel[0, 0] + 30.0 * np.arange(int((accel[-1, 0] - accel[0, 0]) // 30.0) + 1)


def test_each_sub_recording_gets_its_own_window_and_the_lapse_gets_none() -> None:
    day = [("restless", 3.0), ("quiet", 4.0), ("restless", 3.0)]
    accel = _recording([*day, ("lapse", 2.0), *day])
    epoch_start_s = _epochs(accel)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    hours = (epoch_start_s - accel[0, 0]) / 3600.0
    lapse = (hours >= 10.0) & (hours < 12.0)
    assert result.spt_ran
    assert len(result.spt_windows) == 2
    assert not result.is_spt[lapse].any()
    for night_start in (3.0, 15.0):
        night = (hours >= night_start + 0.5) & (hours < night_start + 3.5)
        assert result.is_spt[night].all()


def test_a_dropout_does_not_shift_later_flags_in_time() -> None:
    # A 20-minute dropout (not a lapse) inside the quiet stretch.
    accel = _recording(
        [("quiet", 2.0), ("lapse", 1.0 / 3.0), ("quiet", 2.0), ("restless", 2.0)]
    )
    epoch_start_s = _epochs(accel)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    hours = (epoch_start_s - accel[0, 0]) / 3600.0
    restless_from = 2.0 + 1.0 / 3.0 + 2.0
    assert result.is_sib[(hours > 2.5) & (hours < restless_from - 0.1)].all()
    assert not result.is_sib[hours > restless_from + 0.1].any()
    assert not result.is_sib[(hours > 2.05) & (hours < 2.3)].any()


def _window_hours(result, accel) -> list[tuple[float, float]]:
    origin = accel[0, 0]
    return [((t0 - origin) / 3600.0, (t1 - origin) / 3600.0) for t0, t1 in result.spt_windows]


def test_a_dropout_between_still_stretches_counts_for_its_full_length() -> None:
    # Bridged, the split block is 20 + 40 + 20 = 80 min and beats the 50 min
    # block; with the dropout left out it would be only 40 min.
    accel = _recording(
        [
            ("restless", 4.0),
            ("quiet", 50 / 60),
            ("restless", 2.0),
            ("quiet", 20 / 60),
            ("lapse", 40 / 60),
            ("quiet", 20 / 60),
            ("restless", 3.0),
        ]
    )
    epoch_start_s = _epochs(accel)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    ((start, stop),) = _window_hours(result, accel)
    split_block = 4.0 + 50 / 60 + 2.0
    assert start == pytest.approx(split_block, abs=0.1)
    assert stop == pytest.approx(split_block + 80 / 60, abs=0.1)


def test_a_dropout_next_to_movement_is_movement_and_is_not_filled() -> None:
    # 30 min restless + 40 min dropout = a 70 min movement gap, too long to fill.
    accel = _recording(
        [
            ("restless", 3.0),
            ("quiet", 2.0),
            ("restless", 0.5),
            ("lapse", 40 / 60),
            ("quiet", 1.5),
            ("restless", 3.0),
        ]
    )
    epoch_start_s = _epochs(accel)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    ((start, stop),) = _window_hours(result, accel)
    assert start == pytest.approx(3.0, abs=0.1)
    assert stop == pytest.approx(5.0, abs=0.1)


def test_a_still_block_at_the_end_of_a_recording_is_not_extended() -> None:
    accel = _recording([("restless", 7.0), ("quiet", 3.0)])
    epoch_start_s = _epochs(accel)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    ((start, stop),) = _window_hours(result, accel)
    assert start == pytest.approx(7.0, abs=0.1)
    assert stop <= (accel[-1, 0] - accel[0, 0]) / 3600.0 + ggir.GGIR_EPOCH_SECONDS / 3600.0


def test_consolidation_removes_short_blocks_before_filling_gaps() -> None:
    per_min = int(60 / ggir.GGIR_EPOCH_SECONDS)
    still, moving = np.ones(10 * per_min, bool), np.zeros(10 * per_min, bool)
    alternating = np.tile(np.concatenate([still, moving]), 36)

    assert not ggir._consolidate_nomov(alternating).any()


def test_consolidation_fills_interior_gaps_but_not_the_edges() -> None:
    per_min = int(60 / ggir.GGIR_EPOCH_SECONDS)
    block = lambda minutes, value: np.full(minutes * per_min, value, dtype=bool)  # noqa: E731
    mask = np.concatenate(
        [block(20, False), block(120, True), block(40, False), block(120, True), block(20, False)]
    )

    out = ggir._consolidate_nomov(mask)

    assert not out[: 20 * per_min].any()
    assert out[20 * per_min : -20 * per_min].all()
    assert not out[-20 * per_min :].any()


def test_consolidation_drops_blocks_of_exactly_the_minimum() -> None:
    per_min = int(60 / ggir.GGIR_EPOCH_SECONDS)
    edge = np.zeros(per_min, bool)
    mask = np.concatenate([edge, np.ones(30 * per_min, bool), edge])

    assert not ggir._consolidate_nomov(mask).any()


def test_van_hees_sib_counts_stillness_before_the_first_and_after_the_last_change() -> None:
    angle = np.zeros(8000)
    angle[5000] = 45.0

    sib = ggir.van_hees_sib(angle)

    assert sib.all()


def test_van_hees_sib_with_no_long_stretch_follows_the_posture_change_count() -> None:
    # Both spans are under 5 minutes, so neither holds a long enough stretch.
    four_changes = np.zeros(50)
    four_changes[[10, 30]] = 45.0
    ten_changes = np.zeros(50)
    ten_changes[[5, 15, 25, 35, 45]] = 45.0

    assert ggir.van_hees_sib(four_changes).all()
    assert not ggir.van_hees_sib(ten_changes).any()


def test_filter_keeps_sleep_inside_spt_and_flags_outside_spt_sleep_as_naps() -> None:
    probabilities = np.array(
        [
            [0.1, 0.6, 0.2, 0.1],  # sleep, quiet and inside SPT -> stays
            [0.1, 0.6, 0.2, 0.1],  # sleep, not a SIB but inside SPT -> stays
            [0.1, 0.6, 0.2, 0.1],  # sleep, SIB but outside SPT -> kept, nap
            [0.7, 0.1, 0.1, 0.1],  # already wake
        ]
    )
    is_gap = np.zeros(4, dtype=bool)
    ggir_result = SimpleNamespace(
        is_sib=np.array([True, False, True, True]),
        is_spt=np.array([True, True, False, False]),
        spt_ran=True,
    )

    stage, is_nap = _apply_ggir_filter(
        probabilities, ("wake", "light", "deep", "rem"), is_gap, ggir_result
    )

    assert list(stage) == ["light", "light", "light", "wake"]
    assert list(is_nap) == [0, 0, 1, 0]


def test_filter_demotes_non_sib_sleep_only_outside_spt() -> None:
    probabilities = np.tile([0.1, 0.6, 0.2, 0.1], (3, 1))
    is_gap = np.zeros(3, dtype=bool)
    # All three are non-SIB sleep, but the middle epoch is inside the SPT.
    ggir_result = SimpleNamespace(
        is_sib=np.zeros(3, dtype=bool),
        is_spt=np.array([False, True, False]),
        spt_ran=True,
    )

    stage, is_nap = _apply_ggir_filter(
        probabilities, ("wake", "light", "deep", "rem"), is_gap, ggir_result
    )

    assert list(stage) == ["wake", "light", "wake"]
    assert list(is_nap) == [0, 0, 0]


def test_gaps_and_demoted_sleep_outside_spt_are_not_naps() -> None:
    probabilities = np.tile([0.1, 0.6, 0.2, 0.1], (3, 1))
    is_gap = np.array([True, False, False])
    outside_spt = SimpleNamespace(
        is_sib=np.array([True, False, True]), is_spt=np.zeros(3, dtype=bool), spt_ran=True
    )

    stage, is_nap = _apply_ggir_filter(
        probabilities, ("wake", "light", "deep", "rem"), is_gap, outside_spt
    )

    assert list(stage) == ["gap", "wake", "light"]
    assert list(is_nap) == [0, 0, 1]


def test_out_of_order_timestamps_do_not_crash_evaluate() -> None:
    accel = _recording([("quiet", 1.0), ("lapse", 2.0), ("quiet", 1.0)], fs=1.0)
    accel = accel[::-1].copy()
    epoch_start_s = accel[:, 0].min() + 30.0 * np.arange(10)

    result = ggir.evaluate(accel, epoch_start_s, epoch_start_s.size)

    assert result.is_sib.shape == (10,)


def test_gap_epochs_stay_gap_and_sleep_outside_spt_is_a_nap() -> None:
    probabilities = np.array([[0.1, 0.6, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1]])
    is_gap = np.array([True, False])
    no_spt = SimpleNamespace(
        is_sib=np.array([True, True]), is_spt=np.zeros(2, dtype=bool), spt_ran=False
    )

    stage, is_nap = _apply_ggir_filter(
        probabilities, ("wake", "light", "deep", "rem"), is_gap, no_spt
    )

    assert list(stage) == ["gap", "light"]
    assert list(is_nap) == [0, 1]


def test_summary_counts_sleep_from_the_given_labels_and_missing_from_gaps() -> None:
    stage_raw = np.array(["light", "wake", "gap", "deep"], dtype=object)
    is_gap = np.array([False, False, True, False])

    summary = _summarize(stage_raw, is_gap, nap_minutes=1.5, subject_id="abc")

    assert summary.subject_id == "abc"
    assert summary.tst_minutes == pytest.approx(1.0)
    assert summary.missing_data_minutes == pytest.approx(0.5)
    assert summary.nap_minutes == pytest.approx(1.5)


def test_summary_from_corrected_labels_excludes_demoted_sleep() -> None:
    probabilities = np.array([[0.1, 0.6, 0.2, 0.1]] * 2)
    is_gap = np.zeros(2, dtype=bool)
    ggir_result = SimpleNamespace(
        is_sib=np.array([True, False]), is_spt=np.array([True, False]), spt_ran=True
    )

    stage, _ = _apply_ggir_filter(
        probabilities, ("wake", "light", "deep", "rem"), is_gap, ggir_result
    )
    summary = _summarize(stage, is_gap, nap_minutes=0.0, subject_id=None)

    assert summary.tst_minutes == pytest.approx(0.5)


def test_sib_does_not_flip_sleep_on_a_recording_without_spt() -> None:
    # A <= 9 h sub-recording is all SPT, so the filter leaves its sleep alone
    # even where vanHees2015 sees no sustained inactivity.
    probabilities = np.tile([0.1, 0.6, 0.2, 0.1], (4, 1))
    is_gap = np.zeros(4, dtype=bool)
    all_spt = SimpleNamespace(
        is_sib=np.array([False, False, False, False]),
        is_spt=np.ones(4, dtype=bool),
        spt_ran=False,
    )

    stage, is_nap = _apply_ggir_filter(
        probabilities, ("wake", "light", "deep", "rem"), is_gap, all_spt
    )
    summary = _summarize(stage, is_gap, nap_minutes=0.0, subject_id=None)

    assert list(stage) == ["light"] * 4
    assert list(is_nap) == [0, 0, 0, 0]
    assert summary.tst_minutes == pytest.approx(2.0)


class _StubModel:
    def infer(self, chunks: np.ndarray) -> np.ndarray:
        return np.zeros((*chunks.shape[:2], 4))


def _stub_predictor():
    from bidoze.predict import Predictor

    predictor = Predictor.__new__(Predictor)
    predictor.model = _StubModel()
    predictor.stage_names = ("wake", "light", "deep", "rem")
    predictor.context_epochs = 8
    predictor.features = lambda accel, origin: (np.zeros((5, 3, 2)), 0.0)
    return predictor


def _break_filter(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ggir, "evaluate", boom)
    monkeypatch.setattr(
        "bidoze.data.batching.split_gap_row", lambda pooled: (None, np.zeros(5))
    )
    monkeypatch.setattr(
        "bidoze.data.prepare.clean_accelerometer_rows", lambda a: a
    )


def test_a_failing_filter_falls_back_to_raw_with_a_warning(monkeypatch) -> None:
    _break_filter(monkeypatch)
    accel = np.zeros((10, 4))

    with pytest.warns(RuntimeWarning):
        result = _stub_predictor().predict(accel)

    assert result.stage is None and result.is_nap is None
    assert result.is_sib is None and result.is_spt is None
    assert result.stage_raw is not None
    assert result.summary.nap_minutes is None


def test_ggir_error_fails_reraises(monkeypatch) -> None:
    _break_filter(monkeypatch)

    with pytest.raises(RuntimeError):
        _stub_predictor().predict(np.zeros((10, 4)), ggir_error_fails=True)

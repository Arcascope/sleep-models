"""Device reporting keeps children, nights, and reference sources separate."""
import io

import numpy as np
import pandas as pd
import pytest

from devices import paired_devices, per_recording
from prepare import cache_arrays
from report import METHODS


def frame(rows):
    d = pd.DataFrame(rows)
    defaults = dict(reference_source="kubios_none_mean_hr", reference=80.,
                    night=1, endpoint=60., recording="night1")
    for key, value in defaults.items():
        if key not in d:
            d[key] = value
    for method in METHODS:
        if method not in d:
            d[method] = np.nan
    return d


def test_device_means_keep_equal_nights_within_child_and_equal_children():
    rows = [dict(device="actigraph", participant="a", recording="a_n1", peak_narrow=80.)] * 100
    rows += [dict(device="actigraph", participant="a", recording="a_n2", peak_narrow=90.),
             dict(device="actigraph", participant="b", recording="b_n1", peak_narrow=82.),
             dict(device="apple", participant="a", recording="a_apple_n1", peak_narrow=180.),
             dict(device="apple", participant="a", recording="a_apple_n2", peak_narrow=280.)]
    recordings = per_recording(frame(rows))
    children = recordings.groupby(["device", "participant"]).mae.mean()
    device = children.groupby("device").mean()
    assert device["actigraph"] == 3.5
    assert device["apple"] == 150.


def test_paired_contrasts_ignore_unmatched_children_and_timestamps():
    d = frame([
        dict(device="actigraph", participant="a", endpoint=60., peak_narrow=81.),
        dict(device="actigraph", participant="a", endpoint=75., peak_narrow=81.),
        dict(device="apple", participant="a", endpoint=75., peak_narrow=84.),
        dict(device="apple", participant="a", endpoint=90., peak_narrow=180.),
        dict(device="apple", participant="b", endpoint=75., peak_narrow=280.),
    ])
    paired, audit = paired_devices(d, np.random.default_rng(0))
    p = paired[paired.method == "peak_narrow"].iloc[0]
    assert p.participants == 1
    assert p.matched_windows == 1
    assert p.device_macro_mae == 4.
    assert p.actigraph_macro_mae == 1.
    assert p.difference_device_minus_actigraph == 3.
    assert audit[0]["reference_disagreements"] == 0


def test_paired_devices_require_the_same_reference_at_each_endpoint():
    d = frame([dict(device="actigraph", participant="a", reference=80., peak_narrow=81.),
               dict(device="apple", participant="a", reference=90., peak_narrow=84.)])
    with pytest.raises(ValueError, match="References disagree"):
        paired_devices(d, np.random.default_rng(0))


def test_pandas_extension_strings_roundtrip_without_pickle():
    d = pd.DataFrame({"psg_stage":pd.Series(["L", "R", ""], dtype="string"),
                      "endpoint":[60., 75., 90.]})
    arrays = cache_arrays(d)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as a:
        assert a["psg_stage"].tolist() == ["L", "R", ""]
        assert np.array_equal(a["endpoint"], [60., 75., 90.])


def test_no_complete_nightbeat_chunks_still_produces_numeric_missing_anchors(monkeypatch):
    import nightbeat_adapter
    monkeypatch.setattr(nightbeat_adapter, "load_functions", lambda vendor: {"get_butter_sos": lambda *args, **kwargs: None})
    from nightbeat_adapter import predict
    empty = predict(np.zeros((100, 3)), np.ones(100, dtype=bool), 0., None)
    merged = pd.DataFrame({"endpoint":[60.]}).merge(empty, on="endpoint", how="left")
    assert np.isnan(merged.nb_anchor.to_numpy()).all()


def test_full_cohort_folds_hold_out_every_child_once():
    from full import grouped_folds
    folds = grouped_folds([f"p{i:03d}" for i in range(235)])
    held_out = [child for fold in folds for child in fold["test"]]
    assert len(held_out) == len(set(held_out)) == 235
    for fold in folds:
        assert not set(fold["train"]) & set(fold["test"])
        assert not set(fold["validation"]) & set(fold["test"])
        assert not set(fold["train"]) & set(fold["validation"])
        assert len(fold["train"]) + len(fold["validation"]) + len(fold["test"]) == 235

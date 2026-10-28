"""Participant ids survive pseudonymization; the BiDoze comparison pairs the same epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arcasleep_watchtree import cohorts, report


def test_keyed_dataset_keeps_one_participant_id_across_nights(tmp_path, monkeypatch):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "arcasleep_pseudonym.key").write_text("ab" * 16)
    monkeypatch.setattr(cohorts, "DATA_ROOT", tmp_path)
    cohort = cohorts.Cohort("wearanize", "ds")
    ids = ["sub1_s1", "sub1_s2", "sub2_s1"]
    people = cohorts.person_ids(cohort, ids)
    assert people["sub1_s1"] == people["sub1_s2"] != people["sub2_s1"]
    assert "sub1" not in people["sub1_s1"]


def test_unkeyed_dataset_reports_the_participant(tmp_path, monkeypatch):
    monkeypatch.setattr(cohorts, "DATA_ROOT", tmp_path)
    cohort = cohorts.Cohort("weaver_garmin", "ds")
    assert cohorts.person_ids(cohort, ["c1_garmin_n1"]) == {"c1_garmin_n1": "c1"}


def test_bidoze_comparison_uses_only_epochs_both_models_scored(tmp_path):
    n = 200
    epochs = np.arange(n)
    stage = np.tile([0, 1, 2, 3], n // 4)
    raw = np.array([0, 1, 3, 5])[stage]
    calls = pd.DataFrame(
        {"subject": "s1", "epoch": epochs, "y_raw": raw, "call": stage}
    ).drop(index=[10, 11])
    calls.loc[calls.epoch == 12, "call"] = 4
    calls.to_parquet(tmp_path / "bidoze_epochs.parquet")
    pd.DataFrame(
        {"subject": "s1", "epoch": epochs, "in_psg": True, "y4": stage}
    ).to_parquet(tmp_path / "features-wearanize_uncropped.parquet")
    oof = pd.concat(
        [
            pd.DataFrame(
                {
                    "experiment": experiment,
                    "config": config,
                    "cohort": "wearanize",
                    "subject": "s1",
                    "person": "p1",
                    "epoch": epochs,
                    "y4": stage,
                    "pred": stage,
                }
            )
            for experiment in ("cv_within", "loco")
            for config in ("with_hr", "no_hr")
        ]
    )
    nights, _ = report.bidoze_nights(tmp_path, oof, tmp_path)
    counts = nights.groupby("config").n_epochs.sum()
    assert counts["bidoze_ctx"] == counts["watchtree_cv_within_with_hr"] == n - 3

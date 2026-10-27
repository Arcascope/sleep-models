"""Preparing recordings with the wrist stream kept beyond the PSG."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pisces_lite")

from pisces_lite.datasets import align_trim  # noqa: E402

from arcasleep.data.constants import GAP_LABEL, OUTSIDE_PSG_LABEL  # noqa: E402
from arcasleep.data.cohorts import SOURCE_FILENAME  # noqa: E402
from arcasleep.data.prepare import (  # noqa: E402
    align_uncropped,
    load_dataset,
    prepare_subject,
    scored_minutes,
)

EPOCH = 30.0


def accel_frame(start: float, stop: float, hz: float = 2.0) -> pd.DataFrame:
    t = np.arange(start, stop, 1.0 / hz)
    return pd.DataFrame(
        {"TIMESTAMP": t, "ACC_X": 0.0, "ACC_Y": 0.0, "ACC_Z": np.ones_like(t)}
    )


def psg_frame(first: float, stages: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"TIMESTAMP": first + EPOCH * np.arange(len(stages)), "PSG": stages}
    )


def test_grid_covers_the_whole_stream_and_keeps_inside_labels() -> None:
    accel = accel_frame(3.0, 1500.0)
    stages = [0, 2, 2, 3, 5, 5, -1, 0]
    psg = psg_frame(307.0, stages)

    kept, grid = align_uncropped(accel, psg)
    _, cropped = align_trim(accel, psg)

    assert grid["TIMESTAMP"].iloc[0] % EPOCH == pytest.approx(307.0 % EPOCH)
    assert np.allclose(np.diff(grid["TIMESTAMP"]), EPOCH)
    assert kept["TIMESTAMP"].min() >= grid["TIMESTAMP"].iloc[0]
    assert kept["TIMESTAMP"].max() <= grid["TIMESTAMP"].iloc[-1]
    labels = grid["PSG"].to_numpy()
    first = int(round((cropped["TIMESTAMP"].iloc[0] - grid["TIMESTAMP"].iloc[0]) / EPOCH))
    assert np.array_equal(labels[first : first + len(cropped)], cropped["PSG"].to_numpy())
    outside = np.ones(labels.size, dtype=bool)
    outside[first : first + len(cropped)] = False
    assert (labels[outside] == OUTSIDE_PSG_LABEL).all()
    assert outside.sum() > 0
    assert first > 0 and first + len(cropped) < labels.size


def test_psg_longer_than_the_stream_leaves_no_outside_epochs() -> None:
    accel = accel_frame(100.0, 400.0)
    psg = psg_frame(0.0, [0] * 30)

    _, grid = align_uncropped(accel, psg)

    assert OUTSIDE_PSG_LABEL not in grid["PSG"].to_numpy()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_streams_that_do_not_overlap_are_refused() -> None:
    with pytest.raises(ValueError):
        align_uncropped(accel_frame(0.0, 100.0), psg_frame(1000.0, [0] * 4))


def write_converted(path, accel: pd.DataFrame, psg: pd.DataFrame, name: str) -> None:
    (path / "cleaned_accelerometer").mkdir(parents=True)
    (path / "cleaned_psg").mkdir(parents=True)
    accel.to_csv(
        path / "cleaned_accelerometer" / "subj_cleaned_motion.out",
        sep=" ", header=False, index=False,
    )
    psg.to_csv(path / "cleaned_psg" / "subj_cleaned_psg.out", sep=" ", header=False, index=False)
    (path / "data_set.json").write_text(
        json.dumps(
            {
                "name": name,
                "csv": {"delimiter": " "},
                "id_pattern_by_feature": {
                    "accelerometer": "<<ID>>_cleaned_motion.out",
                    "psg": "<<ID>>_cleaned_psg.out",
                },
            }
        )
    )


def test_prepare_subject_matches_the_cropped_path_inside_the_psg(tmp_path) -> None:
    accel = accel_frame(-400.0, 1500.0, hz=40.0)
    accel = accel[(accel["TIMESTAMP"] < 600.0) | (accel["TIMESTAMP"] >= 660.0)]
    psg = psg_frame(0.0, [0, 1, 2, 2, 3, 5, 5, 2, 0, 0, 2, 2, -1, 0, 0, 2, 2, 2, 5, 0])
    write_converted(tmp_path / "toy", accel, psg, "toy")
    data_set = load_dataset(tmp_path / "toy")

    cropped = prepare_subject(data_set, "subj")
    whole = prepare_subject(data_set, "subj", uncropped=True)

    assert whole.accel_values.shape[0] > cropped.accel_values.shape[0]
    assert whole.labels.size > cropped.labels.size
    first = int(np.flatnonzero(whole.labels != OUTSIDE_PSG_LABEL)[0])
    inside = whole.labels[first : first + cropped.labels.size]
    # The last PSG epoch holds one cropped sample and is a gap there only.
    assert np.array_equal(inside[:-1], cropped.labels[:-1])
    assert GAP_LABEL in whole.labels
    assert scored_minutes(whole.labels) == pytest.approx(scored_minutes(cropped.labels), abs=1.0)


def write_eesm19_raw(raw, converted, sid: str) -> None:
    sub, ses = "sub-001", "ses-001"
    eeg = raw / sub / ses / "eeg"
    eeg.mkdir(parents=True)
    onsets = 2.0 + EPOCH * np.arange(3)
    pd.DataFrame({"onset": onsets, "duration": EPOCH, "Scoring1": 1}).to_csv(
        eeg / f"{sub}_{ses}_task-sleep_acq-scoring1_events.tsv", sep="\t", index=False
    )
    actigraphy = raw / "sourcedata" / sub / ses / "actigraphy"
    actigraphy.mkdir(parents=True)
    n = int(200 * 100.0)
    lines = ["Device Type,GENEActiv"] + ["2018-06-13 23:00:00:000,0.1,0.2,0.9,0,0,0"] * n
    (actigraphy / "night.csv").write_text("\n".join(lines) + "\n")
    converted.mkdir()
    (converted / "cleaned_psg").mkdir()
    pd.DataFrame({"t": EPOCH * np.arange(3), "s": [0, 2, 2]}).to_csv(
        converted / "cleaned_psg" / f"{sid}_cleaned_psg.out", sep=" ", header=False, index=False
    )
    (converted / "conversion_manifest.json").write_text(
        json.dumps(
            [
                {"id": sid, "raw": f"{sub}/{ses}", "status": "converted",
                 "alignment": {"offset_s": 40.0}},
                {"id": "sub099_s9", "raw": "sub-099/ses-009", "status": "dropped"},
            ]
        )
    )


def test_eesm19_adapter_keeps_samples_outside_the_scored_window(tmp_path) -> None:
    sid = "sub001_s1"
    write_eesm19_raw(tmp_path / "raw", tmp_path / "converted", sid)
    root = tmp_path / "uncropped"
    root.mkdir()
    (root / SOURCE_FILENAME).write_text(
        json.dumps(
            {"cohort": "eesm19", "raw_root": str(tmp_path / "raw"),
             "converted_dir": str(tmp_path / "converted")}
        )
    )
    (root / "pisces_lite_adapter.py").write_text(
        "from arcasleep.data.cohorts import load_data_sets  # noqa: F401\n"
    )
    data_set = load_dataset(root)

    assert list(data_set.ids) == [sid]
    whole = prepare_subject(data_set, sid, uncropped=True)

    assert whole.accel_values[0, 0] < 0.0
    assert whole.accel_values[-1, 0] > 3 * EPOCH
    assert (whole.labels[:1] == OUTSIDE_PSG_LABEL).all()
    assert (whole.labels[-2:-1] == OUTSIDE_PSG_LABEL).all()


def write_xlsx(path, rows: list[list]) -> None:
    def cell(r: int, c: int, value) -> str:
        ref = f"{chr(65 + c)}{r + 1}"
        if isinstance(value, str):
            return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'
        return f'<c r="{ref}"><v>{value}</v></c>'

    body = "".join(
        f'<row r="{r + 1}">' + "".join(cell(r, c, v) for c, v in enumerate(row)) + "</row>"
        for r, row in enumerate(rows)
    )
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{body}</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def test_wearanize_adapter_keeps_samples_outside_the_scored_window(tmp_path) -> None:
    raw = tmp_path / "raw"
    (raw / "3.Manual_synchronization").mkdir(parents=True)
    write_xlsx(
        raw / "3.Manual_synchronization" / "Manual_sync_zmax_psg_emp_actpal.xlsx",
        [["SubjectID", "Emp_start_sec"], ["'Sub001s1'", 40.0], ["'Sub002s1'", -999.0]],
    )
    empatica = raw / "1.Raw_data" / "Sub001s1" / "3.Empatica"
    empatica.mkdir(parents=True)
    n = int(32 * 300)
    acc = "\n".join(f"0,0,64" for _ in range(n))
    with zipfile.ZipFile(empatica / "recording.zip", "w") as z:
        z.writestr("ACC.csv", f"0,0,0\n32.0,32.0,32.0\n{acc}\n")
    converted = tmp_path / "converted"
    (converted / "cleaned_psg").mkdir(parents=True)
    pd.DataFrame({"t": EPOCH * np.arange(4), "s": [0, 2, 2, 0]}).to_csv(
        converted / "cleaned_psg" / "sub001_s1_cleaned_psg.out", sep=" ", header=False, index=False
    )
    root = tmp_path / "uncropped"
    root.mkdir()
    (root / SOURCE_FILENAME).write_text(
        json.dumps(
            {"cohort": "wearanize", "raw_root": str(raw), "converted_dir": str(converted)}
        )
    )
    (root / "pisces_lite_adapter.py").write_text(
        "from arcasleep.data.cohorts import load_data_sets  # noqa: F401\n"
    )
    data_set = load_dataset(root)

    assert list(data_set.ids) == ["sub001_s1"]
    whole = prepare_subject(data_set, "sub001_s1", uncropped=True)

    assert whole.accel_values[0, 0] < 0.0
    assert whole.accel_values[-1, 0] > 3 * EPOCH
    assert whole.accel_values.shape[1] == 4
    assert whole.labels.size > 4
    assert OUTSIDE_PSG_LABEL in whole.labels

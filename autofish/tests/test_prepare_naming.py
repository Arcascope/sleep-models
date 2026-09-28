import json

from autofish.data.prepare import load_dataset


def _write_dataset(root, directory_name: str, dataset_name: str) -> None:
    path = root / directory_name
    (path / "cleaned_accelerometer").mkdir(parents=True)
    (path / "cleaned_psg").mkdir(parents=True)
    for subject in ("a", "b"):
        (path / "cleaned_accelerometer" / f"{subject}.csv").write_text(
            "t,x,y,z\n0,1,0,0\n1,1,0,0\n"
        )
        (path / "cleaned_psg" / f"{subject}.csv").write_text("t,stage\n0,0\n")
    (path / "data_set.json").write_text(json.dumps({"name": dataset_name}))


def test_dataset_name_comes_from_data_set_json_not_directory(tmp_path) -> None:
    _write_dataset(tmp_path, "submission", "SleepAccel-Clinical")

    data_set = load_dataset(tmp_path / "submission", "submission")

    assert data_set.name == "SleepAccel-Clinical"


def test_metadata_name_is_also_accepted_as_the_selector(tmp_path) -> None:
    _write_dataset(tmp_path, "submission", "SleepAccel-Clinical")

    assert load_dataset(tmp_path / "submission", "SleepAccel-Clinical").name == (
        "SleepAccel-Clinical"
    )


def test_single_dataset_loads_without_a_name(tmp_path) -> None:
    _write_dataset(tmp_path, "submission", "SleepAccel-Clinical")

    assert load_dataset(tmp_path / "submission").name == "SleepAccel-Clinical"


def test_directory_name_still_wins_when_no_data_set_json(tmp_path) -> None:
    path = tmp_path / "weaver"
    (path / "cleaned_accelerometer").mkdir(parents=True)
    (path / "cleaned_psg").mkdir(parents=True)
    for subject in ("a", "b"):
        (path / "cleaned_accelerometer" / f"{subject}.csv").write_text(
            "t,x,y,z\n0,1,0,0\n1,1,0,0\n"
        )
        (path / "cleaned_psg" / f"{subject}.csv").write_text("t,stage\n0,0\n")

    assert load_dataset(path, "weaver").name == "weaver"

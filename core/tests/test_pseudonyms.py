import pytest

from arcasleep.data.pseudonyms import (
    PSEUDONYM_KEY_FILENAME,
    cache_ids,
    load_pseudonym_key,
    pseudonym,
)

KEY = bytes(range(32))


def test_a_dataset_without_a_key_keeps_its_ids(tmp_path) -> None:
    assert load_pseudonym_key(tmp_path) is None
    assert cache_ids(None, ["a", "b"]) == {"a": "a", "b": "b"}


def test_a_keyed_dataset_never_caches_a_source_id(tmp_path) -> None:
    (tmp_path / PSEUDONYM_KEY_FILENAME).write_text(KEY.hex() + "\n")
    ids = ["s01", "s02", "s03"]

    mapped = cache_ids(load_pseudonym_key(tmp_path), ids)

    assert set(mapped) == set(ids)
    assert not set(mapped.values()) & set(ids)
    assert len(set(mapped.values())) == len(ids)


def test_pseudonyms_are_stable_under_one_key_and_differ_across_keys() -> None:
    assert pseudonym(KEY, "s01") == pseudonym(KEY, "s01")
    assert pseudonym(KEY, "s01") != pseudonym(bytes(32), "s01")


def test_a_short_key_is_refused(tmp_path) -> None:
    (tmp_path / PSEUDONYM_KEY_FILENAME).write_text("00ff")
    with pytest.raises(ValueError):
        load_pseudonym_key(tmp_path)

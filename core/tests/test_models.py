from __future__ import annotations

from pathlib import Path

import pytest

from arcasleep.models import checkpoint_family


def test_a_file_that_is_not_a_checkpoint_archive_is_a_value_error(tmp_path: Path) -> None:
    junk = tmp_path / "junk.pisces"
    junk.write_bytes(b"not a zip archive")

    with pytest.raises(ValueError):
        checkpoint_family(junk)

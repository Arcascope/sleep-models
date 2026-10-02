from __future__ import annotations

from arcasleep.cli import _extra_for


def test_a_family_command_missing_its_network_names_the_family_extra() -> None:
    assert _extra_for("optax", "bidoze") == "bidoze"
    assert _extra_for("jax", "bidoze") == "bidoze"


def test_other_missing_modules_name_their_own_extra() -> None:
    assert _extra_for("pisces_lite", "bidoze") == "inference"
    assert _extra_for("jax", None) == "inference"
    assert _extra_for("not_a_dependency", "bidoze") is None

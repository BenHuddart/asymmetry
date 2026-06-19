"""Tests for the PSI free-text field-geometry classifier.

PSI ``.bin``/``.mdu``/``.root`` files carry no structured field-state code, so
``field_direction`` is read from an explicit ``TF``/``LF``/``ZF`` tag in the
free-text comment/setup/title (see ``docs/porting/field-geometry/``).  The
classifier must trust only an unambiguous token and never guess.
"""

from __future__ import annotations

import pytest

from asymmetry.core.io.base import field_direction_from_text

pytestmark = [pytest.mark.io]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Real musrfit example comments (see tests/test_psi_loader.py fixtures).
        ("FeSe 9p4 TF100 p107apr09_sample*1p02", "Transverse"),
        ("Y124 TF150G 4.5K (ab)", "Transverse"),
        # Bare and spelled-out tags.
        ("TF", "Transverse"),
        ("wTF 50G", "Transverse"),
        ("transverse field 100 G", "Transverse"),
        ("LF200", "Longitudinal"),
        ("longitudinal calibration", "Longitudinal"),
        ("ZF", "Zero field"),
        ("ZF 0G run", "Zero field"),
        ("zero-field relaxation", "Zero field"),
        ("zero field", "Zero field"),
    ],
)
def test_explicit_tag_classifies(text: str, expected: str) -> None:
    assert field_direction_from_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "FeSe sample 4.5K",  # no geometry tag at all
        "half moon orientation",  # 'half' must not match \blf\b
        "stuff happens",  # must not match \btf\b
        "alfalfa field",  # 'alf' must not match \blf\b
    ],
)
def test_no_or_spurious_tag_returns_unknown(text: str) -> None:
    assert field_direction_from_text(text) == ""


def test_conflicting_tags_are_ambiguous() -> None:
    # Two distinct geometry tags → unknown rather than a misleading guess.
    assert field_direction_from_text("TF100 then LF200") == ""


def test_scans_multiple_fields_and_ignores_blanks() -> None:
    assert field_direction_from_text(None, "", "FeSe", "ZF 0G") == "Zero field"


def test_field_magnitude_alone_is_not_a_signal() -> None:
    # A zero field value must not be read as ZF; only the explicit string counts.
    assert field_direction_from_text("FeSe 0 G 4.5K") == ""

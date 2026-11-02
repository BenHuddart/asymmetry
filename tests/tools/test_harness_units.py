"""Enforce the physical-unit dispatch authority."""

from pathlib import Path

import pytest

from tests._harness import _load_harness


def test_existing_unit_comparisons_match_the_frozen_budget():
    assert _load_harness().find_unit_comparison_violations() == []


@pytest.mark.parametrize(
    "source",
    [
        'if units == "ns":\n    pass\n',
        'if unit in {"T", "mT"}:\n    pass\n',
        "if token in _TESLA_UNIT_TOKENS:\n    pass\n",
        'if node.attrs.get("units") == "K":\n    pass\n',
    ],
)
def test_unit_ratchet_rejects_new_dispatch_outside_authority(tmp_path: Path, source):
    path = tmp_path / "src/asymmetry/core/io/example.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    failures = _load_harness().find_unit_comparison_violations(tmp_path, baseline={})
    assert len(failures) == 1
    assert "physical_units" in failures[0].message


def test_unit_ratchet_allows_authority_and_unrelated_comparisons(tmp_path: Path):
    folder = tmp_path / "src/asymmetry/core/io"
    folder.mkdir(parents=True)
    (folder / "physical_units.py").write_text('if units == "ns":\n    pass\n')
    (folder / "example.py").write_text(
        'if mode == "time":\n    pass\n'
        'if units.node("axis") is not None:\n    pass\n'
        "if unit is not None:\n    pass\n"
    )
    assert _load_harness().find_unit_comparison_violations(tmp_path, baseline={}) == []

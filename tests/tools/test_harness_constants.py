"""Structural checks for the physical-constant authority and shrinking exceptions."""

from pathlib import Path

import pytest

from tests._harness import _load_harness


def test_physical_constants_have_one_authority() -> None:
    assert _load_harness().find_physical_constant_duplicate_violations() == []


@pytest.mark.parametrize(
    ("source", "violates"),
    [
        ("R_GAS = 8.314\n", True),
        ("kb_mev = 0.1\n", True),
        ("self.R_GAS = 8.3\n", True),
        ("x = 2.19703\n", True),
        ("x = 0.0135538\n", True),
        ("x = 8.3\n", False),
        ("x = 1e-7\n", False),
        ('x = "2.197"\n', False),
        ("# 2.197\nx = 0\n", False),
        ("x = 2.19480\n", False),
    ],
)
def test_constant_ratchet_distinguishes_literals_and_unrelated_values(
    tmp_path: Path, source: str, violates: bool
) -> None:
    path = tmp_path / "src/asymmetry/core/example.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    harness = _load_harness()
    assert (
        bool(harness.find_physical_constant_duplicate_violations(tmp_path, baseline={})) is violates
    )


def test_constant_ratchet_exempts_authority_and_independent_tests(tmp_path: Path) -> None:
    for name in ("src/asymmetry/core/utils/constants.py", "tests/core/test_example.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("R_GAS = 8.314\n")
    assert _load_harness().find_physical_constant_duplicate_violations(tmp_path, baseline={}) == []


def test_constant_ratchet_rejects_a_stale_baseline(tmp_path: Path) -> None:
    path = tmp_path / "docs/screenshots/scenarios/corpus/example.py"
    path.parent.mkdir(parents=True)
    path.write_text("x = 1\n")
    failures = _load_harness().find_physical_constant_duplicate_violations(
        tmp_path,
        baseline={"docs/screenshots/scenarios/corpus/example.py": (1, "Published arithmetic")},
    )
    assert len(failures) == 1
    assert "stale" in failures[0].message


def test_constant_ratchet_rejects_growth_inside_a_baseline_file(tmp_path: Path) -> None:
    path = tmp_path / "docs/screenshots/scenarios/corpus/example.py"
    path.parent.mkdir(parents=True)
    path.write_text("x = 2.197\ny = 135.5\n")
    failures = _load_harness().find_physical_constant_duplicate_violations(
        tmp_path,
        baseline={"docs/screenshots/scenarios/corpus/example.py": (1, "Published arithmetic")},
    )
    assert len(failures) == 1
    assert "grew" in failures[0].message

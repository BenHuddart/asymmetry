from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = runpy.run_path(str(ROOT / "tools" / "harness.py"))
RETIRED_LOCATIONS = (
    "docs/plans",
    "docs/porting",
    "docs/studies",
    "docs/audit",
    "docs/investigations",
    "docs/PLANS.md",
)
BASELINE = Path("tools/baselines/retired_docs.txt")


def _write_tree(root: Path, entries: str = "") -> None:
    baseline = root / BASELINE
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(entries, encoding="utf-8")


def _add_file(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("historical content\n", encoding="utf-8")
    return path


def test_current_tree_matches_the_retired_docs_baseline() -> None:
    assert HARNESS["find_retired_doc_violations"](ROOT) == []


@pytest.mark.parametrize("location", RETIRED_LOCATIONS)
def test_new_file_in_each_retired_location_is_rejected(tmp_path: Path, location: str) -> None:
    _write_tree(tmp_path)
    relative = location if location.endswith(".md") else f"{location}/new.md"
    path = _add_file(tmp_path, relative)

    failures = HARNESS["find_retired_doc_violations"](tmp_path)

    assert len(failures) == 1
    assert failures[0].path == path
    assert "GitHub issue" in failures[0].message


def test_new_file_inside_a_frozen_study_directory_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "docs/porting/study/README.md\n")
    _add_file(tmp_path, "docs/porting/study/README.md")
    path = _add_file(tmp_path, "docs/porting/study/new.md")

    failures = HARNESS["find_retired_doc_violations"](tmp_path)

    assert [failure.path for failure in failures] == [path]


def test_deleting_a_file_and_its_baseline_line_passes(tmp_path: Path) -> None:
    _write_tree(tmp_path, "docs/porting/study/README.md\ndocs/studies/other.md\n")
    path = _add_file(tmp_path, "docs/porting/study/README.md")
    _add_file(tmp_path, "docs/studies/other.md")
    assert HARNESS["find_retired_doc_violations"](tmp_path) == []

    path.unlink()
    _write_tree(tmp_path, "docs/studies/other.md\n")

    assert HARNESS["find_retired_doc_violations"](tmp_path) == []


def test_deleting_every_retired_file_with_an_empty_baseline_passes(tmp_path: Path) -> None:
    _write_tree(tmp_path, "docs/PLANS.md\n")
    path = _add_file(tmp_path, "docs/PLANS.md")
    path.unlink()
    _write_tree(tmp_path)

    assert HARNESS["find_retired_doc_violations"](tmp_path) == []


def test_deleted_file_left_in_baseline_is_reported_with_its_line(tmp_path: Path) -> None:
    _write_tree(tmp_path, "docs/PLANS.md\n")

    failures = HARNESS["find_retired_doc_violations"](tmp_path)

    assert len(failures) == 1
    assert failures[0].path == tmp_path / BASELINE
    assert failures[0].line == 1
    assert "delete line 1" in failures[0].message.lower()


def test_reintroduced_file_after_its_line_was_removed_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path)
    path = _add_file(tmp_path, "docs/porting/retired.md")

    failures = HARNESS["find_retired_doc_violations"](tmp_path)

    assert [failure.path for failure in failures] == [path]


@pytest.mark.parametrize(
    ("entries", "line"),
    [
        ("\n", 1),
        (" docs/plans/a.md\n", 1),
        ("docs/plans/a.md \n", 1),
        ("/docs/plans/a.md\n", 1),
        ("docs/plans/../a.md\n", 1),
        ("docs/reference/a.rst\n", 1),
        ("docs/plans/a.md\ndocs/plans/a.md\n", 2),
        ("docs/plans/b.md\ndocs/plans/a.md\n", 2),
        ("docs/plans//a.md\n", 1),
        ("docs/plans/./a.md\n", 1),
        ("docs/plans/a\\b.md\n", 1),
    ],
)
def test_malformed_baseline_entries_are_reported(tmp_path: Path, entries: str, line: int) -> None:
    _write_tree(tmp_path, entries)

    failures = HARNESS["find_retired_doc_violations"](tmp_path)

    assert any(
        failure.path == tmp_path / BASELINE
        and failure.line == line
        and "malformed" in failure.message.lower()
        for failure in failures
    )


def test_hidden_files_and_pycache_are_ignored(tmp_path: Path) -> None:
    _write_tree(tmp_path)
    for relative in (
        "docs/porting/.DS_Store",
        "docs/porting/.hidden/cache.md",
        "docs/porting/__pycache__/module.pyc",
    ):
        _add_file(tmp_path, relative)

    assert HARNESS["find_retired_doc_violations"](tmp_path) == []


def test_missing_baseline_is_reported(tmp_path: Path) -> None:
    failures = HARNESS["find_retired_doc_violations"](tmp_path)

    assert len(failures) == 1
    assert failures[0].path == tmp_path / BASELINE
    assert "Missing" in failures[0].message


def test_required_knowledge_files_avoid_retired_locations() -> None:
    for relative in HARNESS["REQUIRED_KNOWLEDGE_FILES"]:
        assert not any(
            relative == location or relative.startswith(f"{location}/")
            for location in RETIRED_LOCATIONS
        )

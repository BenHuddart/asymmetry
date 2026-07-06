"""Tests for tools/release_prep.py (version bump + changelog roll)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import release_prep  # noqa: E402
from release_prep import (  # noqa: E402
    ReleasePrepError,
    bump_pyproject,
    compute_new_version,
    extract_unreleased_body,
    read_current_version,
    roll_changelog,
)

PYPROJECT = """\
[project]
name = "asymmetry"
version = "0.5.0"

[tool.ruff]
target-version = "py310"
"""

CHANGELOG = """\
# Changelog

## [Unreleased]

### Added

- **A new feature.** It does things.

### Fixed

- A bug was fixed.

## [0.5.0] - 2026-06-21

### Added

- Old released content.
"""


class TestVersionMath:
    def test_read_current_version(self):
        assert read_current_version(PYPROJECT) == (0, 5, 0)

    def test_read_rejects_missing_version(self):
        with pytest.raises(ReleasePrepError, match="exactly one"):
            read_current_version("[project]\nname = 'x'\n")

    def test_read_ignores_target_version(self):
        # target-version = "py310" must not count as a version line.
        assert read_current_version(PYPROJECT) == (0, 5, 0)

    @pytest.mark.parametrize(
        ("bump", "expected"),
        [("major", (1, 0, 0)), ("minor", (0, 6, 0)), ("patch", (0, 5, 1))],
    )
    def test_bump_kinds(self, bump, expected):
        assert compute_new_version((0, 5, 0), bump, None) == expected

    def test_explicit_version(self):
        assert compute_new_version((0, 5, 0), None, "0.7.2") == (0, 7, 2)

    @pytest.mark.parametrize("bad", ["0.5.0", "0.4.9", "0.5"])
    def test_explicit_version_must_increase_and_parse(self, bad):
        with pytest.raises(ReleasePrepError):
            compute_new_version((0, 5, 0), None, bad)

    def test_exactly_one_of_bump_and_version(self):
        with pytest.raises(ReleasePrepError, match="exactly one"):
            compute_new_version((0, 5, 0), None, None)
        with pytest.raises(ReleasePrepError, match="exactly one"):
            compute_new_version((0, 5, 0), "minor", "0.6.0")


class TestChangelogRoll:
    def test_extract_unreleased_body(self):
        body = extract_unreleased_body(CHANGELOG)
        assert body.startswith("### Added")
        assert "A bug was fixed." in body
        assert "Old released content" not in body

    def test_roll_inserts_heading_and_keeps_unreleased_empty(self):
        rolled = roll_changelog(CHANGELOG, "0.6.0", "2026-07-06")
        lines = rolled.splitlines()
        i = lines.index("## [Unreleased]")
        assert lines[i + 1] == ""
        assert lines[i + 2] == "## [0.6.0] - 2026-07-06"
        # The previous body now belongs to 0.6.0.
        assert extract_unreleased_body(rolled) == ""
        # Released history is untouched.
        assert "## [0.5.0] - 2026-06-21" in rolled
        assert "Old released content." in rolled

    def test_roll_rejects_empty_unreleased(self):
        empty = CHANGELOG.replace(
            "### Added\n\n- **A new feature.** It does things.\n\n"
            "### Fixed\n\n- A bug was fixed.\n\n",
            "",
        )
        assert extract_unreleased_body(empty) == ""
        with pytest.raises(ReleasePrepError, match="empty"):
            roll_changelog(empty, "0.6.0", "2026-07-06")

    def test_roll_rejects_existing_heading(self):
        with pytest.raises(ReleasePrepError, match="already contains"):
            roll_changelog(CHANGELOG, "0.5.0", "2026-07-06")

    def test_bump_pyproject(self):
        bumped = bump_pyproject(PYPROJECT, "0.6.0")
        assert 'version = "0.6.0"' in bumped
        assert 'version = "0.5.0"' not in bumped
        assert 'target-version = "py310"' in bumped


class TestMain:
    @pytest.fixture()
    def repo(self, tmp_path: Path) -> Path:
        (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
        (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
        return tmp_path

    def test_end_to_end_bump(self, repo: Path, capsys, monkeypatch):
        output_file = repo / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        rc = release_prep.main(["--bump", "minor", "--date", "2026-07-06", "--root", str(repo)])
        assert rc == 0
        assert 'version = "0.6.0"' in (repo / "pyproject.toml").read_text()
        changelog = (repo / "CHANGELOG.md").read_text()
        assert "## [0.6.0] - 2026-07-06" in changelog
        out = capsys.readouterr().out
        assert "0.5.0 -> 0.6.0" in out
        assert "A new feature" in out  # promoted section echoed for CI summary
        assert output_file.read_text() == "version=0.6.0\ntag=v0.6.0\n"

    def test_dry_run_writes_nothing(self, repo: Path, capsys):
        rc = release_prep.main(
            ["--bump", "patch", "--date", "2026-07-06", "--root", str(repo), "--dry-run"]
        )
        assert rc == 0
        assert (repo / "pyproject.toml").read_text() == PYPROJECT
        assert (repo / "CHANGELOG.md").read_text() == CHANGELOG
        assert "would release 0.5.0 -> 0.5.1" in capsys.readouterr().out

    def test_validation_failure_is_clean_error(self, repo: Path, capsys):
        (repo / "CHANGELOG.md").write_text(
            "# Changelog\n\n## [Unreleased]\n\n## [0.5.0] - 2026-06-21\n\n- Old.\n",
            encoding="utf-8",
        )
        rc = release_prep.main(["--bump", "minor", "--root", str(repo)])
        assert rc == 1
        assert "empty" in capsys.readouterr().err
        assert (repo / "pyproject.toml").read_text() == PYPROJECT


class TestRealRepoFiles:
    """The script's assumptions hold against the actual repo files."""

    ROOT = Path(__file__).resolve().parents[2]

    def test_pyproject_has_exactly_one_version_line(self):
        read_current_version((self.ROOT / "pyproject.toml").read_text())

    def test_changelog_unreleased_section_parses(self):
        extract_unreleased_body((self.ROOT / "CHANGELOG.md").read_text())

    def test_dry_run_against_real_repo(self, capsys):
        # [Unreleased] is legitimately empty right after a release commit, and
        # this test runs on that commit's CI - both outcomes are correct
        # behavior, so pin whichever applies to the current tree.
        body = extract_unreleased_body((self.ROOT / "CHANGELOG.md").read_text())
        rc = release_prep.main(["--bump", "minor", "--root", str(self.ROOT), "--dry-run"])
        if body.strip():
            assert rc == 0
        else:
            assert rc == 1
            assert "empty" in capsys.readouterr().err

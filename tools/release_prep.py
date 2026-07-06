"""Prepare a release: bump the project version and roll the changelog.

This is the mechanical half of cutting a release (see RELEASING.md for the
process and the agent guidelines around it). It:

1. reads the current version from ``pyproject.toml``;
2. computes the new version from ``--bump {major,minor,patch}`` or takes an
   explicit ``--version``, requiring it to be strictly greater;
3. validates that ``CHANGELOG.md`` has a non-empty ``[Unreleased]`` section
   and no heading for the new version yet;
4. rewrites ``pyproject.toml`` and inserts ``## [X.Y.Z] - YYYY-MM-DD``
   directly below ``## [Unreleased]`` (the roll style used by every previous
   release), leaving a fresh empty ``[Unreleased]`` above it.

It never touches git: committing, tagging, and pushing belong to the
``cut-release`` workflow (or a human) so the script stays trivially testable.

Usage:
    python tools/release_prep.py --bump minor [--dry-run]
    python tools/release_prep.py --version 0.6.0 --date 2026-07-06

With ``--dry-run`` nothing is written; the plan is still printed. The promoted
changelog section is always echoed to stdout so a CI step can surface it in
the run summary. When ``GITHUB_OUTPUT`` is set, ``version=`` and ``tag=`` are
appended to it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from pathlib import Path

_VERSION_RE = re.compile(r"^version = \"(\d+)\.(\d+)\.(\d+)\"$", re.MULTILINE)
_UNRELEASED_HEADING = "## [Unreleased]"


class ReleasePrepError(RuntimeError):
    """A validation failure that should abort the release preparation."""


def read_current_version(pyproject_text: str) -> tuple[int, int, int]:
    matches = _VERSION_RE.findall(pyproject_text)
    if len(matches) != 1:
        raise ReleasePrepError(
            f"expected exactly one 'version = \"X.Y.Z\"' line in pyproject.toml, "
            f"found {len(matches)}"
        )
    return tuple(int(part) for part in matches[0])  # type: ignore[return-value]


def compute_new_version(
    current: tuple[int, int, int],
    bump: str | None,
    explicit: str | None,
) -> tuple[int, int, int]:
    if (bump is None) == (explicit is None):
        raise ReleasePrepError("pass exactly one of --bump or --version")
    if explicit is not None:
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", explicit)
        if not m:
            raise ReleasePrepError(f"--version must be X.Y.Z, got {explicit!r}")
        new = tuple(int(g) for g in m.groups())
        if new <= current:
            raise ReleasePrepError(
                f"--version {explicit} is not greater than the current "
                f"version {'.'.join(map(str, current))}"
            )
        return new  # type: ignore[return-value]
    major, minor, patch = current
    if bump == "major":
        return (major + 1, 0, 0)
    if bump == "minor":
        return (major, minor + 1, 0)
    if bump == "patch":
        return (major, minor, patch + 1)
    raise ReleasePrepError(f"unknown bump kind {bump!r}")


def extract_unreleased_body(changelog_text: str) -> str:
    """Return the body of the [Unreleased] section (heading excluded)."""
    lines = changelog_text.splitlines()
    try:
        start = lines.index(_UNRELEASED_HEADING)
    except ValueError:
        raise ReleasePrepError(f"CHANGELOG.md has no '{_UNRELEASED_HEADING}' heading") from None
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end]).strip("\n")


def roll_changelog(changelog_text: str, version: str, date: str) -> str:
    body = extract_unreleased_body(changelog_text)
    if not body.strip():
        raise ReleasePrepError(
            "the [Unreleased] section of CHANGELOG.md is empty - nothing to release"
        )
    new_heading = f"## [{version}] - {date}"
    if re.search(rf"^## \[{re.escape(version)}\]", changelog_text, re.MULTILINE):
        raise ReleasePrepError(f"CHANGELOG.md already contains a heading for version {version}")
    # Insert the release heading directly below "## [Unreleased]" (plus the
    # blank line that follows it), so the accumulated body now belongs to the
    # released version and [Unreleased] is left empty - the same roll shape as
    # every previous release commit.
    marker = _UNRELEASED_HEADING + "\n\n"
    if marker not in changelog_text:
        raise ReleasePrepError(f"expected '{_UNRELEASED_HEADING}' to be followed by a blank line")
    return changelog_text.replace(marker, marker + new_heading + "\n\n", 1)


def bump_pyproject(pyproject_text: str, version: str) -> str:
    return _VERSION_RE.sub(f'version = "{version}"', pyproject_text, count=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bump", choices=["major", "minor", "patch"])
    parser.add_argument("--version", help="explicit new version (X.Y.Z)")
    parser.add_argument(
        "--date",
        default=_dt.date.today().isoformat(),
        help="release date for the changelog heading (default: today)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing pyproject.toml and CHANGELOG.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without writing anything",
    )
    args = parser.parse_args(argv)

    pyproject_path = args.root / "pyproject.toml"
    changelog_path = args.root / "CHANGELOG.md"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    changelog_text = changelog_path.read_text(encoding="utf-8")

    try:
        current = read_current_version(pyproject_text)
        new = compute_new_version(current, args.bump, args.version)
        version = ".".join(map(str, new))
        promoted = extract_unreleased_body(changelog_text)
        new_changelog = roll_changelog(changelog_text, version, args.date)
        new_pyproject = bump_pyproject(pyproject_text, version)
    except ReleasePrepError as exc:
        print(f"release-prep: error: {exc}", file=sys.stderr)
        return 1

    current_str = ".".join(map(str, current))
    action = "would release" if args.dry_run else "releasing"
    print(f"release-prep: {action} {current_str} -> {version} ({args.date})")
    print()
    print(f"## [{version}] - {args.date}")
    print()
    print(promoted)

    if not args.dry_run:
        pyproject_path.write_text(new_pyproject, encoding="utf-8")
        changelog_path.write_text(new_changelog, encoding="utf-8")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"version={version}\n")
            fh.write(f"tag=v{version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

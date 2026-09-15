#!/usr/bin/env python3
"""Render ``references/commands.md`` for the packaged ``asymmetry-analysis`` skill.

The skill's command reference is the CLI's own ``--help`` text, one section per
subcommand, so an agent reading the skill never has to guess a flag spelling.
Generating it keeps it honest: :func:`asymmetry.cli.build_parser` is the single
source, and ``tests/tools/test_skill_package.py`` re-renders and compares, so a
new flag that never reaches the reference fails the suite.

Usage::

    python tools/agent_eval/render_command_reference.py          # write the file
    python tools/agent_eval/render_command_reference.py --check   # compare only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from asymmetry.cli import build_parser  # noqa: E402  (after the sys.path fix above)

#: Where the rendered reference lives, relative to the repository root.
REFERENCE_PATH = Path("src/asymmetry/resources/skills/asymmetry-analysis/references/commands.md")

#: Terminal width argparse renders against, pinned so the output does not
#: depend on the terminal the generator happened to run in.
WIDTH = 88

_HEADER = """\
# `asymmetry` command reference

Generated from the CLI's own parser by
`tools/agent_eval/render_command_reference.py`; do not edit by hand.

Every command takes `--json` (machine-readable payload on stdout).

`survey`, `reduce`, `wizard` and `fit-series` persist state in the work
directory `./asymmetry-work` — in the directory you run the command from, not
in the data folder — so the next command picks it up. `fit` and `trend` read
that state and add only what `--plot` (and `trend --csv`) asks for. `alpha`
and `info` are stateless — they load, print and write nothing — and `skill`
writes into the agent's own skill directory instead.

One work directory holds one data folder's session. For a second folder in the
same project, pass `--workdir asymmetry-work-<short-name>` and keep using it
for that folder's commands.

Exit codes: 0 success, 1 user error (one line on stderr), 2 internal error
(a traceback).
"""


def _subparser_actions(parser: argparse.ArgumentParser) -> list[argparse._SubParsersAction]:
    return [
        action
        for action in parser._actions  # noqa: SLF001 — argparse exposes no public walk
        if isinstance(action, argparse._SubParsersAction)
    ]


def _walk(parser: argparse.ArgumentParser, prefix: str) -> list[tuple[str, str]]:
    """``(command path, help text)`` for *parser* and every subcommand under it."""
    sections = [(prefix, parser.format_help())]
    for action in _subparser_actions(parser):
        for name, subparser in action.choices.items():
            sections.extend(_walk(subparser, f"{prefix} {name}"))
    return sections


def render() -> str:
    """The full reference text."""
    os.environ["COLUMNS"] = str(WIDTH)
    sections = _walk(build_parser(), "asymmetry")
    parts = [_HEADER]
    for path, help_text in sections:
        parts.append(f"\n## `{path}`\n\n```\n{help_text.rstrip()}\n```\n")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Write (or, with ``--check``, verify) the reference. Returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the file on disk differs from a fresh render, writing nothing",
    )
    args = parser.parse_args(argv)

    target = REPO_ROOT / REFERENCE_PATH
    rendered = render()
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current == rendered:
            print(f"{REFERENCE_PATH} is current")
            return 0
        print(
            f"{REFERENCE_PATH} is stale; rerun "
            "'python tools/agent_eval/render_command_reference.py'",
            file=sys.stderr,
        )
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

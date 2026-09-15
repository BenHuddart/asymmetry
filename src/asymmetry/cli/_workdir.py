"""Where a command's work directory comes from — declared and resolved once.

``--workdir`` and its default are the same on every command that has one, and
so is the rule that one work directory holds one data folder. Both live here
rather than in each command module, so a new command cannot spell the default
differently or reach the work directory without the binding check.

The name is spelled here as well as in
:mod:`asymmetry.core.workflow.workdir` because the help text is built while
the parser is (on *every* invocation, ``--help`` included) and importing the
workflow package costs a couple of seconds; the two are pinned together by
``tests/core/test_cli_commands.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from asymmetry.cli._output import UserError

if TYPE_CHECKING:
    from asymmetry.core.workflow.workdir import WorkDir

#: The default work directory's name, relative to the current directory.
WORKDIR_NAME = "asymmetry-work"


def add_workdir_argument(parser: argparse.ArgumentParser, *, purpose: str) -> None:
    """Declare ``--workdir`` on *parser*, with the default every command shares."""
    parser.add_argument(
        "--workdir",
        default=None,
        help=f"Work directory to {purpose} (default: ./{WORKDIR_NAME})",
    )


def workdir_for(folder: str | Path, root: str | Path | None) -> WorkDir:
    """The work directory for *folder*: ``./asymmetry-work``, or *root* if given.

    Raises :class:`UserError` when the directory already holds a different
    data folder's session — the two would otherwise overwrite each other's
    spectra, recipes and series, which are keyed on run number alone.
    """
    from asymmetry.core.workflow.workdir import WorkDir, WorkDirMismatchError

    try:
        return WorkDir.default(root).bind(folder)
    except WorkDirMismatchError as exc:
        raise UserError(str(exc)) from None


__all__ = ["WORKDIR_NAME", "add_workdir_argument", "workdir_for"]

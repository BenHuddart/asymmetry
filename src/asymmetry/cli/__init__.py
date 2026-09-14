"""Command-line interface for Asymmetry.

``asymmetry <command>`` drives the scriptable façade in
:mod:`asymmetry.core.workflow`, in the order an analysis runs: ``survey`` a
folder of runs, measure ``alpha`` on a calibration run, ``reduce`` runs to
asymmetry, screen one of them with the fit ``wizard``, ``fit`` a run or a whole
scan with ``fit-series``, and read the ``trend`` out of the result — plus
``skill`` to install this workflow as an agent skill, and the original
``info`` file summary. Every command takes ``--json`` and writes its state
into the work directory ``<folder>/.asymmetry`` so the next command can pick
it up; ``reduce``, ``wizard``, ``fit``, ``fit-series`` and ``trend`` also take
``--plot`` to write headless PNGs alongside it.

Exit codes: 0 on success, 1 on a user error (one line on stderr), 2 on an
internal error (a traceback, because that is a bug worth reporting).

``--verbose`` (on the main parser, before the subcommand) shows every warning
as Python's own traceback-style block; without it, a repeated warning
(the fit wizard's ``AsymmetryScaleWarning`` and others) prints once, not once
per occurrence — see :func:`_collapse_repeated_warnings`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import traceback
import warnings

from asymmetry import __version__
from asymmetry.cli._output import UserError
from asymmetry.cli.commands import alpha as alpha_command
from asymmetry.cli.commands import fit as fit_command
from asymmetry.cli.commands import fit_series as fit_series_command
from asymmetry.cli.commands import info as info_command
from asymmetry.cli.commands import reduce as reduce_command
from asymmetry.cli.commands import skill as skill_command
from asymmetry.cli.commands import survey as survey_command
from asymmetry.cli.commands import trend as trend_command
from asymmetry.cli.commands import wizard as wizard_command

#: Subcommand modules, in the order they appear in ``--help``: the workflow in
#: the order an analysis runs, then the skill installer, then the standalone
#: file inspector.
_COMMANDS = (
    survey_command,
    alpha_command,
    reduce_command,
    wizard_command,
    fit_command,
    fit_series_command,
    trend_command,
    skill_command,
    info_command,
)


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser, with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog="asymmetry",
        description="Asymmetry — μSR data analysis",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Show every warning as Python's own traceback-style block, instead "
            "of collapsing repeats of the same warning to one line on stderr"
        ),
    )

    subparsers = parser.add_subparsers(dest="command")
    for module in _COMMANDS:
        module.add_parser(subparsers)
    return parser


def _collapse_repeated_warnings() -> None:
    """Print each distinct (category, message) warning once, not a traceback block.

    An agent reads stderr; a candidate-screening run can raise the same
    ``AsymmetryScaleWarning`` (or another warning) many times over, and
    Python's default ``showwarning`` prints a multi-line ``file:line:
    Category: message`` block for every one of them. This keeps the
    substance — one line per distinct warning — and drops the repeats.
    """
    seen: set[tuple[type, str]] = set()

    def _show_warning(message, category, filename, lineno, file=None, line=None):
        key = (category, str(message))
        if key in seen:
            return
        seen.add(key)
        print(f"asymmetry: warning: {category.__name__}: {message}", file=sys.stderr)

    warnings.showwarning = _show_warning


def main(argv: list[str] | None = None) -> None:
    """Parse *argv* and run the named subcommand.

    Raises :class:`SystemExit` with code 1 for a user error and 2 for an
    internal one; returns normally (exit code 0) on success, so both the
    ``asymmetry`` console script and ``python -m asymmetry`` behave the same.
    """
    mp.freeze_support()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if not args.verbose:
        _collapse_repeated_warnings()

    # The CLI's error boundary: the two failure modes the command contract
    # promises. A UserError is something the caller can fix, so it prints one
    # line; anything else is a bug, so it prints the traceback and exits 2
    # rather than pretending the run succeeded.
    try:
        args.func(args)
    except UserError as exc:
        print(f"asymmetry: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception:
        traceback.print_exc()
        raise SystemExit(2) from None


if __name__ == "__main__":
    mp.freeze_support()
    main()


__all__ = ["__version__", "build_parser", "main"]

"""Command-line interface for Asymmetry.

``asymmetry <command>`` drives the scriptable façade in
:mod:`asymmetry.core.workflow`: ``survey`` a folder of runs, measure ``alpha``
on a calibration run, ``reduce`` runs to asymmetry — plus the original ``info``
file summary. Every command takes ``--json`` and writes its state into the
work directory ``<folder>/.asymmetry`` so the next command can pick it up.

Exit codes: 0 on success, 1 on a user error (one line on stderr), 2 on an
internal error (a traceback, because that is a bug worth reporting).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import traceback

from asymmetry import __version__
from asymmetry.cli._output import UserError
from asymmetry.cli.commands import alpha as alpha_command
from asymmetry.cli.commands import info as info_command
from asymmetry.cli.commands import reduce as reduce_command
from asymmetry.cli.commands import survey as survey_command

#: Subcommand modules, in the order they appear in ``--help``: the workflow in
#: the order an analysis runs, then the standalone file inspector.
_COMMANDS = (survey_command, alpha_command, reduce_command, info_command)


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser, with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog="asymmetry",
        description="Asymmetry — μSR data analysis",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    for module in _COMMANDS:
        module.add_parser(subparsers)
    return parser


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

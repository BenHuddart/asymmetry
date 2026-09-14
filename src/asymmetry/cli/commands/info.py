"""``asymmetry info`` — print a loaded file's metadata summary."""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``info`` subcommand."""
    parser = subparsers.add_parser("info", help="Show metadata for a data file")
    parser.add_argument("file", help="Path to a μSR data file")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Load the file and print its summary."""
    from asymmetry.core.io import load

    run_result = load(args.file)
    print(run_result.summary())

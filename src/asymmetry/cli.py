"""Command-line interface for Asymmetry."""

import argparse
import multiprocessing as mp

from asymmetry import __version__


def main(argv: list[str] | None = None) -> None:
    mp.freeze_support()

    parser = argparse.ArgumentParser(
        prog="asymmetry",
        description="Asymmetry — μSR data analysis",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # --- info command ---
    info_parser = sub.add_parser("info", help="Show metadata for a data file")
    info_parser.add_argument("file", help="Path to a μSR data file (.wim, etc.)")

    args = parser.parse_args(argv)

    if args.command == "info":
        from asymmetry.core.io import load

        run = load(args.file)
        print(run.summary())
    else:
        parser.print_help()


if __name__ == "__main__":
    mp.freeze_support()
    main()

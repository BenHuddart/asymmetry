"""``asymmetry audit`` — which numbers in a draft summary no command printed."""

from __future__ import annotations

import argparse
from pathlib import Path

from asymmetry.cli._numbers import unsupported_laws, unverified_numbers
from asymmetry.cli._output import UserError, emit_json, payload
from asymmetry.cli._workdir import OUTPUT_LOG, WORKDIR_NAME


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``audit`` subcommand."""
    parser = subparsers.add_parser(
        "audit",
        help="List the numbers in a draft summary that no command's output printed",
    )
    parser.add_argument("draft", help="The draft summary (a text or Markdown file)")
    parser.add_argument(
        "--workdir",
        action="append",
        default=None,
        help=(
            f"Work directory whose {OUTPUT_LOG} to check against (repeatable; default: "
            f"every ./{WORKDIR_NAME}* directory here)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Print each unverified number with the line it sits on."""
    draft = Path(args.draft)
    if not draft.is_file():
        raise UserError(f"No draft at {draft}.")
    roots = (
        [Path(root) for root in args.workdir]
        if args.workdir
        else sorted(Path.cwd().glob(f"{WORKDIR_NAME}*"))
    )
    logs = [root / OUTPUT_LOG for root in roots if (root / OUTPUT_LOG).is_file()]
    if not logs:
        raise UserError(
            f"No {OUTPUT_LOG} in {', '.join(str(root) for root in roots) or 'this directory'}; "
            f"run the analysis commands from this directory first."
        )
    log_text = "\n".join(log.read_text(encoding="utf-8") for log in logs)
    text = draft.read_text(encoding="utf-8")
    found = unverified_numbers(text, log_text)
    laws = unsupported_laws(text, log_text)

    if args.json:
        emit_json(
            payload(
                logs=[str(log) for log in logs],
                unsupported_laws=[{"law": law, "phrase": phrase} for law, phrase in laws],
                unverified=[
                    {"text": entry.text, "line_number": entry.line_number, "line": entry.line}
                    for entry in found
                ],
            )
        )
        return
    for law, phrase in laws:
        print(
            f"The draft says {phrase!r}, but every {law} fit this session printed LAW NOT "
            f"ESTABLISHED: describe that trend in plain words instead."
        )
    if not found and not laws:
        print(
            f"No unprinted numbers found in {draft}. Now send its text as your whole final "
            f"message, starting at its title — the user sees neither this output nor the "
            f"file, and the reply says nothing about this check."
        )
        return
    if not found:
        return
    print(
        f"{len(found)} number(s) in {draft} appear in no logged command output — "
        f"arithmetic, a conversion, or a value from memory. Remove each, quote the printed "
        f"value instead, or say the relation in words:"
    )
    for entry in found:
        print(f"  line {entry.line_number}: {entry.text!r} in: {entry.line}")


__all__ = ["add_parser", "run"]

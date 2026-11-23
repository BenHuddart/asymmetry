"""``asymmetry info`` — print a loaded file's metadata summary.

Stateless: it loads the file, prints what it found, and writes nothing.
"""

from __future__ import annotations

import argparse
from typing import Any

from asymmetry.cli._output import UserError, emit_json, payload, render_table

#: Metadata key held back from ``--json``: the loader's verbatim copy of the
#: NeXus tree, which is thousands of nodes and would bury the run's own
#: metadata in the payload. ``summary`` and the human output never showed it
#: either.
_BULKY_METADATA_KEY = "nexus_fields"


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``info`` subcommand."""
    parser = subparsers.add_parser("info", help="Show metadata for a data file")
    parser.add_argument("file", help="Path to a μSR data file")
    parser.add_argument(
        "--units", action="store_true", help="Show original unit declarations and interpretations"
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Load the file and print its summary."""
    from asymmetry.core.io import load, unit_report
    from asymmetry.core.transform.grouping import group_names

    try:
        run_result = load(args.file)
    except ValueError as exc:
        # The file boundary: a format no loader reads is the user's to hear about.
        raise UserError(str(exc)) from None
    # The names `--pair` takes, and the pair the file reduces on by default.
    groups = group_names(run_result.run)
    grouping = run_result.run.grouping
    pair = (
        (int(grouping["forward_group"]), int(grouping["backward_group"]))
        if "forward_group" in grouping and "backward_group" in grouping
        else None
    )
    units = unit_report(run_result) if args.units else None
    if args.units and not args.json:
        if units:
            print(
                render_table(
                    ["Quantity", "Source", "Declaration and interpretation"],
                    [[record.quantity, record.path, record.describe()] for record in units],
                )
            )
        if not units:
            print("No NeXus unit-report information available.")
        return
    if not args.json:
        print(run_result.summary())
        print(
            "  Groups      : "
            + ", ".join(f"{gid} {name}" for gid, name in groups.items())
            + (
                f"  (default pair {groups[pair[0]]}/{groups[pair[1]]})"
                if pair is not None
                else "  (independent pair unavailable)"
            )
        )
        return

    emit_json(
        payload(
            file=str(args.file),
            run_number=int(run_result.run_number),
            n_points=int(run_result.n_points),
            groups={str(gid): name for gid, name in groups.items()},
            default_pair=list(pair) if pair is not None else None,
            metadata=_metadata(run_result.metadata),
            summary=run_result.summary(),
            **({"units": [record.to_dict() for record in units]} if units is not None else {}),
        )
    )


def _metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """The run's metadata, JSON-safe and without the bulky NeXus field tree."""
    from asymmetry.core.workflow.jsonio import json_safe

    return {key: json_safe(value) for key, value in metadata.items() if key != _BULKY_METADATA_KEY}

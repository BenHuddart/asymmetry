"""Shared CLI plumbing: the error vocabulary, JSON payloads and table rendering.

Exit codes (the contract every subcommand keeps):

* ``0`` — success.
* ``1`` — a user error: a one-line message on stderr, no traceback. Raise
  :class:`UserError` from a command to produce one.
* ``2`` — an internal error: the traceback, because the user has found a bug.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from asymmetry import __version__

#: Schema version stamped into every machine-readable payload.
SCHEMA = 1


class UserError(Exception):
    """A problem with what the user asked for, not with the code.

    Raised by a command when the arguments name a folder that does not exist,
    a run that is not there, an unparsable run spec and so on. The dispatcher
    turns it into a one-line stderr message and exit code 1.
    """


def payload(**fields: Any) -> dict[str, Any]:
    """A machine-readable payload carrying the schema and version stamps."""
    return {"schema": SCHEMA, "asymmetry_version": __version__, **fields}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def emit_json(data: dict[str, Any]) -> None:
    """Print a payload as indented JSON on stdout."""
    print(json.dumps(data, indent=2, default=_json_default))


def format_number(value: float | None, digits: int = 3) -> str:
    """A fixed-width-friendly rendering of an optional number (``"-"`` when absent)."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """A compact column-aligned table: header, rule, then one line per row."""
    widths = [len(head) for head in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(head.ljust(widths[i]) for i, head in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


__all__ = [
    "SCHEMA",
    "UserError",
    "emit_json",
    "format_number",
    "payload",
    "render_table",
]

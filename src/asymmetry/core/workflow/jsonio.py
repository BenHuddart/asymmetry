"""The JSON conventions every workflow document and payload keeps.

One rule, and it is the reason this module exists: **what we write is standard
JSON**. Python's ``json`` module happily emits the non-standard ``Infinity`` and
``NaN`` tokens, which it alone reads back — ``jq``, JavaScript and every strict
parser reject them. The payloads here are written to be read by other tools (an
agent piping ``--json`` into ``jq``, a notebook reading the work directory), so
a non-finite number is written as ``null``: "no value", which is what an
infinite bound or an undefined criterion means anyway.

:func:`json_safe` also renders numpy scalars and arrays, which the analysis
layers produce everywhere, and *raises* on anything else — an object with no
JSON rendering is a bug in the caller, not something to stringify silently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Render *value* as standard-JSON data (see the module docstring).

    Raises :class:`TypeError` naming the type of anything it cannot render.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def dumps(payload: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize *payload* as standard JSON."""
    return json.dumps(json_safe(payload), indent=indent, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* as standard JSON, newline-terminated."""
    path.write_text(dumps(payload) + "\n", encoding="utf-8")


__all__ = ["dumps", "json_safe", "write_json"]

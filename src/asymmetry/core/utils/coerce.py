"""Small value-coercion helpers shared across the core analysis layer."""

from __future__ import annotations

import numpy as np


def optional_float(value: object) -> float | None:
    """Coerce *value* to a finite ``float``, or ``None`` if that is not possible.

    Returns ``None`` for ``None``, for anything :func:`float` rejects
    (``TypeError``/``ValueError``), and for non-finite results (``inf``/
    ``nan``) — the only valid inputs for the times, fields, and frequencies
    this parses are finite, so a non-finite value is treated as "unset".

    This is the value/number-domain coercion used when reading serialised
    config. The GUI's text-entry variant ``_parse_optional_float``
    (``gui/panels/maxent_panel.py``) parses raw widget *text* and is a
    separate concern — it stays local to the panel.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

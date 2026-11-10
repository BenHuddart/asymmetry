"""Declared NeXus sequence slots at the file boundary."""

import numpy as np


def period_declaration(entry, read_optional, safe_str, n_periods):
    """Retain sequence declarations without inventing data/dwell correspondence."""
    beam = read_optional(read_optional(entry, "instrument"), "beam")
    types = np.asarray(read_optional(beam, "period_type", default=[]))
    slots = (
        [int(i) for i in np.flatnonzero(types == 1) + 1]
        if types.ndim == 1 and types.dtype.kind in "iuf"
        else []
    )
    matched = bool(types.size and len(slots) == n_periods)
    return {
        "origin": "NeXus:instrument/beam",
        "types": types.tolist(),
        "labels": safe_str(read_optional(beam, "period_labels")),
        "data_dae_slots": slots if matched else [],
        "limitation": ""
        if matched
        else "DAE correspondence unavailable or inconsistent with stored data periods",
    }


def logged_data_periods(entry, read_optional, n_periods):
    """DAE slots for existing period-log selection, with its ordinal fallback."""
    beam = read_optional(read_optional(entry, "instrument"), "beam")
    types = np.asarray(read_optional(beam, "period_type", default=[]), dtype=float)
    return (
        [int(number) for number in np.flatnonzero(types == 1) + 1]
        if types.size
        else list(range(1, n_periods + 1))
    )

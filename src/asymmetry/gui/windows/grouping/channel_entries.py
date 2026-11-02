"""Reversible ordinal offsets for the grouping editor's zero-based draft."""

from numbers import Real


def shift_channel_entries(entries, offset: int) -> list:
    """Preserve invalid entries for editing; only the core validates membership."""
    return [
        value + offset if isinstance(value, Real) and not isinstance(value, bool) else value
        for value in entries
    ]

"""Shared display-formatting helpers for fit parameters."""

from __future__ import annotations

from asymmetry.core.fitting.parameters import get_param_info


def format_param_label(name: str) -> str:
    """Return a display label with Greek symbols and units where applicable."""
    return get_param_info(name).unicode_label()

"""Shared display-formatting helpers for fit parameters."""

from __future__ import annotations

import math

from asymmetry.core.fitting.parameters import get_param_info


def format_param_label(name: str) -> str:
    """Return a display label with Greek symbols and units where applicable."""
    return get_param_info(name).unicode_label()


def format_value_error(value: float, error: float, *, sig_error_digits: int = 2) -> str:
    """Format ``value ± error`` with the value's precision matched to the error.

    The error is shown to ``sig_error_digits`` significant digits and the value
    rounded to the same decimal place — the convention that keeps a quoted
    ``0.4667 ± 0.0016`` from misleadingly carrying more (or fewer) digits than
    the uncertainty supports. Falls back to a bare ``%.5g`` value when the
    error is zero, non-finite, or absurdly large relative to the value; a
    non-finite value renders as ``—``.
    """
    value = float(value)
    error = float(error)
    if not math.isfinite(value):
        return "—"
    if not math.isfinite(error) or error <= 0.0:
        return f"{value:.5g}"
    # Decimal place of the error's last shown significant digit.
    exponent = math.floor(math.log10(error))
    decimals = max(0, sig_error_digits - 1 - exponent)
    if decimals > 12 or exponent > 12:
        # Pathological scale mismatch — matched-precision formatting would
        # produce an unreadable string; fall back to independent rounding.
        return f"{value:.5g} ± {error:.{sig_error_digits}g}"
    return f"{value:.{decimals}f} ± {error:.{decimals}f}"

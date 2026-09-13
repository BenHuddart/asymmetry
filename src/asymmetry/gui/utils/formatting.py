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


def format_value_uncertainty(value: float, error: float | None) -> str:
    """Format ``value`` with its uncertainty in the parenthesised convention.

    The digits in parentheses are the error's significant digits, aligned with
    the value's last shown decimals: ``35.8(5)`` is 35.8 ± 0.5, ``2.85(13)`` is
    2.85 ± 0.13, ``1.24(3)`` is 1.24 ± 0.03. The error keeps two significant
    digits only while its leading digit is 1 or 2 — rounding 0.13 to a single
    digit would throw away a quarter of the uncertainty, while 0.43 → 0.4 costs
    little — and the value is rounded to the error's last decimal place. With
    no usable error (``None``, non-finite, or non-positive) the value alone is
    rendered to four significant digits.
    """
    value = float(value)
    if error is None:
        return f"{value:.4g}"
    error = float(error)
    if not math.isfinite(error) or error <= 0.0:
        return f"{value:.4g}"
    # Read the leading digit and exponent off the scientific form rather than
    # log10: 0.3 / 10**floor(log10(0.3)) is 2.9999…, which reads as a 2.
    mantissa, exponent_text = f"{error:e}".split("e")
    decimals = max(0, (2 if mantissa[0] in "12" else 1) - 1 - int(exponent_text))
    return f"{value:.{decimals}f}({round(error, decimals) * 10**decimals:.0f})"

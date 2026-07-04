"""Shared alpha-estimate display helpers for the grouping/calibration dialogs.

Extracted from the former ``grp_io.py`` (line-based ``.grp`` file
serialization) when the ``.grp`` Load/Save feature was retired: project
persistence (grouping profiles saved in ``.asymp``) and instrument presets
made the WiMDA-style ``.grp`` file redundant, but :data:`ALPHA_METHOD_ITEMS`
and :func:`format_value_with_uncertainty` are still used by the grouping
dialog and the alpha/background calibration dialogs, so they moved here
rather than disappearing with the rest of the module.
"""

from __future__ import annotations

import numpy as np

#: Alpha estimation methods offered by the Estimate control: combo label,
#: grouping-dict key, and a one-line explanation shown as the tooltip.
ALPHA_METHOD_ITEMS = (
    (
        "Diamagnetic (TF)",
        "diamagnetic",
        "Minimise the weighted asymmetry over a transverse-field calibration "
        "run, so A(t) oscillates symmetrically about zero.",
    ),
    (
        "General (LF/ZF)",
        "general",
        "Balance lifetime-corrected counts between early and late times; "
        "works on relaxing LF/ZF data, but needs visible relaxation.",
    ),
    (
        "Count ratio ΣF/ΣB",
        "ratio",
        "Plain count ratio (Mantid AlphaCalc). Transverse-field calibration "
        "runs only — relaxing polarization biases it.",
    ),
)


def format_value_with_uncertainty(value: float, error: float | None) -> str:
    """Format ``1.2345 ± 0.0067`` compactly as ``1.2345(67)``."""
    if error is None or not np.isfinite(error) or error <= 0.0:
        return f"{value:.4f}"
    exponent = int(np.floor(np.log10(error)))
    decimals = max(0, 1 - exponent)
    scaled_error = int(round(error * 10**decimals))
    if scaled_error >= 100:  # rounding pushed it to three digits, e.g. 0.0995
        scaled_error = int(round(scaled_error / 10))
        decimals -= 1
    if decimals < 0:  # uncertainty >= ~100: integer digits on both sides
        return f"{value:.0f}({int(round(error))})"
    return f"{value:.{decimals}f}({scaled_error})"


__all__ = [
    "ALPHA_METHOD_ITEMS",
    "format_value_with_uncertainty",
]

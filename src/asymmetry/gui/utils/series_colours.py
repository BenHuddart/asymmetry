"""Per-run trace colours for a series overlay, graded along the series axis."""

from __future__ import annotations

from collections.abc import Sequence

from asymmetry.gui.styles import tokens

#: Okabe-Ito trace colours cycled when the series axis cannot grade the runs.
_FALLBACK_TRACE_COLOURS = (
    tokens.TRACE_BLUE,
    tokens.TRACE_GREEN,
    tokens.TRACE_ORANGE,
    tokens.TRACE_MAGENTA,
    tokens.TRACE_SKY,
    tokens.TRACE_VERMILLION,
)

#: Viridis sample range — the top end is too light on a white surface, so grade
#: only across the darker/mid band.
_VIRIDIS_LO = 0.10
_VIRIDIS_HI = 0.85


def series_colours(axis_values: Sequence[float | None]) -> list[str]:
    """One hex colour per run: viridis graded by axis value, else Okabe-Ito cycled.

    Grading needs every run to carry a value and the values a non-degenerate range.
    """
    floats = [float(value) for value in axis_values if value is not None]
    if len(floats) == len(axis_values) and floats and max(floats) > min(floats):
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        cmap = colormaps["viridis"]
        lo, hi = min(floats), max(floats)
        span = _VIRIDIS_HI - _VIRIDIS_LO
        return [to_hex(cmap(_VIRIDIS_LO + span * (value - lo) / (hi - lo))) for value in floats]
    return [
        _FALLBACK_TRACE_COLOURS[index % len(_FALLBACK_TRACE_COLOURS)]
        for index in range(len(axis_values))
    ]

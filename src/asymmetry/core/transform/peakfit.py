r"""Sub-bin peak location from a local parabola.

A cursor readout for *reading a peak off a spectrum without fitting it*: given a
sampled curve (an FFT/MaxEnt frequency spectrum, an ALC resonance), the position
and height of a local maximum to better than the bin spacing.  The estimate is
the vertex of the unique parabola through the peak bin and its two neighbours.

This ports the behaviour of WiMDA's ``parabpkextrap`` (``Plot.pas``): a 3-point
parabolic extrapolation whose vertex is reported only when the parabola opens
downward (a genuine maximum).  For exactly three points the least-squares
parabola of the Pascal source reduces to the unique interpolating parabola, so
the closed form here is exact and equivalent.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def parabolic_peak(x3: ArrayLike, y3: ArrayLike) -> tuple[float, float] | None:
    r"""Vertex of the parabola through three points, when it is a maximum.

    Parameters
    ----------
    x3, y3
        Three abscissae and ordinates — typically the peak bin and its immediate
        neighbours.  The abscissae need not be evenly spaced.

    Returns
    -------
    tuple or None
        ``(x_peak, y_peak)`` for the vertex of ``y = a x² + b x + c`` through the
        three points when ``a < 0`` (a maximum) **and** the vertex lies within
        the sampled span ``[min(x), max(x)]``; otherwise ``None``.

    Notes
    -----
    Mirrors WiMDA ``parabpkextrap``'s reject-when-``a ≥ 0`` guard (no downward
    peak → no readout).  The in-span requirement is a deliberate tightening: a
    3-point fit gives a meaningful vertex only between the outer samples, so a
    vertex extrapolated beyond them is rejected rather than reported.
    """
    x = np.asarray(x3, dtype=float)
    y = np.asarray(y3, dtype=float)
    if x.size != 3 or y.size != 3:
        return None
    x0, x1, x2 = (float(v) for v in x)
    y0, y1, y2 = (float(v) for v in y)
    if x0 == x1 or x1 == x2 or x0 == x2:
        return None

    # Divided differences → the interpolating parabola y = a x² + b x + c.
    d01 = (y1 - y0) / (x1 - x0)
    d12 = (y2 - y1) / (x2 - x1)
    a = (d12 - d01) / (x2 - x0)
    if not np.isfinite(a) or a >= 0.0:  # opens upward / flat → no maximum
        return None
    b = d01 - a * (x1 + x0)
    c = y0 - a * x0 * x0 - b * x0

    x_peak = -b / (2.0 * a)
    lo, hi = (x0, x2) if x0 <= x2 else (x2, x0)
    if not np.isfinite(x_peak) or x_peak < lo or x_peak > hi:
        return None
    y_peak = a * x_peak * x_peak + b * x_peak + c
    return float(x_peak), float(y_peak)

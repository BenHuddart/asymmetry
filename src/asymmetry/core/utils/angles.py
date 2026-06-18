"""Angle helpers for periodic (orientation) quantities.

A crystal-rotation scan measures a property as a function of orientation, and
the underlying physics is periodic in the angle — the muon Knight shift, for
example, depends on the dipolar tensor through ``cos²θ``, giving a 180° period.
Folding the measured angles into one period overlays equivalent orientations so
the periodic structure is visible and a periodic model can be fitted.
"""

from __future__ import annotations

import math


def wrap_angle_deg(value_deg: float, period_deg: float = 180.0, origin_deg: float = 0.0) -> float:
    """Fold an angle (degrees) into ``[origin_deg, origin_deg + period_deg)``.

    Returns the value unchanged (as a float) when it is non-finite or the period
    is not positive, so a malformed entry never raises here — callers downstream
    already treat non-finite abscissae as off-axis.
    """
    value = float(value_deg)
    period = float(period_deg)
    if not math.isfinite(value) or not math.isfinite(period) or period <= 0.0:
        return value
    # Python's % folds to [0, period) for a positive modulus, handling negatives.
    folded = (value - float(origin_deg)) % period
    return float(origin_deg) + folded

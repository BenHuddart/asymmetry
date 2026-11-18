"""Analytic Gaussian-width average of static LF KT (Hayano et al., 1979)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import spherical_jn

from asymmetry.core.fitting.strong_collision import EvaluationWorkLimitError

_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(16)
GAUSSIAN_WIDTH_NODE_BUDGET = 1 << 21
_INTERVAL_BLOCK = 1024


def _gaussian_width_intervals(
    lower: NDArray, length: NDArray | float, width_scale: float, phase_scale: float, spread: float
) -> NDArray:
    """Integrate the width-averaged derivative on specified scaled time intervals.

    For X~N(1,spread²), Gaussian tilting gives mean=1/D and variance=spread²/D.
    M2 and M4 average X²exp(-X²s²/2) and X⁴exp(-X²s²/2). The entire factors
    j1(x)/x and sinc(x) avoid inverse-field cancellation in the Hayano derivative.
    """
    coordinate = lower[:, None] + np.asarray(length)[..., None] * (_GAUSS_NODES + 1) / 2
    s = width_scale * coordinate
    phase = phase_scale * coordinate
    denominator = 1 + (spread * s) ** 2
    mean = 1 / denominator
    variance = spread**2 / denominator
    envelope = np.exp(-(s**2) / (2 * denominator)) / np.sqrt(denominator)
    second = envelope * (mean**2 + variance)
    fourth = envelope * (mean**4 + 6 * mean**2 * variance + 3 * variance**2)
    # j1(x)/x=1/3-x²/30+x⁴/840-x⁶/45360+O(x⁸); for x<=0.1 the
    # omitted term is <2.6e-15. Direct j1 underflows for tiny nonzero phases.
    small = phase <= 0.1
    squared = phase[small] ** 2
    bessel = np.empty_like(phase)
    bessel[small] = 1 / 3 + squared * (-1 / 30 + squared * (1 / 840 - squared / 45360))
    bessel[~small] = spherical_jn(1, phase[~small]) / phase[~small]
    derivative = 2 * width_scale * (s**3 * fourth * bessel - s * second * np.sinc(phase / np.pi))
    return np.sum(derivative * _GAUSS_WEIGHTS, axis=-1) * length / 2


def gaussian_width_polarisation(
    time: NDArray, delta: float, omega: float, spread: float
) -> NDArray:
    """Average reflected Gaussian widths at nonnegative times, without interpolation.

    Shared full GL16 panels resolve Delta*t by <=1 and omega*t by <=8; each
    endpoint remainder is integrated directly. Scaling with physical rates avoids
    forming omega/Delta for vanishingly narrow distributions. Total quadrature
    work is checked before panel/index allocation; bounded blocks limit scratch.
    """
    if omega == 0:
        u = delta * time
        denominator = 1 + (spread * u) ** 2
        return 1 / 3 + 2 / 3 * denominator**-1.5 * (1 - u**2 / denominator) * np.exp(
            -(u**2) / (2 * denominator)
        )
    scale = max(delta, omega / 8)
    maximum = float(time.max()) * scale
    # The dimensionless panel count uses physical phase, so tiny Delta never
    # creates an infinite ratio. Every endpoint costs one GL16 remainder rule.
    if not np.isfinite(maximum) or np.floor(maximum) + time.size > GAUSSIAN_WIDTH_NODE_BUDGET / 16:
        raise EvaluationWorkLimitError(
            "Gaussian-broadened KT evaluation exceeds the "
            f"{GAUSSIAN_WIDTH_NODE_BUDGET:,} quadrature nodes work budget. "
            "Shorten the selected time window or reduce the number of requested times."
        )
    panels = int(np.floor(maximum))
    prefix = np.empty(panels + 1)
    prefix[0] = 0
    for start in range(0, panels, _INTERVAL_BLOCK):
        end = min(start + _INTERVAL_BLOCK, panels)
        areas = _gaussian_width_intervals(
            np.arange(start, end), 1.0, delta / scale, omega / scale, spread
        )
        prefix[start + 1 : end + 1] = prefix[start] + np.cumsum(areas)
    output = np.empty(time.size)
    flat = time.ravel()
    for start in range(0, time.size, _INTERVAL_BLOCK):
        end = min(start + _INTERVAL_BLOCK, time.size)
        coordinate = flat[start:end] * scale
        counts = np.floor(coordinate).astype(int)
        remainder = _gaussian_width_intervals(
            counts, coordinate - counts, delta / scale, omega / scale, spread
        )
        output[start:end] = 1 + prefix[counts] + remainder
    return output.reshape(time.shape)

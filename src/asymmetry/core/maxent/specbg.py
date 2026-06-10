"""Zero-frequency lineshape subtraction for ZF/LF field-distribution displays.

In zero/longitudinal field the MaxEnt spectrum is dominated by a strong central
(zero-frequency) feature — the static peak of the field distribution.  Weak
satellite or precession structure sits on its flank and is hard to see.  This
display-time tool subtracts a model of that central peak, a zero-centred
pseudo-Voigt scaled to the measured central amplitude, so the surrounding
lineshape stands out (after WiMDA's ``SpecBG``).

It is **display-only**: it operates on a copy of the plotted spectrum, never the
engine's reconstructed spectrum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Empirical factor relating the edit's Gaussian-width value to the 1/e width of
# the subtracted Gaussian (carried verbatim from WiMDA's SpecBG; magic number).
_GAUSSIAN_WIDTH_FACTOR = 1.201

_MIN_WIDTH = 1.0e-3


def subtract_zero_frequency(
    frequencies: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    *,
    gaussian_width: float,
    lorentzian_width: float,
    lorentzian_fraction: float,
    anchor: float | None = None,
) -> NDArray[np.float64]:
    """Return *spectrum* with a zero-centred pseudo-Voigt central peak removed.

    The subtracted shape is

        Δ(x) = a₀·[ (1−η)·exp(−(x/(w_G·1.201))²) + η/(1 + (x/w_L)²) ],

    where ``x`` is the axis coordinate (the spectrum is centred so the static
    peak is near zero), ``w_G``/``w_L`` are the Gaussian/Lorentzian widths in the
    axis units, ``η`` the Lorentzian fraction (clamped to [0, 1]), and ``a₀`` the
    central amplitude — the spectrum value at the bin nearest zero unless an
    explicit *anchor* is given.  Returns a new array; the input is unchanged.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    values = np.asarray(spectrum, dtype=np.float64).copy()
    if freqs.size == 0 or values.size == 0:
        return values

    fraction = float(np.clip(lorentzian_fraction, 0.0, 1.0))
    gauss = max(float(gaussian_width), _MIN_WIDTH) * _GAUSSIAN_WIDTH_FACTOR
    lorentz = max(float(lorentzian_width), _MIN_WIDTH)

    if anchor is None:
        anchor = float(values[int(np.argmin(np.abs(freqs)))])

    gaussian = np.exp(-((freqs / gauss) ** 2))
    lorentzian = 1.0 / (1.0 + (freqs / lorentz) ** 2)
    background = anchor * ((1.0 - fraction) * gaussian + fraction * lorentzian)
    return values - background

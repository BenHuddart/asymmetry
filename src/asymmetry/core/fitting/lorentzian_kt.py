"""Stable exact zero-field Lorentzian strong-collision polarisation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lorentzian_zf_polarisation(t: NDArray, width: float, nu: float) -> NDArray:
    r"""Evaluate the exact rational renewal transform without large-rate subtraction.

    The static transform is F(p)=1/(3p)+(2/3)p/(p+a)^2. Its renewal transform
    F(s+nu)/(1-nu*F(s+nu)) has one real slow pole and a conjugate pair.
    Bracket the real pole after scaling by min(a,nu); bounded coefficients
    retain the small local-field scale when nu/a is enormous. The paired
    modes use cos and t*sinc, which remain finite when the poles coalesce.
    Inputs width and nu are non-negative inverse microseconds.
    """
    t = np.abs(np.asarray(t, dtype=float))
    if width == 0.0:
        return np.ones_like(t)
    if nu == 0.0:
        u = width * t
        return 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - u) * np.exp(-u)
    from scipy.optimize import brentq

    scale = max(width, nu)
    minimum = min(width, nu)
    ratio = minimum / scale
    constant = (2.0 + 4.0 * ratio) / 3.0 if nu <= width else (4.0 + 2.0 * ratio) / 3.0

    def pole_polynomial(r: float) -> float:
        return (
            (-(ratio**2) * r + 2.0 * ratio * (1.0 + ratio)) * r
            - (1.0 + 10.0 * ratio / 3.0 + ratio**2)
        ) * r + constant

    root = brentq(
        pole_polynomial,
        0.0,
        4.0 / 3.0,
        xtol=np.nextafter(0.0, 1.0),
        rtol=4.0 * np.finfo(float).eps,
    )
    slow = -minimum * root
    if nu <= width:
        frequency = (
            np.sqrt(width)
            * np.sqrt(nu)
            * np.sqrt(4.0 / 3.0 - root + ratio * (-root + 3.0 * root**2 / 4.0))
        )
    else:
        frequency = width * np.sqrt(1.0 / (3.0 * (1.0 - ratio * root)) - (1.0 - root / 2.0) ** 2)
    a = width / scale
    rate = nu / scale
    slow_scaled = slow / scale
    centre = -rate - a - slow_scaled / 2.0
    frequency_scaled = frequency / scale
    initial_slope = -4.0 * a / 3.0
    amplitude = (2.0 * a**2 + centre**2 + frequency_scaled**2 - 2.0 * centre * initial_slope) / (
        (slow_scaled - centre) ** 2 + frequency_scaled**2
    )
    cosine = 1.0 - amplitude
    sine = (initial_slope - amplitude * slow_scaled - cosine * centre) * scale
    output = np.asarray(amplitude * np.exp(slow * t))
    fast = np.exp(-(nu + (width + slow / 2.0)) * t)
    live = fast != 0.0
    phase = frequency * t[live]
    output[live] += fast[live] * (cosine * np.cos(phase) + sine * t[live] * np.sinc(phase / np.pi))
    output[t == 0.0] = 1.0
    return output

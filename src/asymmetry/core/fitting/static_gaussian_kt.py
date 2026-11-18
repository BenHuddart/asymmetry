"""Dimensionless static Gaussian field polarisation (Hayano et al., 1979)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcx


def static_gaussian_polarisation(u: NDArray, ratio: float) -> NDArray:
    """Evaluate G(u, r) for u=|Delta*t| and r=|omega_L/Delta|.

    In weak fields, integrate the Taylor series of the Hayano derivative
    using Gaussian moments. Its even powers of r avoid inverse-field
    cancellation and retain smooth field derivatives at zero.
    Stronger fields use the analytic complex-erfcx Gaussian sine integral.
    The two expressions evaluate the same static Gaussian field average.
    """
    if ratio == 0.0:
        return 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - u**2) * np.exp(-0.5 * u**2)
    if ratio < 1.0:
        # M_k = integral_0^u x^(2k+1) exp(-x²/2) dx obeys
        # M_k = 2k M_(k-1) - u^(2k) exp(-u²/2). The integrated
        # Bessel series is G = 1 - sum_k c_k [M_k + u^(2k+2) exp(-u²/2)],
        # c_k = 2(-r²)^k / [(2k+1)! (2k+3)]. For r<=1 the omitted
        # k>=16 terms sum to <4e-19; the u=12 tail is <1e-28.
        squared = np.minimum(u, 12.0) ** 2
        envelope = np.exp(-0.5 * squared)
        moment = -np.expm1(-0.5 * squared)
        power = squared * envelope
        coefficient = 2.0 / 3.0
        result = 1.0 - coefficient * (moment + power)
        for k in range(1, 16):
            moment = 2 * k * moment - power
            power = power * squared
            coefficient *= -(ratio**2) / (2 * k * (2 * k + 3))
            result -= coefficient * (moment + power)
        return result
    envelope = np.exp(-0.5 * u**2)
    phase = ratio * u
    integral = np.sqrt(np.pi / 2.0) * np.imag(
        erfcx(-1j * ratio / np.sqrt(2.0))
        - envelope * np.exp(1j * phase) * erfcx((u - 1j * ratio) / np.sqrt(2.0))
    )
    inverse_ratio = 1.0 / ratio
    return (
        1.0
        - 2.0 * inverse_ratio**2 * (1.0 - envelope * np.cos(phase))
        + 2.0 * inverse_ratio**3 * integral
    )

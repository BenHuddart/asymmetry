"""Dimensionless static Gaussian field polarisation (Hayano et al., 1979)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcx, spherical_jn

# The Gaussian envelope suppresses the omitted derivative beyond u=12 by <1e-28.
# 48-point Gauss–Legendre resolves that envelope and r*u <= 12 in the weak-field branch.
_ABSCISSA, _WEIGHTS = np.polynomial.legendre.leggauss(48)


def static_gaussian_polarisation(u: NDArray, ratio: float) -> NDArray:
    """Evaluate G(u, r) for u=|Delta*t| and r=|omega_L/Delta|.

    In weak fields, integrate the derivative of the Hayano expression,
    ``G'=2 exp(-u²/2) [u³ j1(r*u)/(r*u) - u sinc(r*u)]``. Both terms have
    finite zero-field limits, avoiding cancellation of inverse powers of r.
    Stronger fields use the analytic complex-erfcx Gaussian sine integral.
    The two expressions evaluate the same static Gaussian field average.
    """
    if ratio == 0.0:
        return 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - u**2) * np.exp(-0.5 * u**2)
    if ratio < 1.0:
        end = np.minimum(u, 12.0)
        x = end[..., None] * (_ABSCISSA + 1.0) / 2.0
        z = ratio * x
        j1_over_z = np.divide(spherical_jn(1, z), z, out=np.full_like(z, 1.0 / 3.0), where=z != 0.0)
        return 1.0 + end * np.sum(
            _WEIGHTS * np.exp(-0.5 * x**2) * (x**3 * j1_over_z - x * np.sinc(z / np.pi)),
            axis=-1,
        )
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

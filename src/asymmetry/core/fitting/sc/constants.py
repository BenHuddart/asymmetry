"""Physical constants and conversion helpers for superconducting models.

The Brandt relation used here is the common London-limit proportionality
for a triangular vortex lattice:

sigma_sc = C_B * gamma_mu * phi_0 / lambda_L^2

where:
- sigma_sc is in s^-1
- gamma_mu is in rad s^-1 T^-1
- phi_0 is in Wb (T m^2)
- lambda_L is in m

The second-moment width sigma_2 = gamma_mu * sqrt(<Delta B^2>) belongs to
exp(-sigma_2^2*t^2/2). The Gaussian component uses exp(-sigma_G^2*t^2), so
sigma_2 = sqrt(2)*sigma_G. Every conversion requires the width convention.

The helper conversions return practical units used in muSR analysis:

- input/output sigma in ``us^-1``
- input/output lambda in ``nm``

These conversions provide convenient estimates of absolute penetration depth.
Interpret absolute values in the context of the London-limit assumptions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.utils.constants import (
    MAGNETIC_FLUX_QUANTUM_WB,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

ArrayLikeFloat = NDArray[np.float64]

# Rounded sqrt(0.00371): Brandt, PRB 68, 054506 (2003), Eq. (11) gives the field variance.
BRANDT_COEFFICIENT = 0.0609

# Existing constant is in MHz/T cycles-per-second units, so convert to rad/s/T.
GAMMA_MU_RAD_S_PER_T = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * 1.0e6


class GaussianWidth(Enum):
    """Factor converting a stated Gaussian width into the second-moment width."""

    SECOND_MOMENT = 1.0
    GAUSSIAN_COMPONENT = np.sqrt(2.0)


def sigma_to_lambda_nm(
    sigma_us: NDArray[np.float64] | list[float] | float,
    *,
    width: GaussianWidth,
    brandt_coefficient: float = BRANDT_COEFFICIENT,
) -> ArrayLikeFloat:
    """Convert superconducting sigma (us^-1) to penetration depth lambda (nm).

    ``width=GaussianWidth.SECOND_MOMENT`` takes sigma_2 from exp(-sigma_2^2*t^2/2).
    ``width=GaussianWidth.GAUSSIAN_COMPONENT`` takes sigma_G from exp(-sigma_G^2*t^2).
    The conversion first applies sigma_2 = sqrt(2)*sigma_G when needed.

    For sigma <= 0, the returned lambda is +inf.
    """
    sigma_arr = np.asarray(sigma_us, dtype=float)
    sigma_s = np.maximum(sigma_arr, 0.0) * GaussianWidth(width).value * 1.0e6

    prefactor = float(brandt_coefficient) * GAMMA_MU_RAD_S_PER_T * MAGNETIC_FLUX_QUANTUM_WB
    lambda_m = np.full_like(sigma_s, np.inf, dtype=float)

    positive = sigma_s > 0.0
    lambda_m[positive] = np.sqrt(prefactor / sigma_s[positive])
    return np.asarray(lambda_m * 1.0e9, dtype=float)


def lambda_nm_to_sigma_us(
    lambda_nm: NDArray[np.float64] | list[float] | float,
    *,
    width: GaussianWidth,
    brandt_coefficient: float = BRANDT_COEFFICIENT,
) -> ArrayLikeFloat:
    """Convert penetration depth lambda (nm) to superconducting sigma (us^-1).

    The required ``width`` selects exp(-sigma_2^2*t^2/2) (SECOND_MOMENT) or
    exp(-sigma_G^2*t^2) (GAUSSIAN_COMPONENT), with sigma_G = sigma_2/sqrt(2).
    """
    lam_nm = np.asarray(lambda_nm, dtype=float)
    lam_m = np.maximum(lam_nm, 0.0) * 1.0e-9

    prefactor = float(brandt_coefficient) * GAMMA_MU_RAD_S_PER_T * MAGNETIC_FLUX_QUANTUM_WB
    sigma_s = np.zeros_like(lam_m, dtype=float)

    positive = lam_m > 0.0
    sigma_s[positive] = prefactor / np.square(lam_m[positive])
    return np.asarray(sigma_s * 1.0e-6 / GaussianWidth(width).value, dtype=float)

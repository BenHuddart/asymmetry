"""Physical constants and conversion helpers for superconducting models.

The Brandt relation used here is the common London-limit proportionality
for a triangular vortex lattice:

sigma_sc = C_B * gamma_mu * phi_0 / lambda_L^2

where:
- sigma_sc is in s^-1
- gamma_mu is in rad s^-1 T^-1
- phi_0 is in Wb (T m^2)
- lambda_L is in m

The helper conversions return practical units used in muSR analysis:

- input/output sigma in ``us^-1``
- input/output lambda in ``nm``

These conversions provide convenient estimates of absolute penetration depth.
Interpret absolute values in the context of the London-limit assumptions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.utils.constants import MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

ArrayLikeFloat = NDArray[np.float64]

# Flux quantum h/2e in Wb = T m^2.
FLUX_QUANTUM_WB = 2.067833848e-15
# Boltzmann constant in meV K^-1 for optional meV gap inputs.
BOLTZMANN_CONSTANT_MEV_PER_K = 8.617333262e-2
# Brandt prefactor for triangular lattice in the London limit.
BRANDT_COEFFICIENT = 0.0609

# Existing constant is in MHz/T cycles-per-second units, so convert to rad/s/T.
GAMMA_MU_RAD_S_PER_T = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * 1.0e6


def sigma_to_lambda_nm(
    sigma_us: NDArray[np.float64] | list[float] | float,
    *,
    brandt_coefficient: float = BRANDT_COEFFICIENT,
) -> ArrayLikeFloat:
    """Convert superconducting sigma (us^-1) to penetration depth lambda (nm).

    For sigma <= 0, the returned lambda is +inf.
    """
    sigma_arr = np.asarray(sigma_us, dtype=float)
    sigma_s = np.maximum(sigma_arr, 0.0) * 1.0e6

    prefactor = float(brandt_coefficient) * GAMMA_MU_RAD_S_PER_T * FLUX_QUANTUM_WB
    lambda_m = np.full_like(sigma_s, np.inf, dtype=float)

    positive = sigma_s > 0.0
    lambda_m[positive] = np.sqrt(prefactor / sigma_s[positive])
    return np.asarray(lambda_m * 1.0e9, dtype=float)


def lambda_nm_to_sigma_us(
    lambda_nm: NDArray[np.float64] | list[float] | float,
    *,
    brandt_coefficient: float = BRANDT_COEFFICIENT,
) -> ArrayLikeFloat:
    """Convert penetration depth lambda (nm) to superconducting sigma (us^-1)."""
    lam_nm = np.asarray(lambda_nm, dtype=float)
    lam_m = np.maximum(lam_nm, 0.0) * 1.0e-9

    prefactor = float(brandt_coefficient) * GAMMA_MU_RAD_S_PER_T * FLUX_QUANTUM_WB
    sigma_s = np.zeros_like(lam_m, dtype=float)

    positive = lam_m > 0.0
    sigma_s[positive] = prefactor / np.square(lam_m[positive])
    return np.asarray(sigma_s * 1.0e-6, dtype=float)

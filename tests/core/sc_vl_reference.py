"""Independent unit-cell quadrature of the defined calibrated London kernel.

The raw coefficients are Kadono et al., PRB 76, 094501 (2007), Eqs. (5)–(7),
and Barker et al., PRB 98, 104506 (2018), Eqs. (32)–(35). The RMS rescaling
and scalar powder substitution are separate Asymmetry definitions. Physical
constants and the rounded Brandt coefficient are shared; numerical helpers are not.
"""

import numpy as np

from asymmetry.core.fitting.sc.constants import BRANDT_COEFFICIENT
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MAGNETIC_FLUX_QUANTUM_WB,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)


def lattice_offsets(lam, field, bc2, *, powder=False, n_g=100, n_grid=512):
    """Explicit Fourier tensor at midpoint unit-cell coordinates, in tesla."""
    field_t = abs(field) * GAUSS_TO_TESLA
    effective_m = lam * (3**0.25 if powder else 1) * 1e-9
    modes = np.arange(-n_g, n_g + 1)
    # The triangular reciprocal metric follows from a cell of area Phi0/B.
    metric = modes[:, None] ** 2 + modes[None, :] ** 2 - modes[:, None] * modes[None, :]
    g2 = 8 * np.pi**2 * field_t / (np.sqrt(3) * MAGNETIC_FLUX_QUANTUM_WB) * metric
    xi2 = MAGNETIC_FLUX_QUANTUM_WB / (2 * np.pi * bc2)
    coefficients = np.exp(-xi2 * g2 / 2) / (1 + effective_m**2 * g2)
    coefficients[n_g, n_g] = 0
    coordinates = (np.arange(n_grid) + 0.5) / n_grid
    waves = np.exp(2j * np.pi * coordinates[:, None] * modes)
    offsets = (waves @ coefficients @ waves.T).real.ravel()
    b = field_t / bc2
    factor = (1 - b) * (1 + 1.21 * (1 - np.sqrt(b)) ** 3) / 2.21
    target_rms = BRANDT_COEFFICIENT * MAGNETIC_FLUX_QUANTUM_WB / effective_m**2 * factor
    return offsets * (target_rms / np.sqrt(np.mean(offsets**2)))


def lattice_average(times, lam, field, bc2, *, powder=False, n_g=100, n_grid=512):
    """Blocked direct exponentials, independent of histogram and FFT transforms."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    offsets = lattice_offsets(lam, field, bc2, powder=powder, n_g=n_g, n_grid=n_grid)
    rate = 2 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    result = np.zeros(times.shape, dtype=complex)
    for start in range(0, offsets.size, 1024):
        result += np.exp(1j * rate * offsets[start : start + 1024, None] * times).sum(axis=0)
    return result / offsets.size

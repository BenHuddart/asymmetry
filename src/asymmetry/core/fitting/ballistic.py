"""Ballistic longitudinal-field relaxation model.

Implements field-dependent relaxation rates for ballistic spin excitations:

- lambda(B_LF) = lambda_ball(B_LF) + lambda_0D
- lambda_ball(B_LF) = (C^2 / 4) * J(omega)
- omega = gamma_e * B_LF

For dimension n in {1, 2, 3}, the transport autocorrelation follows

    S_nD(t) = [J0(2 D_hop t)]^(2 n)

where J0 is the zeroth-order Bessel function and D_hop is the hopping rate in
us^-1. The spectral density is evaluated numerically using the same one-sided
cosine-transform convention used by the diffusive LF transport models:

    J(omega) = 2 * integral_0^inf S(t) cos(omega t) dt

Numerics
--------
The oscillatory transform is evaluated with scipy.integrate.quad using cosine
weighting in the scaled integration variable u = 2 D_hop t. This keeps the
integrand independent of D_hop apart from the reduced frequency kappa =
omega / (2 D_hop), which improves numerical stability across wide hopping-rate
ranges. The finite-frequency integral is evaluated on a bounded u interval
rather than using SciPy's infinite-cycle extrapolation, which is prone to
IntegrationWarning churn in fitting workflows.

For the one-dimensional low-frequency regime, the implementation uses the
standard asymptotic form J(omega) ~= (1 / (pi D_hop)) ln(16 D_hop / omega)
instead of forcing the oscillatory quadrature through the logarithmic limit.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.special import j0

from asymmetry.core.utils.constants import ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G


_QUAD_LIMIT = 500
_QUAD_EPSABS = 1e-8
_QUAD_EPSREL = 1e-6
_LOW_FREQUENCY_1D_RATIO_MAX = 2.0
_LOW_FREQUENCY_ND_RATIO_MAX = 0.01
_ZERO_FREQUENCY_U_MAX = {
    2: 2000.0,
    3: 1200.0,
}
_FINITE_FREQUENCY_U_MAX = {
    1: 1200.0,
    2: 600.0,
    3: 400.0,
}
_FINITE_FREQUENCY_U_MIN = {
    1: 200.0,
    2: 300.0,
    3: 200.0,
}
_MIN_1D_OSCILLATION_CYCLES = 160.0
_MIN_ND_OSCILLATION_CYCLES = 48.0


ArrayLikeFloat = NDArray[np.float64]


def _validate_dimension(n: int) -> None:
    if n not in {1, 2, 3}:
        raise ValueError("n must be one of {1, 2, 3}")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def _select_finite_frequency_u_max(kappa: float, n: int) -> float:
    base_u_max = _FINITE_FREQUENCY_U_MAX[n]
    if kappa <= 0.0:
        return base_u_max

    if n == 1:
        # The 1D tail only converges conditionally, so keep enough cosine
        # periods once we leave the low-frequency asymptotic branch.
        oscillation_u_max = _MIN_1D_OSCILLATION_CYCLES * (2.0 * math.pi / kappa)
        return float(max(_FINITE_FREQUENCY_U_MIN[n], min(base_u_max, oscillation_u_max)))

    oscillation_u_max = _MIN_ND_OSCILLATION_CYCLES * (2.0 * math.pi / kappa)
    return float(max(_FINITE_FREQUENCY_U_MIN[n], min(base_u_max, oscillation_u_max)))


@lru_cache(maxsize=256)
def _spectral_density_zero_frequency(d_hop: float, n: int) -> float:
    def s_scaled_scalar(u: float) -> float:
        return float(j0(u) ** (2 * n))

    integral, _ = quad(
        s_scaled_scalar,
        0.0,
        _ZERO_FREQUENCY_U_MAX[n],
        limit=_QUAD_LIMIT,
        epsabs=_QUAD_EPSABS,
        epsrel=_QUAD_EPSREL,
    )
    return float(integral / d_hop)


def autocorrelation_nD(
    t: NDArray[np.float64] | list[float],
    D_hop: float,
    n: int = 1,
) -> ArrayLikeFloat:
    """Return S_nD(t) for n-dimensional ballistic transport.

    Parameters
    ----------
    t
        Time in microseconds.
    D_hop
        Hopping rate in us^-1.
    n
        Dimensionality, one of 1, 2, or 3.
    """
    _validate_dimension(n)
    _validate_non_negative("D_hop", D_hop)

    tt = np.asarray(t, dtype=float)
    if np.any(tt < 0.0):
        raise ValueError("t must be >= 0")

    if D_hop == 0.0:
        return np.ones_like(tt, dtype=float)

    bessel = j0(2.0 * D_hop * tt)
    return np.asarray(np.power(bessel, 2 * n), dtype=float)


def spectral_density(omega: float, D_hop: float, n: int = 1) -> float:
    """Return one-sided cosine spectral density J(omega).

    Uses the convention J(omega) = 2 * integral_0^inf S(t) cos(omega t) dt.
    For n = 1 and omega = 0, the integral diverges logarithmically, so this
    function returns +inf. For D_hop = 0 and omega > 0, the transform tends to
    zero in the distributional sense, which is the value returned here.
    """
    _validate_dimension(n)
    _validate_non_negative("D_hop", D_hop)

    w = abs(float(omega))
    if D_hop == 0.0:
        return float("inf") if w == 0.0 else 0.0
    if w == 0.0 and n == 1:
        return float("inf")
    if n == 1 and (w / D_hop) <= _LOW_FREQUENCY_1D_RATIO_MAX:
        return float((1.0 / (math.pi * D_hop)) * math.log((16.0 * D_hop) / w))
    if n >= 2 and (w / D_hop) <= _LOW_FREQUENCY_ND_RATIO_MAX:
        return _spectral_density_zero_frequency(D_hop, n)

    kappa = w / (2.0 * D_hop)

    def s_scaled_scalar(u: float) -> float:
        return float(j0(u) ** (2 * n))

    if kappa == 0.0:
        return _spectral_density_zero_frequency(D_hop, n)
    else:
        u_max = _select_finite_frequency_u_max(kappa, n)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            integral, _ = quad(
                s_scaled_scalar,
                0.0,
                u_max,
                weight="cos",
                wvar=kappa,
                limit=_QUAD_LIMIT,
                epsabs=_QUAD_EPSABS,
                epsrel=_QUAD_EPSREL,
            )

    value = float(integral / D_hop)
    if value < 0.0:
        return 0.0
    return value


def lambda_ball(
    B_LF: NDArray[np.float64] | list[float] | float,
    C: float,
    D_hop: float,
    n: int = 1,
) -> ArrayLikeFloat:
    """Return field-dependent ballistic relaxation rate lambda_ball(B_LF)."""
    _validate_dimension(n)
    _validate_non_negative("D_hop", D_hop)

    b_arr = np.asarray(B_LF, dtype=float)
    omega = np.abs(ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G * b_arr)

    out = np.empty_like(omega, dtype=float)
    flat_omega = np.ravel(omega)
    flat_out = np.ravel(out)

    cache: dict[float, float] = {}
    for i, w in enumerate(flat_omega):
        ww = float(w)
        if ww not in cache:
            cache[ww] = spectral_density(ww, D_hop=D_hop, n=n)
        flat_out[i] = cache[ww]

    prefactor = (float(C) ** 2) / 4.0
    return np.asarray(prefactor * out, dtype=float)


def lambda_total(
    B_LF: NDArray[np.float64] | list[float] | float,
    C: float,
    D_hop: float,
    lambda_0D: float = 0.0,
    n: int = 1,
) -> ArrayLikeFloat:
    """Return total LF relaxation lambda(B_LF) including field-independent term."""
    lam = lambda_ball(B_LF=B_LF, C=C, D_hop=D_hop, n=n)
    return np.asarray(lam + float(lambda_0D), dtype=float)


def is_scipy_available() -> bool:
    """Return whether SciPy is available.

    With eager imports, SciPy is now a hard requirement loaded at module import.
    This function always returns True; it exists for backward compatibility.
    """
    return True
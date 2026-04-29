"""Diffusive longitudinal-field relaxation model.

Implements field-dependent relaxation rates for diffusive spin excitations:

- lambda(B_LF) = lambda_diff(B_LF) + lambda_0D
- lambda_diff(B_LF) = (C^2 / 4) * J(omega)
- omega = gamma_e * B_LF

The diffusion autocorrelation uses the nD form from Pratt (J. Phys.: Conf.
Ser. 2462 012038, 2023), with n in {1, 2, 3}:

S_nD(t) = [exp(-2 D_nD t) I0(2 D_nD t)]^n
          [exp(-2 D_perp t) I0(2 D_perp t)]^(3-n)

Numerics
--------
The spectral density is evaluated as a one-sided cosine transform:

J(omega) = 2 * integral_0^inf S(t) cos(omega t) dt

We evaluate this using scipy.integrate.quad with cosine weighting. A finite
upper limit t_max is chosen adaptively from the diffusion rates to provide
stable and reproducible behavior suitable for GUI fitting workflows.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import i0e

from asymmetry.core.utils.constants import ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G

# Integration controls exposed as module constants for advanced tuning.
_T_MAX_SCALE = 400.0
# Keep a small floor to avoid t_max -> 0 for extremely fast diffusion.
# A large floor (e.g., 100 us) can cause spurious cliffs at high field because
# oscillatory weighted integration spans far beyond the correlation timescale.
_MIN_T_MAX_US = 0.1
_QUAD_LIMIT = 500
_QUAD_EPSABS = 1e-8
_QUAD_EPSREL = 1e-6


ArrayLikeFloat = NDArray[np.float64]


def _validate_dimension(n: int) -> None:
    if n not in {1, 2, 3}:
        raise ValueError("n must be one of {1, 2, 3}")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def autocorrelation_nD(
    t: NDArray[np.float64] | list[float],
    D_nD: float,
    D_perp: float = 0.0,
    n: int = 2,
) -> ArrayLikeFloat:
    """Return S_nD(t) for n-dimensional diffusion.

    Parameters
    ----------
    t
        Time in microseconds.
    D_nD
        In-plane diffusion rate in us^-1.
    D_perp
        Perpendicular diffusion rate in us^-1.
    n
        Dimensionality, one of 1, 2, or 3.
    """
    _validate_dimension(n)
    _validate_non_negative("D_nD", D_nD)
    _validate_non_negative("D_perp", D_perp)

    tt = np.asarray(t, dtype=float)
    if np.any(tt < 0.0):
        raise ValueError("t must be >= 0")

    in_plane = np.power(i0e(2.0 * D_nD * tt), n)
    if n == 3:
        return np.asarray(in_plane, dtype=float)

    perp_power = 3 - n
    perp = np.power(i0e(2.0 * D_perp * tt), perp_power)
    return np.asarray(in_plane * perp, dtype=float)


def _select_t_max_us(omega: float, D_nD: float, D_perp: float, n: int) -> float:
    """Choose a robust finite integration upper bound for S(t)."""
    # For large t, exp(-2Dt) I0(2Dt) ~ 1/sqrt(4*pi*D*t), giving t^{-n/2} asymptotics.
    # We scale by the smallest active diffusion rate to capture the long-time tail.
    active_rates: list[float] = [D_nD]
    if n < 3 and D_perp > 0.0:
        active_rates.append(D_perp)

    min_rate = max(min(active_rates), 1e-6)
    t_from_rate = _T_MAX_SCALE / min_rate

    # For very small omega, increase range to sample the first oscillation period.
    if omega > 0.0:
        period_scale = 40.0 * (2.0 * math.pi / omega)
    else:
        period_scale = t_from_rate

    return float(max(_MIN_T_MAX_US, t_from_rate, period_scale))


def spectral_density(omega: float, D_nD: float, D_perp: float = 0.0, n: int = 2) -> float:
    """Return one-sided cosine spectral density J(omega).

    Uses the convention J(omega) = 2 * integral_0^inf S(t) cos(omega t) dt.
    For omega = 0, this function returns +inf because the integral diverges for
    algebraic long-time tails relevant to low-dimensional diffusion.
    """
    _validate_dimension(n)
    _validate_non_negative("D_nD", D_nD)
    _validate_non_negative("D_perp", D_perp)

    w = abs(float(omega))
    if w == 0.0:
        return float("inf")

    t_max = _select_t_max_us(w, D_nD, D_perp, n)

    def s_scalar(t: float) -> float:
        return float(autocorrelation_nD(np.array([t]), D_nD=D_nD, D_perp=D_perp, n=n)[0])

    integral, _ = quad(
        s_scalar,
        0.0,
        t_max,
        weight="cos",
        wvar=w,
        limit=_QUAD_LIMIT,
        epsabs=_QUAD_EPSABS,
        epsrel=_QUAD_EPSREL,
    )
    return float(2.0 * integral)


def lambda_diff(
    B_LF: NDArray[np.float64] | list[float] | float,
    C: float,
    D_nD: float,
    D_perp: float = 0.0,
    n: int = 2,
) -> ArrayLikeFloat:
    """Return field-dependent diffusive relaxation rate lambda_diff(B_LF)."""
    _validate_dimension(n)
    _validate_non_negative("D_nD", D_nD)
    _validate_non_negative("D_perp", D_perp)

    b_arr = np.asarray(B_LF, dtype=float)
    omega = np.abs(ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G * b_arr)

    out = np.empty_like(omega, dtype=float)
    flat_omega = np.ravel(omega)
    flat_out = np.ravel(out)

    # quad is scalar; cache repeated omega values to reduce total integrations.
    cache: dict[float, float] = {}
    for i, w in enumerate(flat_omega):
        ww = float(w)
        if ww not in cache:
            cache[ww] = spectral_density(ww, D_nD=D_nD, D_perp=D_perp, n=n)
        flat_out[i] = cache[ww]

    prefactor = (float(C) ** 2) / 4.0
    return np.asarray(prefactor * out, dtype=float)


def lambda_total(
    B_LF: NDArray[np.float64] | list[float] | float,
    C: float,
    D_nD: float,
    lambda_0D: float = 0.0,
    D_perp: float = 0.0,
    n: int = 2,
) -> ArrayLikeFloat:
    """Return total LF relaxation lambda(B_LF) including field-independent term."""
    lam = lambda_diff(B_LF=B_LF, C=C, D_nD=D_nD, D_perp=D_perp, n=n)
    return np.asarray(lam + float(lambda_0D), dtype=float)


def is_scipy_available() -> bool:
    """Return whether SciPy is available.

    With eager imports, SciPy is now a hard requirement loaded at module import.
    This function always returns True; it exists for backward compatibility.
    """
    return True

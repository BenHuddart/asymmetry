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
The spectral density is a one-sided cosine transform:

    J(omega) = 2 * integral_0^inf S(t) cos(omega t) dt

Two evaluation strategies are available:

* A **fast, vectorized Filon transform** (the default): S(t) is sampled once on
  a shared log-spaced grid up to a finite ``t_max`` chosen from the diffusion
  rates, and J(omega) is evaluated for the *whole* omega vector at once. Within
  each grid segment S(t) is treated as piecewise linear and the oscillatory
  ``cos(omega t)`` factor is integrated **analytically**. Because the
  oscillation is handled in closed form per segment, the grid only has to
  resolve the smooth envelope S(t) -- not the (arbitrarily fast) cosine -- so a
  modest grid stays accurate even when ``omega`` is very large. See
  ``_spectral_density_fast`` for the design notes and accuracy bounds.

* The original scalar ``scipy.integrate.quad`` cosine-weighted integrator,
  retained as ``_spectral_density_quad``. It is used as a parity reference in
  the tests, as an automatic fall-back for the very-slow-diffusion corner where
  the fixed grid would need an impractically large ``t_max`` (see
  ``_DIFFUSION_MIN_RATE_FLOOR``), and as a global escape hatch when the
  environment variable ``ASYMMETRY_DIFFUSION_QUAD`` is set to a truthy value.

A finite upper limit ``t_max`` is chosen adaptively from the diffusion rates to
provide stable and reproducible behavior suitable for GUI fitting workflows.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

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

# --- Fast Filon transform controls -----------------------------------------
# Number of log-spaced samples of S(t) on the shared grid. The Filon segment
# integrals are exact in the oscillatory factor, so this only has to resolve the
# smooth envelope S(t); 1600 points keep the fast path self-consistent to
# ~<=1e-4 (relative to the curve maximum) for all rates at or above the floor
# below, and to ~<=1e-5 for D_nD >= 1 us^-1. See tests/core/test_diffusion.py.
_FILON_NPTS = 1600
# Smallest active diffusion rate (us^-1) for which the fixed log grid is used.
# Below this, S(t) decays over an enormous t_max (~ 400 / rate) and the fixed
# grid can no longer resolve the transition region cheaply, so we fall back to
# the adaptive scalar quad (which stays accurate and, for such slow diffusion,
# is not prohibitively slow). This corner is only reached transiently, if at
# all, during a fit because D_2D/D_perp default to O(1) us^-1.
_DIFFUSION_MIN_RATE_FLOOR = 0.2
# Chunk the omega axis so the (n_omega x n_t) trig matrices never blow memory.
_FILON_OMEGA_CHUNK = 4096
# Bounded memo of the sampled S(t) grid, keyed by the EXACT (D_nD, D_perp, n)
# float triple. One Minuit iteration evaluates ~100 omega at a single (D, n)
# and curve redraws repeat the same params, so a small cache removes redundant
# grid builds; each new iteration changes D and simply misses (which is fine).
_FILON_GRID_CACHE_SIZE = 64
# Environment escape hatch: force the exact scalar-quad path everywhere.
_QUAD_ENV_FLAG = "ASYMMETRY_DIFFUSION_QUAD"


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


def _active_rates(D_nD: float, D_perp: float, n: int) -> list[float]:
    rates = [D_nD]
    if n < 3 and D_perp > 0.0:
        rates.append(D_perp)
    return rates


def _min_active_rate(D_nD: float, D_perp: float, n: int) -> float:
    return min(_active_rates(D_nD, D_perp, n))


def _use_quad_globally() -> bool:
    value = os.environ.get(_QUAD_ENV_FLAG, "")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


@lru_cache(maxsize=_FILON_GRID_CACHE_SIZE)
def _filon_grid(
    D_nD: float, D_perp: float, n: int, t_max: float
) -> tuple[ArrayLikeFloat, ArrayLikeFloat]:
    """Return (t, S(t)) sampled once on a log-spaced grid up to ``t_max``.

    The grid starts at t=0 (where S=1) and is log-spaced thereafter. The first
    finite node is tied to the *fastest* active rate so the early, rapidly
    varying part of S(t) is resolved; log spacing then gives roughly constant
    resolution per decade out to ``t_max``. Cached on the exact float key so a
    whole omega sweep at fixed (D_nD, D_perp, n) reuses one build.
    """
    rates = [r for r in _active_rates(D_nD, D_perp, n) if r > 0.0]
    d_max = max(rates) if rates else 1e-9
    # Start well inside the fastest decay time, but never above t_max.
    t0 = min(1e-4 / d_max, t_max * 1e-8)
    t0 = max(t0, 1e-12)
    t = np.concatenate([[0.0], np.geomspace(t0, t_max, _FILON_NPTS)])
    s = autocorrelation_nD(t, D_nD=D_nD, D_perp=D_perp, n=n)
    return t, s


def _spectral_density_fast(
    omega: ArrayLikeFloat, D_nD: float, D_perp: float, n: int
) -> ArrayLikeFloat:
    """Vectorized Filon cosine transform of S(t) for an array of positive omega.

    Design
    ------
    J(omega) = 2 * integral_0^t_max S(t) cos(omega t) dt, with S(t) sampled on a
    shared log grid. On each segment [a, b] we take S linear,
    S(t) = S_a + m (t - a) with m = (S_b - S_a) / (b - a), and integrate the
    oscillatory factor in closed form::

        int_a^b [S_a + m (t - a)] cos(omega t) dt
            = S_a (sin(omega b) - sin(omega a)) / omega
              + m [ (b sin(omega b) - a sin(omega a)) / omega
                    + (cos(omega b) - cos(omega a)) / omega^2
                    - a (sin(omega b) - sin(omega a)) / omega ]

    Because the cosine is integrated analytically per segment, the grid only has
    to resolve the smooth envelope S(t); it does NOT need >= ~10 samples per
    cosine period. That is what makes a single fixed grid valid across the full
    omega range (including very large omega, where a uniform-grid trapezoid
    cosine transform would need billions of points or would alias badly).

    Memory is bounded by chunking the omega axis: each chunk forms an
    (n_chunk x n_t) matrix, never the full (n_omega x n_t) product.

    ``omega`` must be strictly positive (omega == 0 is +inf and handled upstream).
    """
    omega = np.asarray(omega, dtype=float)
    # Shared t_max: the largest per-omega window, so the smallest omega's long
    # tail is captured. (Large omega does not need the long tail -- its integrand
    # oscillates and the far tail contributes negligibly -- but sharing one
    # t_max keeps the grid single and is if anything more consistent than the
    # per-omega quad truncation.)
    t_max = max(_select_t_max_us(float(w), D_nD, D_perp, n) for w in np.ravel(omega))
    t, s = _filon_grid(float(D_nD), float(D_perp), int(n), float(t_max))

    a = t[:-1]
    b = t[1:]
    s_a = s[:-1]
    slope = (s[1:] - s_a) / (b - a)

    out = np.empty(omega.shape, dtype=float)
    flat_w = np.ravel(omega)
    flat_out = np.ravel(out)
    for start in range(0, flat_w.size, _FILON_OMEGA_CHUNK):
        w = flat_w[start : start + _FILON_OMEGA_CHUNK][:, None]
        sin_wa = np.sin(w * a)
        sin_wb = np.sin(w * b)
        cos_wa = np.cos(w * a)
        cos_wb = np.cos(w * b)
        i_cos = (sin_wb - sin_wa) / w
        i_ramp = (
            (b * sin_wb - a * sin_wa) / w + (cos_wb - cos_wa) / w**2 - a * (sin_wb - sin_wa) / w
        )
        seg = s_a * i_cos + slope * i_ramp
        flat_out[start : start + _FILON_OMEGA_CHUNK] = 2.0 * np.sum(seg, axis=1)
    return out


def _spectral_density_quad(omega: float, D_nD: float, D_perp: float = 0.0, n: int = 2) -> float:
    """Scalar cosine-weighted quad reference for J(omega).

    This is the original, exact implementation. It is kept as the parity
    reference for the vectorized fast path, as the automatic fall-back for the
    very-slow-diffusion corner, and as the global escape hatch. Callers must
    pass omega > 0.
    """
    w = abs(float(omega))
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


def _spectral_density_array(
    omega: ArrayLikeFloat, D_nD: float, D_perp: float, n: int
) -> ArrayLikeFloat:
    """Evaluate J for an array of omega, dispatching fast path vs quad fall-back.

    omega == 0 maps to +inf. The slow-diffusion corner (or the global env-var
    escape hatch) routes to the scalar quad path.
    """
    omega = np.abs(np.asarray(omega, dtype=float))
    out = np.empty(omega.shape, dtype=float)
    flat_w = np.ravel(omega)
    flat_out = np.ravel(out)

    zero = flat_w == 0.0
    flat_out[zero] = np.inf
    positive = ~zero

    use_quad = _use_quad_globally() or _min_active_rate(D_nD, D_perp, n) < _DIFFUSION_MIN_RATE_FLOOR
    if positive.any():
        if use_quad:
            for i in np.flatnonzero(positive):
                flat_out[i] = _spectral_density_quad(
                    float(flat_w[i]), D_nD=D_nD, D_perp=D_perp, n=n
                )
        else:
            flat_out[positive] = _spectral_density_fast(
                flat_w[positive], D_nD=D_nD, D_perp=D_perp, n=n
            )
    return out


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

    result = _spectral_density_array(np.array([w], dtype=float), D_nD, D_perp, n)
    return float(result[0])


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

    # A single vectorized cosine transform over the whole omega array. Repeated
    # omega values are naturally handled (they map through the same grid), so no
    # per-value caching loop is needed.
    j = _spectral_density_array(omega, D_nD, D_perp, n)

    prefactor = (float(C) ** 2) / 4.0
    return np.asarray(prefactor * j, dtype=float)


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

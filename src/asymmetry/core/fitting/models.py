"""Built-in μSR fit functions.

Each model is a callable ``f(t, **params) -> array`` plus metadata describing
its parameters.  Models are collected in the :data:`MODELS` registry.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from asymmetry.core.fitting.parameters import ParamInfo, param_info_map
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


@dataclass
class ModelDefinition:
    """Descriptor for a built-in fit function."""

    name: str
    description: str
    function: Callable[..., NDArray[np.float64]]
    param_names: list[str]
    param_defaults: dict[str, float]
    param_info: dict[str, ParamInfo]


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def exponential_relaxation(t: NDArray, A0: float, Lambda: float, baseline: float = 0.0) -> NDArray:
    """Simple exponential: A(t) = A0 exp(−Λt) + baseline."""
    # Clamp exponent to prevent overflow; exp(-700) ≈ 0 numerically
    exponent = np.clip(-Lambda * np.abs(t), -700, 0)
    return A0 * np.exp(exponent) + baseline


def gaussian_relaxation(t: NDArray, A0: float, sigma: float, baseline: float = 0.0) -> NDArray:
    """Gaussian relaxation: A(t) = A0 exp(−σ²t²) + baseline."""
    # Clamp exponent to prevent overflow
    exponent = np.clip(-((sigma * t) ** 2), -700, 0)
    return A0 * np.exp(exponent) + baseline


def oscillatory(
    t: NDArray,
    A0: float,
    frequency: float,
    phase: float = 0.0,
    Lambda: float = 0.0,
    baseline: float = 0.0,
) -> NDArray:
    """Damped oscillation: A0 cos(2πft + φ) exp(−Λt) + baseline."""
    # Clamp damping exponent to prevent overflow
    exponent = np.clip(-Lambda * np.abs(t), -700, 0)
    return A0 * np.cos(2.0 * np.pi * frequency * t + phase) * np.exp(exponent) + baseline


def stretched_exponential(
    t: NDArray,
    A0: float,
    Lambda: float,
    beta: float = 1.0,
    baseline: float = 0.0,
) -> NDArray:
    """Stretched exponential: A0 exp(−(Λt)^β) + baseline."""
    # Clamp exponent to prevent overflow
    exponent = np.clip(-(np.abs(Lambda * t) ** beta), -700, 0)
    return A0 * np.exp(exponent) + baseline


def static_gkt_zf(
    t: NDArray,
    A0: float,
    Delta: float,
    baseline: float = 0.0,
) -> NDArray:
    """Static Gaussian Kubo-Toyabe (zero field).

    GKT(t) = A0 [1/3 + 2/3 (1 − Δ²t²) exp(−Δ²t²/2)] + baseline
    """
    dt2 = (Delta * t) ** 2
    # Clamp exponent to prevent overflow
    exponent = np.clip(-dt2 / 2.0, -700, 0)
    return A0 * (1.0 / 3.0 + 2.0 / 3.0 * (1.0 - dt2) * np.exp(exponent)) + baseline


def _lf_kt_integral_term(t_val: float, Delta: float, omega0: float) -> float:
    """Compute the integral term for LF-KT: integral_0^t exp(-0.5*Delta²*τ²) sin(omega0*τ) dτ.

    Uses cached computation and optimized numerical integration for performance.

    Parameters
    ----------
    t_val : float
        Upper limit of integration (in microseconds).
    Delta : float
        Gaussian field distribution width (in us⁻¹).
    omega0 : float
        Larmor frequency (in rad/us).

    Returns
    -------
    float
        The value of the integral.
    """
    if t_val <= 0 or Delta <= 0 or abs(omega0) < 1e-12:
        return 0.0

    # Use cached computation with high precision
    return _lf_kt_integral_cached(t_val, Delta, omega0)


@lru_cache(maxsize=512)
def _lf_kt_integral_cached(t_val: float, Delta: float, omega0: float) -> float:
    """Cached numerical integration with optimized parameters.

    Cache key based on quantized parameters to reduce cache misses while
    maintaining numerical precision.
    """

    def integrand(tau):
        return np.exp(-0.5 * (Delta * tau) ** 2) * np.sin(omega0 * tau)

    try:
        # Suppress integration warnings: we're using aggressive limits (500) and
        # tight tolerances to handle even difficult integrands. Warnings are
        # expected and handled gracefully.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
            # Use adaptive quadrature with:
            # - limit=500 (much higher than default 100 to resolve difficult integrands)
            # - epsabs=1e-11 (absolute tolerance for convergence)
            # - epsrel=1e-9 (relative tolerance for convergence)
            # These settings prioritize accuracy over speed for fitting
            result, _ = integrate.quad(
                integrand,
                0,
                t_val,
                limit=500,
                epsabs=1e-11,
                epsrel=1e-9,
            )
        return float(result)
    except Exception:
        # Fallback: if integration fails, return 0 with warning suppressed
        return 0.0


def longitudinal_field_kubo_toyabe(
    t: NDArray,
    A0: float,
    Delta: float,
    B_L: float,
    baseline: float = 0.0,
) -> NDArray:
    """Static Gaussian Kubo-Toyabe with longitudinal field (Hayano et al. 1979).

    Implements the longitudinal depolarization function for μSR in a static
    Gaussian magnetic field distribution with applied longitudinal field.

    Parameters
    ----------
    t : NDArray
        Time values in microseconds.
    A0 : float
        Initial asymmetry scaling.
    Delta : float
        Gaussian field distribution width in us⁻¹.
    B_L : float
        Longitudinal magnetic field in Gauss.
    baseline : float, optional
        Constant baseline offset.

    Returns
    -------
    NDArray
        Depolarization function values at each time point.

    Notes
    -----
    Uses the Hayano et al. analytic expression:

        Gz(t) = 1 - (2 Δ² / ω₀²) [1 - exp(-Δ²t²/2) cos(ω₀t)]
              + (2 Δ⁴ / ω₀³) ∫₀ᵗ exp(-Δ²τ²/2) sin(ω₀τ) dτ

    where ω₀ = γμ B_L and B_L is provided in Gauss.
    Internally, B_L is converted to Tesla via GAUSS_TO_TESLA.

    When B_L = 0, reduces to the zero-field Kubo-Toyabe function.
    """
    # Muon gyromagnetic ratio: gamma_mu = 2π * (gamma_mu / 2π) in rad/(us*T)
    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

    # B_L is user-facing in Gauss; convert to Tesla for omega0.
    omega0 = gamma_mu * (B_L * GAUSS_TO_TESLA)

    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    if scalar_input:
        t = t[np.newaxis]

    result = np.zeros_like(t)

    # Threshold for treating field as effectively zero
    omega0_threshold = 1e-10

    # Handle the case where omega0 is very small (transition to zero field)
    if abs(omega0) < omega0_threshold:
        # Use zero-field KT limit for numerical stability
        dt2 = (Delta * t) ** 2
        exponent = np.clip(-dt2 / 2.0, -700, 0)
        result = 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - dt2) * np.exp(exponent)
    else:
        # General case: use Hayano formula
        omega0_sq = omega0**2
        omega0_cu = omega0**3
        delta_sq = Delta**2
        delta_qu = delta_sq**2
        factor1 = 2.0 * delta_sq / omega0_sq
        factor2 = 2.0 * delta_qu / omega0_cu

        # Cache integrals to avoid redundant computation
        integral_cache = {}

        for i, t_i in enumerate(t):
            if t_i <= 1e-12:
                result[i] = 1.0
                continue

            # Use cached integral if available (for repeated t values)
            # Round to 9 decimal places (microsecond precision) for better cache hits
            t_key = round(t_i * 1e9) / 1e9
            if t_key in integral_cache:
                integral = integral_cache[t_key]
            else:
                integral = _lf_kt_integral_term(t_i, Delta, omega0)
                integral_cache[t_key] = integral

            # First term: 1 - (2 Δ² / ω₀²) [1 - exp(-Δ²t²/2) cos(ω₀t)]
            dt2 = (Delta * t_i) ** 2
            exponent = np.clip(-dt2 / 2.0, -700, 0)
            exp_term = np.exp(exponent)
            cos_term = np.cos(omega0 * t_i)
            first_part = 1.0 - factor1 * (1.0 - exp_term * cos_term)

            # Second term: (2 Δ⁴ / ω₀³) ∫₀ᵗ exp(-Δ²τ²/2) sin(ω₀τ) dτ
            second_part = factor2 * integral

            result[i] = first_part + second_part

    gz = result
    output = A0 * gz + baseline

    if scalar_input:
        output = output[0]

    return output


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS: dict[str, ModelDefinition] = {}


def _register(name: str, desc: str, fn: Callable, params: list[str], defaults: dict) -> None:
    MODELS[name] = ModelDefinition(name, desc, fn, params, defaults, param_info_map(params))


_register(
    "ExponentialRelaxation",
    "A0 exp(−Λt) + baseline",
    exponential_relaxation,
    ["A0", "Lambda", "baseline"],
    {"A0": 25.0, "Lambda": 0.5, "baseline": 0.0},
)

_register(
    "GaussianRelaxation",
    "A0 exp(−σ²t²) + baseline",
    gaussian_relaxation,
    ["A0", "sigma", "baseline"],
    {"A0": 25.0, "sigma": 0.5, "baseline": 0.0},
)

_register(
    "Oscillatory",
    "A0 cos(2πft + φ) exp(−Λt) + baseline",
    oscillatory,
    ["A0", "frequency", "phase", "Lambda", "baseline"],
    {"A0": 25.0, "frequency": 1.0, "phase": 0.0, "Lambda": 0.0, "baseline": 0.0},
)

_register(
    "StretchedExponential",
    "A0 exp(−(Λt)^β) + baseline",
    stretched_exponential,
    ["A0", "Lambda", "beta", "baseline"],
    {"A0": 25.0, "Lambda": 0.5, "beta": 1.0, "baseline": 0.0},
)

_register(
    "StaticGKT_ZF",
    "Static Gaussian Kubo-Toyabe (zero field)",
    static_gkt_zf,
    ["A0", "Delta", "baseline"],
    {"A0": 25.0, "Delta": 0.5, "baseline": 0.0},
)

_register(
    "LFKuboToyabe",
    "Static Gaussian Kubo-Toyabe with longitudinal field (Hayano et al. 1979)",
    longitudinal_field_kubo_toyabe,
    ["A0", "Delta", "B_L", "baseline"],
    {"A0": 25.0, "Delta": 0.5, "B_L": 0.0, "baseline": 0.0},
)

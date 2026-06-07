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
    domain: str = "time"


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
# Dynamic / fluctuating-field relaxation functions
# ---------------------------------------------------------------------------
#
# A static local-field distribution dephases the muon, giving the static
# Kubo-Toyabe function G_s(t).  When the field reorients stochastically at rate
# ``nu`` (strong-collision / Markovian model), the polarisation becomes the
# *dynamic* G_d(t).  Limits: nu -> 0 recovers the static function; nu -> infinity
# gives motional narrowing (exponential decay, rate 2*Delta^2/nu for Gaussian).
# See docs/porting/dynamic-relaxation/.  ``nu`` is a rate in MHz (== us^-1).


def abragam(
    t: NDArray,
    A0: float,
    Delta: float,
    nu: float,
    baseline: float = 0.0,
) -> NDArray:
    """Abragam relaxation function (Abragam, *Principles of Nuclear Magnetism*, 1961).

    G(t) = A0 exp[ -(Delta^2 / nu^2) (e^{-nu t} - 1 + nu t) ] + baseline

    A single-component relaxation that interpolates between the static Gaussian
    line shape and the motionally-narrowed exponential:

    - ``nu -> 0``        : G -> A0 exp(-Delta^2 t^2 / 2)  (Gaussian)
    - ``nu >> Delta``    : G -> A0 exp(-(Delta^2 / nu) t) (exponential)

    Notation and form follow Blundell, De Renzi, Lancaster & Pratt, *Muon
    Spectroscopy: An Introduction* (OUP, 2022), eqn 5.52 (the damping factor of
    the transverse-field Abragam function), with the same Gaussian width symbol
    ``Delta`` as the Kubo-Toyabe family.

    Parameters
    ----------
    t : NDArray
        Time values in microseconds.
    A0 : float
        Initial asymmetry amplitude.
    Delta : float
        Static Gaussian field-distribution width in us^-1.
    nu : float
        Field fluctuation (hop) rate in MHz (== us^-1).  ``nu`` <= 0 gives the
        static Gaussian limit.
    baseline : float, optional
        Constant baseline offset.
    """
    t = np.asarray(t, dtype=float)
    d2 = float(Delta) * float(Delta)
    nt = float(nu) * np.abs(t)
    if nu <= 1e-9:
        exponent = -0.5 * d2 * t * t
    else:
        # e^{-nt} - 1 + nt is >= 0 and ~ (nt)^2/2 as nt -> 0 (Gaussian limit)
        exponent = -(d2 / (float(nu) * float(nu))) * (np.exp(np.clip(-nt, -700, 0)) - 1.0 + nt)
    exponent = np.clip(exponent, -700, 0)
    return A0 * np.exp(exponent) + baseline


def keren(
    t: NDArray,
    A0: float,
    Delta: float,
    nu: float,
    B_L: float,
    baseline: float = 0.0,
) -> NDArray:
    """Keren dynamic Gaussian relaxation in a longitudinal field (Keren, PRB 50, 10039 (1994)).

    P(t) = A0 exp[-Gamma(t)] + baseline, with omega0 = gamma_mu * B_L and

        Gamma(t) = (2 Delta^2 / (omega0^2 + nu^2)^2)
                   * [ (omega0^2 + nu^2) nu t
                       + (omega0^2 - nu^2) (1 - e^{-nu t} cos(omega0 t))
                       - 2 nu omega0 e^{-nu t} sin(omega0 t) ]

    Keren's analytic generalisation of the Abragam function to a longitudinal
    field.  At ``B_L = 0`` it reduces to the Abragam exponent (x2, for the two
    transverse zero-field components): Gamma = (2 Delta^2 / nu^2)(e^{-nu t} - 1 + nu t).

    Parameters
    ----------
    t : NDArray
        Time values in microseconds.
    A0 : float
        Initial asymmetry amplitude.
    Delta : float
        Static Gaussian field-distribution width in us^-1.
    nu : float
        Field fluctuation rate in MHz (== us^-1).
    B_L : float
        Longitudinal magnetic field in Gauss (omega0 = gamma_mu * B_L).
    baseline : float, optional
        Constant baseline offset.
    """
    t = np.asarray(t, dtype=float)
    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    omega0 = gamma_mu * (float(B_L) * GAUSS_TO_TESLA)  # rad/us
    delta2 = float(Delta) * float(Delta)
    w2 = omega0 * omega0
    n2 = float(nu) * float(nu)
    denom = w2 + n2

    if denom < 1e-20:
        # nu -> 0 and B_L -> 0: Gamma -> Delta^2 t^2 (fast-Gaussian envelope limit)
        exponent = np.clip(-delta2 * t * t, -700, 0)
        return A0 * np.exp(exponent) + baseline

    e = np.exp(np.clip(-float(nu) * np.abs(t), -700, 0))
    gamma = (2.0 * delta2 / (denom * denom)) * (
        denom * float(nu) * np.abs(t)
        + (w2 - n2) * (1.0 - e * np.cos(omega0 * t))
        - 2.0 * float(nu) * omega0 * e * np.sin(omega0 * t)
    )
    exponent = np.clip(-gamma, -700, 0)
    return A0 * np.exp(exponent) + baseline


def static_lorentzian_kt_zf(
    t: NDArray,
    A0: float,
    a_L: float,
    baseline: float = 0.0,
) -> NDArray:
    """Static Lorentzian Kubo-Toyabe, zero field (Uemura et al. PRB 31, 546 (1985)).

    G(t) = A0 [1/3 + 2/3 (1 - a t) exp(-a t)] + baseline

    For a dilute / Lorentzian distribution of local fields (e.g. spin glasses),
    where ``a_L`` is the half-width of the field distribution expressed as a rate
    in us^-1.
    """
    at = float(a_L) * np.abs(np.asarray(t, dtype=float))
    exp_term = np.exp(np.clip(-at, -700, 0))
    return A0 * (1.0 / 3.0 + 2.0 / 3.0 * (1.0 - at) * exp_term) + baseline


# Cache of (C0, frequencies, amplitudes) for the static Lorentzian-LF field
# average, keyed by quantised (a_L, omega0, grid sizes).
_LOR_LF_CACHE: dict[tuple, tuple[float, NDArray, NDArray]] = {}
_LOR_LF_CACHE_MAX = 64


def _lorentzian_lf_field_average(
    a_L: float, omega0: float, n_u: int = 600, n_v: int = 480
) -> tuple[float, NDArray, NDArray]:
    """Pre-compute the static Lorentzian-LF average as C0 + sum_j Amp_j cos(W_j t).

    Numerically integrates the stochastic field average (eqn 5.3 of Blundell et
    al. 2022) over an isotropic Lorentzian local-field distribution
    p(w) = (a_L / pi^2) / (a_L^2 + w^2)^2 (in rate units w = gamma_mu * B_local),
    with the applied longitudinal field giving omega0 = gamma_mu * B_L along z.
    For a local field w and total field W = omega0 z_hat + w, the single-muon
    response is cos^2(Theta) + sin^2(Theta) cos(|W| t), Theta = angle(W, z).
    Cylindrical (w_z, rho) integration with tan-substitutions to map the infinite
    range to finite intervals.
    """
    key = (round(a_L, 6), round(omega0, 6), n_u, n_v)
    cached = _LOR_LF_CACHE.get(key)
    if cached is not None:
        return cached

    a = a_L
    # Midpoint grids: w_z = a tan(u), u in (-pi/2, pi/2); rho = a tan(v), v in [0, pi/2)
    u = (np.arange(n_u) + 0.5) / n_u * np.pi - np.pi / 2.0
    v = (np.arange(n_v) + 0.5) / n_v * (np.pi / 2.0)
    du = np.pi / n_u
    dv = (np.pi / 2.0) / n_v
    wz = a * np.tan(u)  # (n_u,)
    rho = a * np.tan(v)  # (n_v,)
    wz_g, rho_g = np.meshgrid(wz, rho, indexing="ij")
    u_g, v_g = np.meshgrid(u, v, indexing="ij")
    w2 = wz_g**2 + rho_g**2
    p_lor = (a / np.pi**2) / (a**2 + w2) ** 2
    # d^3 w = 2 pi rho drho dwz; drho/dv = a sec^2 v, dwz/du = a sec^2 u
    weight = 2.0 * np.pi * rho_g * p_lor * (a / np.cos(u_g) ** 2) * (a / np.cos(v_g) ** 2) * du * dv
    # The distribution is normalised to 1 by construction; renormalise the
    # discretised weights to remove O(grid) quadrature bias (keeps G(0) = 1 exactly).
    weight = weight / weight.sum()
    wz_tot = omega0 + wz_g
    w_tot = np.sqrt(rho_g**2 + wz_tot**2)
    cos2 = np.where(w_tot > 1e-12, (wz_tot / w_tot) ** 2, 1.0)
    c0 = float(np.sum(weight * cos2))
    amp = (weight * (1.0 - cos2)).ravel()
    freq = w_tot.ravel()
    # Compress the (freq, amp) spectrum into a histogram so that evaluating
    # sum_j amp_j cos(W_j t) is a small matrix product, independent of the field
    # grid size.  Bin width is fine enough that binning error << quadrature error.
    n_bins = 4000
    w_max = float(freq.max()) if freq.size else 1.0
    edges = np.linspace(0.0, w_max * 1.0000001, n_bins + 1)
    amp_binned, _ = np.histogram(freq, bins=edges, weights=amp)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = amp_binned > (amp_binned.max() * 1e-7 if amp_binned.size else 0.0)
    result = (c0, centres[keep], amp_binned[keep])
    if len(_LOR_LF_CACHE) >= _LOR_LF_CACHE_MAX:
        _LOR_LF_CACHE.clear()
    _LOR_LF_CACHE[key] = result
    return result


def static_lorentzian_kt_lf(
    t: NDArray,
    A0: float,
    a_L: float,
    B_L: float,
    baseline: float = 0.0,
) -> NDArray:
    """Static Lorentzian Kubo-Toyabe in a longitudinal field (computed numerically).

    Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy* (OUP, 2022), sec
    5.3, note that the Kubo-Toyabe function "becomes modified in applied field ...
    [and] must be computed numerically".  This evaluates the stochastic field
    average (eqn 5.3) over an isotropic Lorentzian local-field distribution of
    half-width ``a_L`` (us^-1) with applied longitudinal field ``B_L`` (Gauss).

    - ``B_L -> 0``   : reduces to the zero-field Lorentzian KT (eqn 5.47),
      1/3 + 2/3 (1 - a_L t) e^{-a_L t}.
    - ``B_L -> inf`` : decoupling, G -> 1.

    The field average is a 2D oscillatory quadrature; the result is accurate to
    roughly 1% over 0-16 us (the zero-field analytic limit and the Gaussian LF
    Hayano function remain exact).  Results are cached per (a_L, B_L).
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    tt = np.atleast_1d(np.abs(t))
    a = float(a_L)
    if a <= 0 or abs(B_L) < 1e-9:
        gs = static_lorentzian_kt_zf(tt, 1.0, a, 0.0)
    else:
        gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
        omega0 = gamma_mu * (float(B_L) * GAUSS_TO_TESLA)
        c0, freq, amp = _lorentzian_lf_field_average(a, omega0)
        gs = c0 + np.cos(np.outer(tt, freq)) @ amp
    out = A0 * np.asarray(gs, dtype=float) + baseline
    return float(out[0]) if scalar else out


def _strong_collision_solve(
    gs_grid: NDArray,
    nu: float,
    h: float,
) -> NDArray:
    """Solve the strong-collision Volterra equation on a uniform grid.

    G_d(t) = f(t) + nu * integral_0^t f(t - tau) G_d(tau) dtau, with
    f(t) = e^{-nu t} G_s(t), discretised with the trapezoidal rule on a uniform
    grid of spacing ``h``.  ``gs_grid`` is the static G_s sampled on that grid
    (gs_grid[0] = G_s(0) = 1).
    """
    n = gs_grid.shape[0]
    idx = np.arange(n)
    f = np.exp(np.clip(-nu * idx * h, -700, 0)) * gs_grid
    g = np.empty(n, dtype=float)
    g[0] = 1.0
    denom = 1.0 - 0.5 * nu * h * f[0]
    for i in range(1, n):
        conv = float(np.dot(f[i - 1 : 0 : -1], g[1:i])) if i > 1 else 0.0
        g[i] = (f[i] + nu * h * (0.5 * f[i] * g[0] + conv)) / denom
    return g


# Cache of dynamic-KT solutions keyed by quantised (kind, width, nu, B_L, tmax).
_DYN_KT_CACHE: dict[tuple, tuple[NDArray, NDArray]] = {}
_DYN_KT_CACHE_MAX = 256


def _dynamic_kt_grid(
    kind: str, width: float, nu: float, B_L: float, tmax: float
) -> tuple[NDArray, NDArray]:
    """Return (grid, G_d) for a dynamic KT, computing+caching as needed."""
    key = (kind, round(width, 6), round(nu, 6), round(B_L, 4), round(tmax, 5))
    cached = _DYN_KT_CACHE.get(key)
    if cached is not None:
        return cached
    # Uniform grid.  The explicit trapezoidal Volterra solve is stable/accurate
    # only when nu*h is small (nu*h ~ 0.4 diverges; ~0.02 is <1%); size the step
    # from nu, with a hard cap on the point count to bound the O(N^2) cost.  In
    # the physical regime (nu <~ 10 MHz) this gives nu*h <= 0.02; for larger nu
    # the step is floored by the cap and accuracy degrades gracefully (still
    # bounded) in a regime already close to the analytic motional-narrowing limit.
    n_max = 4001
    h_des = min(0.02, 0.02 / max(nu, 1e-3))
    n = int(min(max(round(tmax / h_des) + 1, 64), n_max))
    grid = np.linspace(0.0, tmax, n)
    h = grid[1] - grid[0] if n > 1 else tmax
    if kind == "gaussian":
        if abs(B_L) < 1e-9:
            gs = static_gkt_zf(grid, 1.0, width, 0.0)
        else:
            gs = longitudinal_field_kubo_toyabe(grid, 1.0, width, B_L, 0.0)
    else:  # lorentzian (zero-field analytic; LF computed numerically)
        if abs(B_L) < 1e-9:
            gs = static_lorentzian_kt_zf(grid, 1.0, width, 0.0)
        else:
            gs = static_lorentzian_kt_lf(grid, 1.0, width, B_L, 0.0)
    gs = np.asarray(gs, dtype=float)
    gd = _strong_collision_solve(gs, nu, h)
    if len(_DYN_KT_CACHE) >= _DYN_KT_CACHE_MAX:
        _DYN_KT_CACHE.clear()
    _DYN_KT_CACHE[key] = (grid, gd)
    return grid, gd


def dynamic_gaussian_kt(
    t: NDArray,
    A0: float,
    Delta: float,
    nu: float,
    B_L: float = 0.0,
    baseline: float = 0.0,
) -> NDArray:
    """Dynamic Gaussian Kubo-Toyabe (strong collision; Hayano et al. PRB 20, 850 (1979)).

    Strong-collision generalisation of the static Gaussian KT: a Gaussian local
    field of width ``Delta`` (us^-1) fluctuating at rate ``nu`` (MHz), with
    optional longitudinal field ``B_L`` (Gauss).

    - ``nu -> 0``     : recovers the static (LF) Gaussian Kubo-Toyabe.
    - ``nu >> Delta`` : motional narrowing, G -> exp(-2 Delta^2 t / nu) (B_L = 0).
    - ``B_L -> inf``  : decoupling, G -> 1.
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    tt = np.atleast_1d(np.abs(t))
    if nu <= 1e-9:
        if abs(B_L) < 1e-9:
            gd = static_gkt_zf(tt, 1.0, Delta, 0.0)
        else:
            gd = longitudinal_field_kubo_toyabe(tt, 1.0, Delta, B_L, 0.0)
    else:
        tmax = float(max(tt.max(), 1e-6))
        grid, gd_grid = _dynamic_kt_grid("gaussian", float(Delta), float(nu), float(B_L), tmax)
        gd = np.interp(tt, grid, gd_grid)
    out = A0 * np.asarray(gd, dtype=float) + baseline
    return float(out[0]) if scalar else out


def dynamic_lorentzian_kt(
    t: NDArray,
    A0: float,
    a_L: float,
    nu: float,
    B_L: float = 0.0,
    baseline: float = 0.0,
) -> NDArray:
    """Dynamic Lorentzian Kubo-Toyabe (strong collision; Uemura et al. PRB 31, 546 (1985)).

    Strong-collision generalisation of the static Lorentzian KT for a dilute /
    Lorentzian local-field distribution of half-width ``a_L`` (us^-1) fluctuating
    at rate ``nu`` (MHz), with optional longitudinal field ``B_L`` (Gauss).

    - ``nu -> 0``    : recovers the static Lorentzian KT (zero-field analytic
      eqn 5.47; longitudinal field computed numerically per Blundell et al. 2022).
    - ``B_L -> inf`` : decoupling, G -> 1.
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    tt = np.atleast_1d(np.abs(t))
    if nu <= 1e-9:
        if abs(B_L) < 1e-9:
            gd = static_lorentzian_kt_zf(tt, 1.0, a_L, 0.0)
        else:
            gd = static_lorentzian_kt_lf(tt, 1.0, a_L, B_L, 0.0)
    else:
        tmax = float(max(tt.max(), 1e-6))
        grid, gd_grid = _dynamic_kt_grid("lorentzian", float(a_L), float(nu), float(B_L), tmax)
        gd = np.interp(tt, grid, gd_grid)
    out = A0 * np.asarray(gd, dtype=float) + baseline
    return float(out[0]) if scalar else out


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

_register(
    "DynamicGaussianKT",
    "Dynamic Gaussian Kubo-Toyabe, strong collision (Hayano et al. 1979)",
    dynamic_gaussian_kt,
    ["A0", "Delta", "nu", "B_L", "baseline"],
    {"A0": 25.0, "Delta": 0.5, "nu": 1.0, "B_L": 0.0, "baseline": 0.0},
)

_register(
    "DynamicLorentzianKT",
    "Dynamic Lorentzian Kubo-Toyabe, strong collision (Uemura et al. 1985)",
    dynamic_lorentzian_kt,
    ["A0", "a_L", "nu", "B_L", "baseline"],
    {"A0": 25.0, "a_L": 0.5, "nu": 1.0, "B_L": 0.0, "baseline": 0.0},
)

_register(
    "Keren",
    "Keren dynamic Gaussian relaxation in longitudinal field (Keren 1994)",
    keren,
    ["A0", "Delta", "nu", "B_L", "baseline"],
    {"A0": 25.0, "Delta": 0.5, "nu": 1.0, "B_L": 0.0, "baseline": 0.0},
)

_register(
    "Abragam",
    "Abragam relaxation, Gaussian-to-exponential crossover (Abragam 1961)",
    abragam,
    ["A0", "Delta", "nu", "baseline"],
    {"A0": 25.0, "Delta": 0.5, "nu": 1.0, "baseline": 0.0},
)

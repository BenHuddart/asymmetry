"""Built-in μSR fit functions.

Each model is a callable ``f(t, **params) -> array`` plus metadata describing
its parameters.  Models are collected in the :data:`MODELS` registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import integrate
from scipy.special import erfcx, j0, sici

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

# When the Larmor rate of the applied longitudinal field is below this fraction of
# the local-field width (omega0 < ratio * width) the field is negligible and the
# Kubo-Toyabe functions use their exact zero-field form, avoiding the
# ill-conditioned 1/omega0 prefactors of the longitudinal-field expressions.
_FIELD_DECOUPLING_RATIO = 0.05


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


def longitudinal_field_kubo_toyabe(
    t: NDArray,
    A0: float,
    Delta: float,
    B_L: float,
    baseline: float = 0.0,
) -> NDArray:
    r"""Static Gaussian Kubo-Toyabe relaxation in a longitudinal field.

    The longitudinal-field depolarisation function for a muon in a static,
    isotropic **Gaussian** distribution of local fields of width ``Delta`` with an
    applied longitudinal decoupling field ``B_L``.  As ``B_L`` is swept through the
    decoupling crossover the polarisation recovers toward unity, which is the
    experimental signature of a *static* (rather than dynamic) local field.

    .. math::

        G_z(t) = 1 - \frac{2\Delta^2}{\omega_0^2}
                 \left[1 - e^{-\Delta^2 t^2/2}\cos(\omega_0 t)\right]
               + \frac{2\Delta^4}{\omega_0^3}
                 \int_0^t e^{-\Delta^2\tau^2/2}\sin(\omega_0\tau)\,d\tau ,

    with :math:`\omega_0 = \gamma_\mu B_L` (``B_L`` in Gauss, converted to Tesla
    internally).  The returned asymmetry is :math:`A(t) = A_0\,G_z(t) + baseline`.
    For ``B_L = 0`` it reduces exactly to the zero-field Gaussian Kubo-Toyabe
    function (:func:`static_gkt_zf`); for large ``B_L`` it tends to 1 (decoupling).

    Parameters
    ----------
    t : NDArray
        Time values in microseconds.
    A0 : float
        Initial asymmetry amplitude at ``t = 0``.
    Delta : float
        Static Gaussian field-distribution width in us^-1 (Delta = gamma_mu * sqrt(<B^2>)).
    B_L : float
        Applied longitudinal magnetic field in Gauss.
    baseline : float, optional
        Constant additive baseline.

    Notes
    -----
    The oscillatory-decaying integral term is evaluated for all requested times at
    once by **cumulative trapezoidal integration on a shared fine grid** whose step
    is sized from ``omega0`` and ``Delta``.  This is both faster and smoother than
    per-point adaptive quadrature (which left the integral noisy), so it speeds up
    the static component and the dynamic Kubo-Toyabe that builds on it.

    References
    ----------
    R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki and R. Kubo,
    "Zero- and low-field spin relaxation studied by positive muons",
    Phys. Rev. B 20, 850 (1979).
    """
    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    omega0 = gamma_mu * (float(B_L) * GAUSS_TO_TESLA)

    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    tt = np.atleast_1d(np.abs(t))  # depolarisation is even in t
    delta = float(Delta)
    dt2 = (delta * tt) ** 2
    exp_term = np.exp(np.clip(-dt2 / 2.0, -700, 0))

    # When the applied field is negligible compared with the local-field width
    # (omega0 << Delta) the longitudinal correction is sub-percent, and the Hayano
    # expression becomes ill-conditioned (the 2*Delta^2/omega0^2 prefactor amplifies
    # floating-point cancellation).  Use the exact zero-field limit there.
    if abs(omega0) < max(1e-10, _FIELD_DECOUPLING_RATIO * delta) or delta <= 0.0 or tt.size == 0:
        # Zero-field (or zero-width / empty) limit: exact analytic Gaussian KT.
        gz = 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - dt2) * exp_term
    else:
        tmax = float(tt.max())
        if tmax <= 0.0:
            gz = np.ones_like(tt)
        else:
            # Step resolves the faster of the omega0 oscillation and the Gaussian
            # envelope; point count capped to bound cost at very high field (where
            # the 1/omega0^3 integral term is negligible anyway).
            h = min(0.01, 0.25 / max(abs(omega0), 1e-9), 0.1 / max(delta, 1e-9))
            n = int(min(max(round(tmax / h) + 1, 64), 200000))
            tau = np.linspace(0.0, tmax, n)
            integrand = np.exp(-0.5 * (delta * tau) ** 2) * np.sin(omega0 * tau)
            integral = integrate.cumulative_trapezoid(integrand, tau, initial=0.0)
            i_t = np.interp(tt, tau, integral)
            factor1 = 2.0 * delta**2 / omega0**2
            factor2 = 2.0 * delta**4 / omega0**3
            gz = 1.0 - factor1 * (1.0 - exp_term * np.cos(omega0 * tt)) + factor2 * i_t

    output = A0 * gz + baseline
    return float(output[0]) if scalar_input else output


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

    P(t) = A0 exp[-Gamma(t)] + baseline, with omega0 = gamma_mu * B_L and::

        Gamma(t) = (2 Delta^2 / (omega0^2 + nu^2)^2) * [
            (omega0^2 + nu^2) nu t + (omega0^2 - nu^2) (1 - e^{-nu t} cos(omega0 t))
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


# Cache of (time grid, static Lorentzian-LF line shape) keyed by quantised
# (a_L, omega0, tmax, grid sizes).
_LOR_LF_CACHE: dict[tuple, tuple[NDArray, NDArray]] = {}
_LOR_LF_CACHE_MAX = 64


def _lorentzian_lf_lineshape(a_L: float, omega0: float, t: NDArray, n_w: int = 2400) -> NDArray:
    """Static Lorentzian-LF line shape by the analytic angular + time average.

    The stochastic field average (eqn 5.3 of Blundell et al. 2022) over an
    isotropic Lorentzian local-field distribution reduces, after doing the
    angular average and the precession integral analytically, to a single smooth
    1-D quadrature over the local-field magnitude ``w``:

        G(t) = integral_0^inf f(w) [ A_cos(w) + B_sin(w, t) ] dw,

    with the magnitude distribution f(w) = (4 a_L/pi) w^2 / (a_L^2 + w^2)^2 and,
    for omega0 = gamma_mu * B_L, c = omega0^2 - w^2, W_lo = |omega0 - w|,
    W_hi = omega0 + w:

        A_cos(w) = [ (W_hi^4 - W_lo^4)/4 + c (W_hi^2 - W_lo^2) + c^2 ln(W_hi/W_lo) ]
                   / (8 omega0^3 w)
        B_sin(w, t) = [ ((omega0^2+w^2)/(2 omega0^2)) I1 - I3/(4 omega0^2)
                        - (c^2/(4 omega0^2)) (Ci(W_hi t) - Ci(W_lo t)) ] / (2 omega0 w)

    where I1 = integral W cos(Wt) dW and I3 = integral W^3 cos(Wt) dW over
    [W_lo, W_hi] (elementary), and Ci is the cosine integral.  Avoiding any
    frequency-domain truncation, this captures the e^{-a_L t} cusp exactly and is
    accurate to better than ~0.1-0.3 % for B_L >~ 20 G (finer for larger fields).
    """
    a = float(a_L)
    w0 = float(omega0)
    s = (np.arange(n_w) + 0.5) / n_w * (np.pi / 2.0)
    w = a * np.tan(s)  # local-field magnitude grid on (0, inf)
    f_w = (4.0 * a / np.pi) * w**2 / (a**2 + w**2) ** 2
    wq = f_w * (a / np.cos(s) ** 2) * ((np.pi / 2.0) / n_w)  # f(w) * quadrature weight
    c = w0**2 - w**2
    w_lo = np.abs(w0 - w)
    w_hi = w0 + w
    coef_w = (w0**2 + w**2) / (2.0 * w0**2)
    pref = 1.0 / (2.0 * w0 * w)
    near = np.abs(w - w0) < 1e-9  # c -> 0 and W_lo -> 0 together; the c^2 terms vanish
    ln_ratio = np.where(
        near, 0.0, np.log(np.clip(w_hi / np.clip(w_lo, 1e-300, None), 1e-300, None))
    )
    a_cos = (1.0 / (8.0 * w0**3 * w)) * (
        (w_hi**4 - w_lo**4) / 4.0 + c * (w_hi**2 - w_lo**2) + c**2 * ln_ratio
    )

    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    pos = t > 1e-12
    if np.any(pos):
        tp = t[pos][None, :]  # (1, n_t)
        whi = w_hi[:, None]
        wlo = w_lo[:, None]

        def _f1(wv: NDArray) -> NDArray:
            return (np.cos(wv * tp) + wv * tp * np.sin(wv * tp)) / tp**2

        def _f3(wv: NDArray) -> NDArray:
            return (
                (wv**3 / tp) * np.sin(wv * tp)
                + (3.0 * wv**2 / tp**2) * np.cos(wv * tp)
                - (6.0 * wv / tp**3) * np.sin(wv * tp)
                - (6.0 / tp**4) * np.cos(wv * tp)
            )

        i1 = _f1(whi) - _f1(wlo)
        i3 = _f3(whi) - _f3(wlo)
        _, ci_hi = sici(whi * tp)
        _, ci_lo = sici(np.clip(wlo * tp, 1e-300, None))
        c_term = np.where(near[:, None], 0.0, (c[:, None] ** 2 / (4.0 * w0**2)) * (ci_hi - ci_lo))
        b_sin = pref[:, None] * (coef_w[:, None] * i1 - i3 / (4.0 * w0**2) - c_term)
        out[pos] = np.sum(wq[:, None] * (a_cos[:, None] + b_sin), axis=0)
    return out


def _static_lorentzian_lf_grid(
    a_L: float, omega0: float, tmax: float, n_t: int = 220
) -> tuple[NDArray, NDArray]:
    """Cached (grid, line shape) for the static Lorentzian-LF, for interpolation.

    The line shape is smooth, so it is evaluated on a modest grid and linearly
    interpolated onto the requested times; this bounds the cost of the (otherwise
    O(n_w * n_t)) ``sici`` evaluation while keeping the interpolation error
    negligible.
    """
    key = (round(a_L, 6), round(omega0, 6), round(tmax, 5), n_t)
    cached = _LOR_LF_CACHE.get(key)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, tmax, n_t)
    gs = _lorentzian_lf_lineshape(a_L, omega0, grid)
    if len(_LOR_LF_CACHE) >= _LOR_LF_CACHE_MAX:
        # Evict the oldest entry rather than wiping the whole cache.
        _LOR_LF_CACHE.pop(next(iter(_LOR_LF_CACHE)))
    _LOR_LF_CACHE[key] = (grid, gs)
    return grid, gs


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
    half-width ``a_L`` (us^-1) with applied longitudinal field ``B_L`` (Gauss),
    via the analytic angular + time reduction in :func:`_lorentzian_lf_lineshape`.

    - ``B_L -> 0``   : reduces to the zero-field Lorentzian KT (eqn 5.47),
      1/3 + 2/3 (1 - a_L t) e^{-a_L t}.
    - ``B_L -> inf`` : decoupling, G -> 1.

    Accurate to better than ~0.1-0.3 % for B_L >~ 20 G (finer at higher field);
    very small fields (omega0 < 0.05 a_L) are treated as zero field, where the
    eqn 5.47 form is exact. The line shape is computed once on a grid and
    interpolated, and cached per (a_L, B_L, tmax).
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    tt = np.atleast_1d(np.abs(t))
    a = float(a_L)
    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    omega0 = gamma_mu * (float(B_L) * GAUSS_TO_TESLA)
    if a <= 0 or omega0 < _FIELD_DECOUPLING_RATIO * max(a, 1e-12):
        # Field negligible vs the distribution width: indistinguishable from ZF,
        # where the analytic eqn 5.47 form is exact (and avoids the ill-conditioned
        # small-omega0 limit of the longitudinal-field reduction).
        gs = static_lorentzian_kt_zf(tt, 1.0, a, 0.0)
    else:
        tmax = float(max(tt.max(), 1e-6))
        grid, gs_grid = _static_lorentzian_lf_grid(a, omega0, tmax)
        gs = np.interp(tt, grid, gs_grid)
    out = A0 * np.asarray(gs, dtype=float) + baseline
    return float(out[0]) if scalar else out


def risch_kehr(t: NDArray, Gamma: float) -> NDArray:
    """Risch-Kehr relaxation for spin transport by 1D diffusion.

    G(t) = e^{Gamma t} erfc(sqrt(Gamma t))

    Relaxation of the muon (or muonium) polarisation when the depolarising
    agent diffuses in one dimension (e.g. a polaron on a conducting-polymer
    chain): the return probability of a 1D random walk gives a
    ``(pi Gamma t)^{-1/2}`` long-time tail instead of an exponential.

    Evaluated as ``erfcx(sqrt(Gamma t))`` (the scaled complementary error
    function), which is numerically stable for all ``Gamma t`` — no asymptotic
    branch switch is needed (WiMDA switches forms at ``Gamma t = 20``).
    ``Gamma`` is used as ``|Gamma|``; WiMDA's mirrored ``2 - G`` branch for
    negative rates is an unphysical fitting convenience and is not ported.

    References
    ----------
    R. Risch and K. W. Kehr, Phys. Rev. B 46, 5246 (1992).
    """
    gt = np.abs(float(Gamma)) * np.abs(np.asarray(t, dtype=float))
    return np.asarray(erfcx(np.sqrt(gt)), dtype=float)


def bessel_oscillation(t: NDArray, frequency: float, phase: float = 0.0) -> NDArray:
    """Zeroth-order Bessel oscillation J0(2 pi f t + phase).

    The polarisation produced by an incommensurate (spin-density-wave) field
    distribution: an Overhauser distribution of local fields between -B_1 and
    +B_1 gives P(t) = J0(gamma_mu B_1 t) (Blundell, De Renzi, Lancaster &
    Pratt, *Muon Spectroscopy*, OUP 2022, eqn 6.47), which at late times looks
    like a damped cosine with a -45 degree phase shift (eqn 6.48).  Here the
    field-distribution edge is parameterised as a frequency
    ``f = gamma_mu B_1 / 2 pi`` in MHz.
    """
    t = np.asarray(t, dtype=float)
    return np.asarray(j0(2.0 * np.pi * float(frequency) * t + float(phase)), dtype=float)


# Cache of Gaussian-broadened-KT curves keyed by quantised (Delta, B_L, width, tmax).
_GBKT_CACHE: dict[tuple, tuple[NDArray, NDArray]] = {}
_GBKT_CACHE_MAX = 128
_GBKT_NODES = 21


def gaussian_broadened_kt(
    t: NDArray,
    Delta: float,
    B_L: float,
    w_rel: float,
) -> NDArray:
    """Static Gaussian Kubo-Toyabe averaged over a Gaussian distribution of Delta.

    G(t) = integral dDelta' p(Delta') G_KT(t; Delta', B_L), with ``p`` a Gaussian
    of mean ``Delta`` and standard deviation ``w_rel * Delta`` (``w_rel`` is the
    *fractional* width).  Models disordered hosts where a single Kubo-Toyabe
    width is too sharp — the distribution of static widths fills in the dip and
    softens the 1/3-tail recovery (cf. the Gaussian-broadened Gaussian of
    Noakes & Kalvius, Phys. Rev. B 56, 2352 (1997); WiMDA's ``Gau broad KT``).

    Evaluated by Gauss-Hermite quadrature (21 nodes) over the Delta
    distribution, with negative quadrature widths reflected to ``|Delta'|`` (as
    in WiMDA).  ``w_rel = 0`` reduces exactly to the static (LF) Kubo-Toyabe.

    Note: WiMDA's ``rel width`` parameter enters as weight ``exp(-(i/7)^2)`` over
    ``Delta(1 + w i/7)``, i.e. a Gaussian of fractional standard deviation
    ``w/sqrt(2)``; ``w_rel`` here *is* the fractional standard deviation
    (``w_rel = w_WiMDA / sqrt(2)``).
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    tt = np.atleast_1d(np.abs(t))
    delta = abs(float(Delta))
    width = abs(float(w_rel))
    if width <= 1e-9 or delta <= 0.0:
        gz = longitudinal_field_kubo_toyabe(tt, 1.0, delta, B_L, 0.0)
        out = np.asarray(gz, dtype=float)
        return float(out[0]) if scalar else out

    tmax = float(max(tt.max(), 1e-6))
    key = (round(delta, 6), round(float(B_L), 4), round(width, 6), round(tmax, 5))
    cached = _GBKT_CACHE.get(key)
    if cached is None:
        nodes, weights = np.polynomial.hermite.hermgauss(_GBKT_NODES)
        grid = np.linspace(0.0, tmax, 800)
        acc = np.zeros_like(grid)
        for x, w in zip(nodes, weights, strict=True):
            delta_i = abs(delta * (1.0 + np.sqrt(2.0) * width * float(x)))
            acc += float(w) * np.asarray(
                longitudinal_field_kubo_toyabe(grid, 1.0, delta_i, B_L, 0.0), dtype=float
            )
        acc /= np.sqrt(np.pi)
        if len(_GBKT_CACHE) >= _GBKT_CACHE_MAX:
            _GBKT_CACHE.pop(next(iter(_GBKT_CACHE)))
        _GBKT_CACHE[key] = (grid, acc)
        cached = (grid, acc)
    grid, curve = cached
    out = np.interp(tt, grid, curve)
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


# Above this fluctuation rate (MHz) the explicit trapezoidal strong-collision
# solver is numerically unstable: the kernel e^{-nu t} G_s(t) decays within a few
# grid steps and the recursion amplifies roundoff (it diverges, not "degrades
# gracefully"), and refining the grid enough to stay stable is prohibitive.  The
# system is then deep in the fast-fluctuation regime, where the analytic
# motional-narrowing limit is accurate -- for the Gaussian case the Keren function
# matches a converged solver to < 0.5 % for nu >~ 6*Delta, which is satisfied at
# this crossover for any physical width.
_DYN_KT_NU_SWITCH = 12.0


def _dynamic_kt_grid(
    kind: str, width: float, nu: float, B_L: float, tmax: float
) -> tuple[NDArray, NDArray]:
    """Return (grid, G_d) for a dynamic KT, computing+caching as needed."""
    key = (kind, round(width, 6), round(nu, 6), round(B_L, 4), round(tmax, 5))
    cached = _DYN_KT_CACHE.get(key)
    if cached is not None:
        return cached

    if nu <= _DYN_KT_NU_SWITCH:
        # Slow/intermediate regime: strong-collision Volterra solve with a step
        # sized so that nu*h <= 0.02 (stable and < 1 % here).
        h_des = min(0.02, 0.02 / max(nu, 1e-3))
        n = int(min(max(round(tmax / h_des) + 1, 64), 20001))
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
        gd = _strong_collision_solve(np.asarray(gs, dtype=float), nu, h)
    elif kind == "gaussian":
        # Fast-fluctuation Gaussian: the Keren function is the analytic motional-
        # narrowing limit (rate ~2*Delta^2/nu), accurate to <0.5% here and bounded.
        grid = np.linspace(0.0, tmax, 800)
        gd = keren(grid, 1.0, width, nu, B_L)
    else:
        # Fast-fluctuation Lorentzian: the relaxation rate saturates (it is
        # ~independent of nu, since a Lorentzian distribution has no finite second
        # moment), so reuse the stable solver evaluated at the crossover rate --
        # continuous across the switch and physically correct.
        nu_eff = _DYN_KT_NU_SWITCH
        h_des = min(0.02, 0.02 / nu_eff)
        n = int(min(max(round(tmax / h_des) + 1, 64), 20001))
        grid = np.linspace(0.0, tmax, n)
        h = grid[1] - grid[0] if n > 1 else tmax
        if abs(B_L) < 1e-9:
            gs = static_lorentzian_kt_zf(grid, 1.0, width, 0.0)
        else:
            gs = static_lorentzian_kt_lf(grid, 1.0, width, B_L, 0.0)
        gd = _strong_collision_solve(np.asarray(gs, dtype=float), nu_eff, h)

    if len(_DYN_KT_CACHE) >= _DYN_KT_CACHE_MAX:
        # Evict the oldest entry (dicts are insertion-ordered) rather than wiping
        # the whole cache, which would force a full recompute on the next call.
        _DYN_KT_CACHE.pop(next(iter(_DYN_KT_CACHE)))
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

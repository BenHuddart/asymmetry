"""Built-in μSR fit functions.

Each model is a callable ``f(t, **params) -> array`` plus metadata describing
its parameters.  Models are collected in the :data:`MODELS` registry.

Scale
-----
Every asymmetry-valued parameter of every model here — the amplitudes ``A0`` /
``A`` and the additive ``baseline`` — is **in percent (0–100)**, and so is the
curve each model returns.  That is the WiMDA-style convention
:attr:`asymmetry.core.data.dataset.MuonDataset.asymmetry` uses, which is why the
seeds in :data:`MODELS` default ``A0`` to ``25`` rather than ``0.25``, and why
``PARAM_INFO_REGISTRY`` gives ``A``/``A0``/``A_bg`` the unit string ``"%"``.
Fitting these models against fraction-scale data (what
:func:`asymmetry.core.transform.compute_asymmetry` and
:func:`asymmetry.core.transform.binned_fb_asymmetry` return) converges to the
wrong minimum and trips
:class:`~asymmetry.core.fitting.engine.AsymmetryScaleWarning`; scale the data with
:func:`asymmetry.core.transform.units.to_percent` first.  The polarization-shape
functions that carry no amplitude (:func:`risch_kehr`,
:func:`bessel_oscillation`, and the ``P(t)`` kernels in the muonium /
muon-fluorine / nuclear-dipole modules) are unit-normalised and therefore
scale-free.  See "Asymmetry units across the API" in the documentation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import integrate
from scipy.linalg import solve_triangular, toeplitz
from scipy.special import erfcx, j0, sici

from asymmetry.core.fitting.parameters import ParamInfo, param_info_map
from asymmetry.core.fitting.registration import insert_definition
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

#: Groundwork for the deferred variable-projection (VarPro) follow-up. VarPro is
#: not wired into a fit path yet (see the PR body: it is a constant-factor win
#: once profiled locals already separate the per-dataset solves, and its marginal
#: errors need a final full Hessian), so this metadata is presently consumed only
#: by tests and the follow-up plan — but it is the generic, name-role marking the
#: study asks for, kept here so the follow-up does not re-derive it.
#:
#: Parameter *role* names that enter a μSR model **linearly** — amplitudes and
#: additive constant backgrounds. These are roles, not component names: an
#: amplitude ``A`` scales its term and a constant baseline ``baseline`` adds to
#: the model, so both appear to first order as ``model = a·term + ...``. Variable
#: projection (VarPro) solves such parameters by linear least-squares inside the
#: objective instead of by Minuit. The set is deliberately generic (matched by
#: name, never by which component a parameter belongs to); an ``f_<Component>``
#: fraction weight is *not* here because a normalised fraction group is not a free
#: linear scale. VarPro always *verifies* affineness numerically at runtime and
#: falls back to the nonlinear treatment when a flagged name is not actually
#: affine in a given model, so this list only has to be a safe over-approximation
#: of "usually linear".
LINEAR_PARAM_ROLE_NAMES: frozenset[str] = frozenset(
    {
        "A",
        "A0",
        "A_bg",
        "a_Dia",
        "baseline",
        "bg",
        "BG",
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
    }
)


def default_linear_params(param_names: list[str]) -> list[str]:
    """Role-based linear-parameter guess for a model's parameter list.

    Returns the subset of ``param_names`` whose (index-stripped) base name is a
    known linear role (:data:`LINEAR_PARAM_ROLE_NAMES`) — i.e. an amplitude or an
    additive constant background. This is only a *candidate* set: VarPro verifies
    affineness numerically before using it, so a false positive is corrected at
    runtime, never silently trusted.
    """
    from asymmetry.core.fitting.parameters import split_parameter_name

    out: list[str] = []
    for name in param_names:
        base, _ = split_parameter_name(name)
        if base in LINEAR_PARAM_ROLE_NAMES:
            out.append(name)
    return out


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
    #: Names of parameters that enter the model linearly (amplitudes, constant
    #: backgrounds). ``None`` means "derive from role names via
    #: :func:`default_linear_params`". Variable projection solves these by linear
    #: least-squares, but only after a numeric affineness check, so an entry here
    #: is a hint, not a guarantee.
    linear_params: list[str] | None = None

    def resolved_linear_params(self) -> list[str]:
        """The linear-parameter set for this model (explicit or role-derived)."""
        if self.linear_params is not None:
            return list(self.linear_params)
        return default_linear_params(self.param_names)


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


def _bounded_cache_get(
    cache: dict, max_size: int, key: tuple, compute: Callable[[], tuple]
) -> tuple:
    """Return ``cache[key]``, computing and inserting it on a miss.

    When full, the *oldest* entry is evicted (dicts are insertion-ordered)
    rather than clearing the cache, which would force a full recompute on the
    next call.  Shared by the grid caches of the static Lorentzian-LF line
    shape and the dynamic Kubo-Toyabe family, so cache-policy fixes (eviction,
    thread-safety) happen in one place.
    """
    cached = cache.get(key)
    if cached is None:
        cached = compute()
        if len(cache) >= max_size:
            cache.pop(next(iter(cache)))
        cache[key] = cached
    return cached


# Cache of (uniform time grid, static Lorentzian-LF line shape) keyed by
# quantised (a_L, omega0, tmax).
_LOR_LF_CACHE: dict[tuple, tuple[NDArray, NDArray]] = {}
_LOR_LF_CACHE_MAX = 64

# Longest spectral FFT the longitudinal-field Lorentzian line shape will use.
# The frequency sampling step must resolve the strip of analyticity of the
# spectral density, whose half-width is a_L, so the sample count grows like
# 1/a_L; below a_L ~ 1e-4 us^-1 the cap is reached and the aliasing error grows
# instead of the cost -- harmless there, because a field that exceeds the
# distribution width by four orders of magnitude has decoupled the muon and the
# line shape is unity to within the same 1e-4.
_LOR_LF_FFT_CAP = 1 << 20
# Period margin, in units of 1/a_L, that the spectral sampling adds beyond the
# requested time window: the transform decays like exp(-a_L t), so 30/a_L puts
# the aliased copy at exp(-30) ~ 1e-13.
_LOR_LF_ALIAS_MARGIN = 30.0
# Larmor phase per time-grid step for the cached static line shape: linear
# interpolation of a cos(omega0 t) oscillation from a grid of step h errs by
# (omega0 h)^2 / 8 of its amplitude, ~1e-4 here.
_LOR_LF_STATIC_STEP_RAD = 0.03


def _lorentzian_lf_uniform(a_L: float, omega0: float, h: float, n: int) -> NDArray:
    """Static Lorentzian-LF line shape on the uniform grid ``t_i = i h``, ``i < n``.

    The stochastic field average (eqn 5.3 of Blundell et al. 2022) over an
    isotropic Lorentzian local-field distribution of half-width ``a`` in a
    longitudinal field of Larmor rate ``omega0`` is

        G(t) = integral_0^inf dw f(w) integral_-1^1 (dmu/2)
               [cos^2(alpha) + sin^2(alpha) cos(W t)],

    with W = |omega0 z + w n| the total-field magnitude and alpha its angle to
    z.  Changing the angular variable to W (W dW = omega0 w dmu) and swapping
    the order of the two integrals turns the time-dependent part into a
    cosine transform of a **spectral density** that integrates in closed form,

        G(t) = 1 - integral_0^inf S(W) dW + integral_0^inf S(W) cos(W t) dW,
        S(W) = a / (2 pi omega0^3 W)
               [ (a^2 + W^2 + omega0^2) ln( (a^2 + (W + omega0)^2)
                                            / (a^2 + (W - omega0)^2) )
                 - 4 omega0 W ],

    for f(w) = (4a/pi) w^2 / (a^2 + w^2)^2.  Its large-W expansion is
    (8a/3pi) [1/W^2 - (2a^2 - 2 omega0^2/5)/W^4 + ...]: the 1/W^2 tail is
    exactly the zero-field density (8a/3pi) W^2/(W^2 + a^2)^2, whose transform
    is the analytic (2/3)(1 - a t) e^{-a t} carrying the e^{-a t} cusp, and
    the 1/W^4 term is removed with c4/(W^2 + kappa^2)^2, c4 = 16 a omega0^2
    / (15 pi), whose transform is (pi c4 / 4 kappa^3)(1 + kappa t) e^{-kappa t}.
    The remainder is even, O(W^-6), and analytic in the strip |Im W| < a, so
    the trapezoidal rule on a uniform W grid -- one real FFT, with the grid
    chosen so its conjugate time step is exactly ``h`` -- converges
    exponentially in the step (validated against the original quadrature
    :func:`_lorentzian_lf_lineshape_reference` to that reference's own
    accuracy, and self-consistent to ~1e-12 under grid refinement).  The
    constant term is fixed by G(0) = 1.
    """
    a = float(a_L)
    w0 = float(omega0)
    n = int(n)
    if 2 * n > _LOR_LF_FFT_CAP:
        raise ValueError(
            f"Lorentzian-LF grid of {n} points needs a {2 * n}-point spectral FFT, "
            f"above the {_LOR_LF_FFT_CAP} cap; callers coarsen the grid instead."
        )
    t = float(h) * np.arange(n)
    g_zf = static_lorentzian_kt_zf(t, 1.0, a, 0.0)
    if n < 2:
        return np.asarray(g_zf, dtype=float)
    c4 = 16.0 * a * w0 * w0 / (15.0 * np.pi)
    kappa = float(np.hypot(a, w0))
    period = (n - 1) * h + _LOR_LF_ALIAS_MARGIN / a
    n_fft = 1 << int(np.ceil(np.log2(max(2 * n, period / h))))
    n_fft = max(min(n_fft, _LOR_LF_FFT_CAP), 1 << int(np.ceil(np.log2(2 * n))))
    d_w = 2.0 * np.pi / (n_fft * h)
    w = d_w * np.arange(1, n_fft)
    q = a * a + (w - w0) ** 2
    s = (a / (2.0 * np.pi * w0**3 * w)) * (
        (a * a + w * w + w0 * w0) * np.log1p(4.0 * w0 * w / q) - 4.0 * w0 * w
    )
    s -= (8.0 * a / (3.0 * np.pi)) * w * w / (w * w + a * a) ** 2
    s -= c4 / (w * w + kappa * kappa) ** 2
    # Even extension of length 2 n_fft: [S(0), s_1 .. s_{N-1}, S(W_N) ~ 0,
    # s_{N-1} .. s_1]; S(0) is the tail-subtraction value, the rest vanish at 0.
    ext = np.concatenate(([-c4 / kappa**4], s, [0.0], s[::-1]))
    spectrum = np.fft.rfft(ext).real * (0.5 * d_w)  # cos transform at t_k = k h / 2
    g_delta = spectrum[0 : 2 * n : 2]
    tail = c4 * (np.pi / (4.0 * kappa**3)) * ((1.0 + kappa * t) * np.exp(-kappa * t) - 1.0)
    return np.asarray(g_zf + (g_delta - g_delta[0]) + tail, dtype=float)


def _lorentzian_lf_lineshape_reference(
    a_L: float, omega0: float, t: NDArray, n_w: int = 2400
) -> NDArray:
    """Static Lorentzian-LF line shape by direct angular-average quadrature.

    This is the original implementation, kept as the validation reference for
    :func:`_lorentzian_lf_uniform` (``tests/core/test_dynamic_relaxation.py``).
    The stochastic field average (eqn 5.3 of Blundell et al. 2022) over an
    isotropic Lorentzian local-field distribution reduces, after doing the
    angular average and the precession integral analytically, to a single
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
    [W_lo, W_hi] (elementary), and Ci is the cosine integral.  The integrand
    has an integrable logarithmic singularity at w = omega0, so the midpoint
    rule on the tan-graded grid converges only slowly: ~1e-3 at the default
    ``n_w``, ~1e-4 at ``n_w = 48000``.
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

        # ``I1`` and ``I3`` are elementary antiderivatives of the same
        # oscillatory kernel, evaluated at the same two limits. Written
        # separately they cost four sin/cos passes over the (n_w, n_t) grid
        # instead of two, and re-derive tp**2..tp**4 four times over — and this
        # kernel is the dominant cost of every longitudinal-field Lorentzian
        # Kubo-Toyabe evaluation, hence of any wizard candidate carrying one.
        # Computing sin/cos once per limit and sharing the inverse powers of tp
        # is a pure arithmetic rearrangement: same expressions, same order of
        # accumulation, half the transcendental work.
        inv_tp = 1.0 / tp
        inv_tp2 = inv_tp * inv_tp
        inv_tp3 = inv_tp2 * inv_tp
        inv_tp4 = inv_tp2 * inv_tp2

        def _f1_f3(wv: NDArray) -> tuple[NDArray, NDArray]:
            x = wv * tp
            sin_x = np.sin(x)
            cos_x = np.cos(x)
            f1 = (cos_x + x * sin_x) * inv_tp2
            f3 = (wv**3 * inv_tp - 6.0 * wv * inv_tp3) * sin_x + (
                3.0 * wv**2 * inv_tp2 - 6.0 * inv_tp4
            ) * cos_x
            return f1, f3

        f1_hi, f3_hi = _f1_f3(whi)
        f1_lo, f3_lo = _f1_f3(wlo)
        i1 = f1_hi - f1_lo
        i3 = f3_hi - f3_lo
        _, ci_hi = sici(whi * tp)
        _, ci_lo = sici(np.clip(wlo * tp, 1e-300, None))
        c_term = np.where(near[:, None], 0.0, (c[:, None] ** 2 / (4.0 * w0**2)) * (ci_hi - ci_lo))
        b_sin = pref[:, None] * (coef_w[:, None] * i1 - i3 / (4.0 * w0**2) - c_term)
        out[pos] = np.sum(wq[:, None] * (a_cos[:, None] + b_sin), axis=0)
    return out


def _static_lorentzian_lf_grid(a_L: float, omega0: float, tmax: float) -> tuple[NDArray, NDArray]:
    """Cached (uniform grid, line shape) for the static Lorentzian-LF, for interpolation.

    The grid step resolves the Larmor oscillation (``_LOR_LF_STATIC_STEP_RAD``
    of phase per step, never coarser than 0.02 us), so linear interpolation
    onto the requested times errs by ~1e-4 of the oscillation amplitude at any
    field; the FFT evaluation costs ~O(n log n) in the grid length.
    """
    key = (round(a_L, 6), round(omega0, 6), round(tmax, 5))

    def _compute() -> tuple[NDArray, NDArray]:
        h = min(0.02, _LOR_LF_STATIC_STEP_RAD / max(abs(float(omega0)), 1e-12))
        n = int(np.ceil(float(tmax) / h)) + 1
        if 2 * n > _LOR_LF_FFT_CAP:
            # The FFT needs 2n samples; beyond the cap coarsen the grid instead
            # of growing the transform (only reachable for extreme B_L * tmax).
            n = _LOR_LF_FFT_CAP // 2
            h = float(tmax) / (n - 1)
        grid = h * np.arange(n)
        return grid, _lorentzian_lf_uniform(a_L, omega0, h, n)

    return _bounded_cache_get(_LOR_LF_CACHE, _LOR_LF_CACHE_MAX, key, _compute)


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
    via the closed-form spectral density of :func:`_lorentzian_lf_uniform`.

    - ``B_L -> 0``   : reduces to the zero-field Lorentzian KT (eqn 5.47),
      1/3 + 2/3 (1 - a_L t) e^{-a_L t}.
    - ``B_L -> inf`` : decoupling, G -> 1.

    The line shape is evaluated by the closed-form spectral density of
    :func:`_lorentzian_lf_uniform` (accurate to ~1e-6 at any field) on a grid
    that resolves the Larmor oscillation, linearly interpolated onto the
    requested times, and cached per (a_L, B_L, tmax); very small fields
    (omega0 < 0.05 a_L) are treated as zero field, where the eqn 5.47 form is
    exact.
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


# Gauss-Hermite quadrature for the Gaussian-broadened KT, computed at import.
_GBKT_NODES = 21
_GBKT_X, _GBKT_WEIGHTS = np.polynomial.hermite.hermgauss(_GBKT_NODES)
_GBKT_WN = _GBKT_WEIGHTS / np.sqrt(np.pi)  # normalised quadrature weights


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
    in WiMDA).  The quadrature is evaluated **directly at the requested times**
    (vectorised over the nodes, with the Hayano longitudinal-field integral
    computed once on a shared fine grid for all nodes), so the model is smooth
    in ``w_rel`` and ``w_rel = 0`` reduces exactly and continuously to the
    static (LF) Kubo-Toyabe.

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
    if width <= 0.0 or delta <= 0.0 or tt.size == 0:
        gz = longitudinal_field_kubo_toyabe(tt, 1.0, delta, B_L, 0.0)
        out = np.asarray(gz, dtype=float)
        return float(out[0]) if scalar else out

    deltas = np.abs(delta * (1.0 + np.sqrt(2.0) * width * _GBKT_X))  # (nodes,)
    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    omega0 = gamma_mu * (float(B_L) * GAUSS_TO_TESLA)

    dt2 = (deltas[:, None] * tt[None, :]) ** 2  # (nodes, n_t)
    exp_term = np.exp(np.clip(-dt2 / 2.0, -700, 0))

    # Per node, use the exact zero-field form when the applied field is
    # negligible against that node's width (same rule as the single-Delta
    # function); otherwise the Hayano longitudinal-field expression with the
    # oscillatory integral evaluated once on a grid shared by all LF nodes.
    zf_mask = np.abs(omega0) < np.maximum(1e-10, _FIELD_DECOUPLING_RATIO * deltas)
    gz_nodes = np.empty_like(dt2)
    gz_nodes[zf_mask] = 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - dt2[zf_mask]) * exp_term[zf_mask]

    lf_idx = np.nonzero(~zf_mask)[0]
    if lf_idx.size:
        tmax = float(max(tt.max(), 1e-6))
        delta_max = float(deltas[lf_idx].max())
        h = min(0.01, 0.25 / max(abs(omega0), 1e-9), 0.1 / max(delta_max, 1e-9))
        n = int(min(max(round(tmax / h) + 1, 64), 200000))
        tau = np.linspace(0.0, tmax, n)
        integrand = (
            np.exp(np.clip(-0.5 * (deltas[lf_idx, None] * tau[None, :]) ** 2, -700, 0))
            * np.sin(omega0 * tau)[None, :]
        )
        integral = integrate.cumulative_trapezoid(integrand, tau, axis=1, initial=0.0)
        cos_term = np.cos(omega0 * tt)
        for row, k in enumerate(lf_idx):
            d_k = deltas[k]
            i_t = np.interp(tt, tau, integral[row])
            factor1 = 2.0 * d_k**2 / omega0**2
            factor2 = 2.0 * d_k**4 / omega0**3
            gz_nodes[k] = 1.0 - factor1 * (1.0 - exp_term[k] * cos_term) + factor2 * i_t

    out = np.asarray(_GBKT_WN @ gz_nodes, dtype=float)
    return float(out[0]) if scalar else out


# Forward-substitution block size for _strong_collision_solve: cross-block
# coupling is an O(n log n) FFT convolution, within-block coupling a dense
# BxB triangular solve, so total cost is ~O(n log n + n B).  512 keeps both
# terms small while holding the FFT products at machine precision.
_STRONG_COLLISION_BLOCK = 512


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

    The trapezoidal discretisation is a unit-lower-triangular Toeplitz system
    for g[1:] (see ``_strong_collision_solve_reference`` for the equivalent
    scalar recursion):

        denom * g[i] - nu*h * sum_{j=1}^{i-1} f[i-j] * g[j] = f[i] * (1 + nu*h/2)

    with ``denom = 1 - nu*h*f[0]/2``.  It is solved by blocked forward
    substitution -- cross-block contributions via FFT convolution of the
    already-known g against the (bounded) kernel f, the within-block coupling
    via a dense triangular solve of the block's Toeplitz matrix -- for
    O(n log n + n B) cost in place of the O(n^2) scalar recursion.
    """
    n = gs_grid.shape[0]
    idx = np.arange(n)
    f = np.exp(np.clip(-nu * idx * h, -700, 0)) * gs_grid
    g = np.empty(n, dtype=float)
    g[0] = 1.0
    if n == 1:
        return g
    denom = 1.0 - 0.5 * nu * h * f[0]
    coef = 1.0 + 0.5 * nu * h
    block = _STRONG_COLLISION_BLOCK
    for start in range(1, n, block):
        end = min(start + block, n)
        bs = end - start
        rhs = f[start:end] * coef
        if start > 1:
            gk = g[1:start]
            fk = f[1:end]
            size = 1 << (gk.shape[0] + fk.shape[0] - 2).bit_length()
            conv = np.fft.irfft(np.fft.rfft(gk, size) * np.fft.rfft(fk, size), size)
            rhs = rhs + nu * h * conv[start - 2 : start - 2 + bs]
        col = np.empty(bs)
        col[0] = denom
        if bs > 1:
            col[1:] = -nu * h * f[1:bs]
        trow = np.zeros(bs)
        trow[0] = denom
        block_mat = toeplitz(col, trow)
        g[start:end] = solve_triangular(block_mat, rhs, lower=True, check_finite=False)
    return g


def _strong_collision_solve_reference(
    gs_grid: NDArray,
    nu: float,
    h: float,
) -> NDArray:
    """Scalar O(n^2) trapezoidal recursion, kept as the reference solution.

    This is the original, obviously-correct implementation of
    ``_strong_collision_solve``; the fast solver is validated against it in the
    tests (``tests/core/test_dynamic_relaxation.py``).
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


def _strong_collision_modes(
    A: NDArray, b: NDArray, c: NDArray, nu: float
) -> tuple[NDArray, NDArray]:
    """Exponential modes of the strong-collision solution for a static function
    with a finite state-space realisation.

    If the static polarisation is ``G_s(t) = c^T exp(A t) b`` for a small real
    matrix ``A`` and vectors ``b``, ``c`` -- true of any finite sum of
    exponentials, damped cosines and ``t^k e^{-at}`` terms -- then with
    ``x(t) = exp((A - nu I) t) b + nu * integral_0^t exp((A - nu I)(t - tau)) b
    G_d(tau) dtau`` the strong-collision equation

        G_d(t) = f(t) + nu * integral_0^t f(t - tau) G_d(tau) dtau,
        f(t) = e^{-nu t} G_s(t),

    is exactly the linear system ``x' = M x``, ``x(0) = b``, ``G_d = c^T x``
    with ``M = A - nu I + nu b c^T``: the fluctuations are a rank-one feedback
    of the observed polarisation into the initial state.  Hence
    ``G_d(t) = sum_k w_k exp(lambda_k t)`` over the eigenvalues ``lambda_k`` of
    ``M``, with ``w_k = (c^T v_k) (V^{-1} b)_k``.  Unlike the equivalent
    rational Laplace transform ``G_s~(s + nu) / (1 - nu G_s~(s + nu))`` this
    never expands a polynomial, so it stays well conditioned when ``nu``
    exceeds the static frequencies by many orders of magnitude (the fast modes
    are a non-defective cluster near ``-nu`` split linearly by ``A``).

    Returns ``(lambda, w)`` restricted to the modes with non-negative imaginary
    part, the weights of the complex pairs doubled, so that
    ``G_d(t) = Re sum_k w_k exp(lambda_k t)``; see :func:`_exponential_sum`.

    The eigendecomposition requires ``M`` to be diagonalisable.  A realisation
    carrying a Jordan block (the ``t e^{-at}`` term of the Lorentzian KT) is
    defective at exactly ``nu = 0``, where ``M = A``; any ``nu > 0`` splits the
    block (the split scales like ``sqrt(nu)``, and the modes are still
    well conditioned at ``nu = 1e-9``, see the tests).  Callers therefore
    evaluate the static ``nu = 0`` limit themselves rather than through this
    helper, as :func:`dynamic_lorentzian_kt` and
    :func:`~asymmetry.core.fitting.muon_fluorine.polarization.dynamic_fmuf_polarization`
    do.
    """
    a_mat = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    feedback = a_mat - float(nu) * np.eye(a_mat.shape[0]) + float(nu) * np.outer(b, c)
    lam, vectors = np.linalg.eig(feedback)
    weights = (c @ vectors) * np.linalg.solve(vectors, b.astype(complex))
    keep = lam.imag >= 0.0
    weights = np.where(lam.imag > 0.0, 2.0 * weights, weights)
    return lam[keep], weights[keep]


def _exponential_sum(t: NDArray, lam: NDArray, weights: NDArray) -> NDArray:
    """``Re sum_k w_k exp(lambda_k t)`` for ``t >= 0`` (any shape), see above."""
    return np.real(np.exp(np.multiply.outer(np.asarray(t, dtype=float), lam)) @ weights)


def _lorentzian_kt_zf_realisation(a_L: float) -> tuple[NDArray, NDArray, NDArray]:
    """State-space ``(A, b, c)`` of the static zero-field Lorentzian Kubo-Toyabe.

    ``G_s(t) = 1/3 + (2/3) e^{-a t} - (2a/3) t e^{-a t}``: a constant mode and a
    2x2 Jordan block for the ``t e^{-a t}`` term.
    """
    a = float(a_L)
    a_mat = np.array([[0.0, 0.0, 0.0], [0.0, -a, 1.0], [0.0, 0.0, -a]])
    b = np.array([1.0 / 3.0, 2.0 / 3.0, -2.0 * a / 3.0])
    c = np.array([1.0, 1.0, 0.0])
    return a_mat, b, c


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
# this crossover for any physical width.  (The zero-field Lorentzian case never
# reaches the solver: it has the exact closed form of
# :func:`_strong_collision_modes` at every rate.)
_DYN_KT_NU_SWITCH = 12.0


def _dynamic_kt_grid(
    kind: str, width: float, nu: float, B_L: float, tmax: float
) -> tuple[NDArray, NDArray]:
    """Return (grid, G_d) for a grid-solved dynamic KT, computing+caching as needed.

    Serves the Gaussian family and the longitudinal-field Lorentzian; the
    zero-field Lorentzian is evaluated in closed form by
    :func:`dynamic_lorentzian_kt` and never comes here.
    """
    key = (kind, round(width, 6), round(nu, 6), round(B_L, 4), round(tmax, 5))
    cached = _DYN_KT_CACHE.get(key)
    if cached is not None:
        return cached

    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    omega0 = abs(gamma_mu * (float(B_L) * GAUSS_TO_TESLA))
    if nu <= _DYN_KT_NU_SWITCH:
        # Slow/intermediate regime: strong-collision Volterra solve.
        nu_solve = float(nu)
    elif kind == "gaussian":
        # Fast-fluctuation Gaussian: the Keren function is the analytic motional-
        # narrowing limit (rate ~2*Delta^2/nu), accurate to <0.5% here and bounded.
        grid = np.linspace(0.0, tmax, 800)
        gd = keren(grid, 1.0, width, nu, B_L)
        if len(_DYN_KT_CACHE) >= _DYN_KT_CACHE_MAX:
            _DYN_KT_CACHE.pop(next(iter(_DYN_KT_CACHE)))  # see _bounded_cache_get
        _DYN_KT_CACHE[key] = (grid, gd)
        return grid, gd
    else:
        # Fast-fluctuation Lorentzian in a field: the relaxation rate saturates
        # (it is ~independent of nu, since a Lorentzian distribution has no
        # finite second moment -- the zero-field closed form shows the rate
        # tending to 4 a_L / 3), so reuse the stable solver evaluated at the
        # crossover rate: continuous across the switch and physically correct.
        nu_solve = _DYN_KT_NU_SWITCH

    # Step sized so that nu*h <= 0.02 (stable and < 1 % here) and so that the
    # static line shape's Larmor oscillation is resolved (<= 0.1 rad per step).
    h_des = min(0.02, 0.02 / max(nu_solve, 1e-3), 0.1 / max(omega0, 1e-3))
    n = int(min(max(round(tmax / h_des) + 1, 64), 20001))
    grid = np.linspace(0.0, tmax, n)
    h = grid[1] - grid[0] if n > 1 else tmax
    if kind == "gaussian":
        if abs(B_L) < 1e-9:
            gs = static_gkt_zf(grid, 1.0, width, 0.0)
        else:
            gs = longitudinal_field_kubo_toyabe(grid, 1.0, width, B_L, 0.0)
    elif width <= 0 or omega0 < _FIELD_DECOUPLING_RATIO * width:
        gs = static_lorentzian_kt_zf(grid, 1.0, width, 0.0)
    else:
        gs = _lorentzian_lf_uniform(width, omega0, h, n)
    gd = _strong_collision_solve(np.asarray(gs, dtype=float), nu_solve, h)

    if len(_DYN_KT_CACHE) >= _DYN_KT_CACHE_MAX:
        _DYN_KT_CACHE.pop(next(iter(_DYN_KT_CACHE)))  # see _bounded_cache_get
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

    In zero field (``omega0 < 0.05 a_L``) the solution is **exact and closed
    form**: the static function is a three-state linear system, so the
    strong-collision integral equation reduces to a 3x3 eigenproblem and
    ``G_d`` is a sum of three exponentials (:func:`_strong_collision_modes`),
    valid at every rate with no grid, cache or fast-fluctuation switch.  In a
    longitudinal field the static line shape has no finite realisation, and the
    Volterra equation is solved on a grid (:func:`_dynamic_kt_grid`).

    - ``nu -> 0``    : recovers the static Lorentzian KT (zero-field analytic
      eqn 5.47; longitudinal field computed numerically per Blundell et al. 2022).
    - ``nu >> a_L``  : the relaxation rate saturates at ``4 a_L / 3`` (a
      Lorentzian distribution has no second moment to narrow).
    - ``B_L -> inf`` : decoupling, G -> 1.
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    tt = np.atleast_1d(np.abs(t))
    a = float(a_L)
    gamma_mu = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    omega0 = abs(gamma_mu * (float(B_L) * GAUSS_TO_TESLA))
    zero_field = a <= 0 or omega0 < _FIELD_DECOUPLING_RATIO * a
    if nu <= 1e-9:
        if zero_field:
            gd = static_lorentzian_kt_zf(tt, 1.0, a, 0.0)
        else:
            gd = static_lorentzian_kt_lf(tt, 1.0, a, B_L, 0.0)
    elif zero_field:
        lam, weights = _strong_collision_modes(*_lorentzian_kt_zf_realisation(a), float(nu))
        gd = _exponential_sum(tt, lam, weights)
    else:
        tmax = float(max(tt.max(), 1e-6))
        grid, gd_grid = _dynamic_kt_grid("lorentzian", a, float(nu), float(B_L), tmax)
        gd = np.interp(tt, grid, gd_grid)
    out = A0 * np.asarray(gd, dtype=float) + baseline
    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS: dict[str, ModelDefinition] = {}


def _register(name: str, desc: str, fn: Callable, params: list[str], defaults: dict) -> None:
    insert_definition(
        MODELS,
        ModelDefinition(name, desc, fn, params, defaults, param_info_map(params)),
        registry_label="MODELS",
    )


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

r"""Thermal kernel and Fermi-surface averaging for superfluid density models.

Implemented kernel
------------------

.. math::

     \rho_s(T)=1+2\left\langle\int_{\Delta(T,k)}^{\infty}
     \frac{\partial f}{\partial E}\frac{E\,dE}{\sqrt{E^2-\Delta^2(T,k)}}
     \right\rangle_{FS}

with :math:`\Delta(T,k)=\Delta_0\,\delta(T/T_c)\,g(k)`.

Numerical strategy
------------------

- The thermal integral uses a nonsingular momentum variable in units of k_B T.
- Angular averages refine a periodic azimuthal trapezoid and a polar Gauss rule.
- Limiting values are imposed for stability: ``rho_s(0)=1`` and ``rho_s(T>=Tc)=0``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline
from scipy.special import expit, roots_legendre

from asymmetry.core.fitting.sc.bcs import delta_bcs
from asymmetry.core.fitting.sc.gaps import GapForm

ArrayLikeFloat = NDArray[np.float64]
ReducedGapFunction = Callable[[ArrayLikeFloat], ArrayLikeFloat]


# u = sqrt(E²-Delta²)/(k_B T) is dimensionless. Analyticity up to |Im u| = π
# gives trapezoid error < 2e-15 at h=1/2; truncating at u=36 adds < 2.4e-16.
_THERMAL_STEP = 0.5
_THERMAL_NODES = np.arange(0.0, 36.0 + _THERMAL_STEP, _THERMAL_STEP)
_THERMAL_NODES.setflags(write=False)


def thermal_integral(
    gap_over_temperature: NDArray[np.float64] | list[float] | float,
) -> ArrayLikeFloat:
    r"""Evaluate I(d) = -∫ f(sqrt(u²+d²)) [1-f(sqrt(u²+d²))] du.

    Here d = Delta/(k_B T), u = sqrt(E²-Delta²)/(k_B T), and f(x) = expit(-x).
    The substitution removes the density-of-states endpoint singularity from
    the thermal response (Prozorov & Giannetta, SST 19, R41, 2006, Eq. (16)).
    The fixed trapezoid is exponentially convergent; I(0) = -1/2 follows without
    a zero-gap branch. Scalar, empty, and N-dimensional shapes are preserved.
    """
    d = np.asarray(gap_over_temperature, dtype=float)
    occupation = expit(-np.hypot(d[..., None], _THERMAL_NODES))
    integrand = occupation * (1.0 - occupation)
    return np.asarray(
        -_THERMAL_STEP
        * (np.sum(integrand, axis=-1) - 0.5 * (integrand[..., 0] + integrand[..., -1])),
        dtype=float,
    )


@lru_cache(maxsize=1)
def _thermal_spline() -> CubicHermiteSpline:
    """Hermite table of I(d), with the derivative integrated analytically."""
    d = np.linspace(0, 80, 16001)
    energy = np.hypot(d[:, None], _THERMAL_NODES)
    occupation = expit(-energy)
    integrand = occupation * (1 - occupation) * (1 - 2 * occupation)
    integrand *= np.divide(d[:, None], energy, out=np.zeros_like(energy), where=energy > 0)
    derivative = _THERMAL_STEP * (
        np.sum(integrand, axis=-1) - 0.5 * (integrand[:, 0] + integrand[:, -1])
    )
    return CubicHermiteSpline(d, thermal_integral(d), derivative, extrapolate=False)


def _angular_thermal_integral(d: ArrayLikeFloat) -> ArrayLikeFloat:
    """Interpolate I with absolute error < 1e-11 on the certified gap range."""
    argument = np.abs(d)
    # For d > 80, |I(d)| <= d K_1(d) < 3e-33 from f(1-f) <= exp(-E).
    return np.where(argument > 80, 0.0, _thermal_spline()(np.minimum(argument, 80)))


_ANGULAR_TOLERANCE = 1e-7
_MAX_ANGULAR_NODES = 1 << 14
_MIN_REDUCED_TEMPERATURE = 0.005


@lru_cache(maxsize=32)
def _polar_grid(n: int, mirror: bool) -> tuple[ArrayLikeFloat, ArrayLikeFloat]:
    """Normalised cos(theta) quadrature, reduced to one hemisphere when symmetric."""
    u, weights = roots_legendre(n)
    if mirror:
        u = (u + 1) / 2
    weights = weights / 2
    theta = np.arccos(u)
    theta.setflags(write=False)
    weights.setflags(write=False)
    return theta, weights


@lru_cache(maxsize=64)
def _azimuthal_grid(n: int, period: float | None) -> ArrayLikeFloat:
    """Nested periodic nodes on one declared |g| period."""
    phi = np.zeros(1) if period is None else np.arange(n) * (period / n)
    phi.setflags(write=False)
    return phi


def _surface_average(scale, gap, gap_params, n_theta, n_phi):
    phi = _azimuthal_grid(n_phi, gap.period)
    if gap.surface == "sphere":
        theta, weights = _polar_grid(n_theta, gap.equator_mirror)
        values = gap.function(theta[:, None], phi[None, :], **gap_params)
    else:
        weights = np.ones(1)
        values = gap.function(phi, **gap_params)
    evaluate = thermal_integral if n_theta == n_phi == 1 else _angular_thermal_integral
    integrals = evaluate(scale * np.atleast_2d(values))
    value = 1 + 2 * float(np.sum(weights * np.mean(integrals, axis=-1)))
    if not np.isfinite(value):
        raise ValueError(f"{gap.name}: surface average must be finite")
    return value


def _converged_surface_average(scale, gap, gap_params, temperature):
    polar = gap.surface == "sphere"
    azimuthal = gap.period is not None
    if not (polar or azimuthal):
        return _surface_average(scale, gap, gap_params, 1, 1)
    if temperature < _MIN_REDUCED_TEMPERATURE:
        raise ValueError(
            f"{gap.name}: angular average requires T/Tc >= {_MIN_REDUCED_TEMPERATURE}; got {temperature:g}."
        )
    n_theta, n_phi = (8 if polar else 1), (16 if azimuthal else 1)
    cache = {}
    angular_scale = scale * (1 + sum(abs(value) for value in gap_params.values()))

    def average(nt, np_):
        if max(nt, np_) > _MAX_ANGULAR_NODES:
            raise ValueError(
                f"{gap.name}: angular average at T/Tc={temperature:g} did not converge within the {_MAX_ANGULAR_NODES}-node cap."
            )
        key = nt, np_
        if key not in cache:
            cache[key] = _surface_average(scale, gap, gap_params, nt, np_)
        return cache[key]

    while True:
        nt = 2 * n_theta if polar else 1
        np_ = 2 * n_phi if azimuthal else 1
        fine = average(nt, np_)
        polar_error = 2 * abs(fine - average(n_theta, np_)) if polar else 0.0
        azimuthal_error = 2 * abs(fine - average(nt, n_phi)) if azimuthal else 0.0
        # Nodal thermal features narrow as T/Delta in azimuth and sqrt(T/Delta)
        # in cos(theta) near a polar endpoint; coarse grids must resolve them.
        resolved_phi = not azimuthal or np_ >= angular_scale
        resolved_theta = not polar or nt >= 4 * np.sqrt(angular_scale)
        if resolved_phi and resolved_theta and polar_error + azimuthal_error <= _ANGULAR_TOLERANCE:
            return fine
        if not resolved_theta or polar_error > _ANGULAR_TOLERANCE / 2:
            n_theta = nt
        if not resolved_phi or azimuthal_error > _ANGULAR_TOLERANCE / 2:
            n_phi = np_


def superfluid_density(
    t_values: NDArray[np.float64] | list[float] | float,
    *,
    tc: float,
    gap_ratio: float,
    gap: GapForm,
    gap_params: dict[str, float] | None = None,
    reduced_gap_function: ReducedGapFunction | None = None,
) -> ArrayLikeFloat:
    r"""Normalised density with a controlled Fermi-surface average.

    ``gap`` declares the function, surface and angular symmetries. Both angular
    coordinates refine until their combined error estimate is <= 1e-7, with a
    16384-node cap per coordinate and explicit failure. Nonconstant angular gaps require
    T/Tc >= .005; zero temperature is analytic. Azimuth-independent cylindrical
    gaps need no angular quadrature and retain the full thermal-kernel domain.
    """
    if not np.isfinite(tc) or tc <= 0:
        raise ValueError("Tc must be finite and > 0")
    temperatures = np.asarray(t_values, dtype=float)
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures < 0):
        raise ValueError("Temperatures must be finite and >= 0")
    kwargs = gap_params or {}
    if not np.isfinite(gap_ratio) or any(not np.isfinite(value) for value in kwargs.values()):
        raise ValueError("Gap ratio and parameters must be finite")
    reduced = temperatures / float(tc)
    density = np.zeros_like(reduced)
    density[reduced == 0] = 1.0
    mask = (reduced > 0) & (reduced < 1)
    gap_scale = delta_bcs if reduced_gap_function is None else reduced_gap_function
    gaps = abs(float(gap_ratio)) * gap_scale(reduced[mask])
    for index, temperature, delta in zip(np.flatnonzero(mask), reduced[mask], gaps, strict=True):
        density.flat[index] = _converged_surface_average(
            delta / temperature, gap, kwargs, temperature
        )
    return density

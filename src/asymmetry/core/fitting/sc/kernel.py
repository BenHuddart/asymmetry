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
- Angular integrals use Gauss-Legendre quadrature.
- Limiting values are imposed for stability: ``rho_s(0)=1`` and ``rho_s(T>=Tc)=0``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.special import expit

from asymmetry.core.fitting.sc.bcs import delta_bcs

ArrayLikeFloat = NDArray[np.float64]
ReducedGapFunction = Callable[[ArrayLikeFloat], ArrayLikeFloat]


# u = sqrt(E²-Delta²)/(k_B T) is dimensionless. Analyticity up to |Im u| = π
# gives trapezoid error < 2e-15 at h=1/2; truncating at u=36 adds < 2.4e-16.
_THERMAL_STEP = 0.5
_THERMAL_NODES = np.arange(0.0, 36.0 + _THERMAL_STEP, _THERMAL_STEP)
_THERMAL_NODES.setflags(write=False)


@lru_cache(maxsize=32)
def _angular_grid_2d(n_phi: int) -> tuple[ArrayLikeFloat, ArrayLikeFloat]:
    """Return phi grid and normalized weights for <f> = (1/2pi) int f(phi) dphi."""
    n = max(int(n_phi), 8)
    nodes, weights = leggauss(n)
    phi = np.pi * (nodes + 1.0)
    avg_weights = 0.5 * weights
    return np.asarray(phi, dtype=float), np.asarray(avg_weights, dtype=float)


@lru_cache(maxsize=16)
def _angular_grid_3d(
    n_theta: int,
    n_phi: int,
) -> tuple[ArrayLikeFloat, ArrayLikeFloat, ArrayLikeFloat]:
    """Return theta, phi mesh and normalized sphere weights."""
    n_t = max(int(n_theta), 6)
    n_p = max(int(n_phi), 8)

    u_nodes, u_weights = leggauss(n_t)  # u = cos(theta) in [-1, 1]
    theta = np.arccos(np.clip(u_nodes, -1.0, 1.0))

    phi_nodes, phi_weights = leggauss(n_p)
    phi = np.pi * (phi_nodes + 1.0)

    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing="ij")
    weight_mesh = 0.25 * np.outer(u_weights, phi_weights)

    return (
        np.asarray(theta_mesh, dtype=float),
        np.asarray(phi_mesh, dtype=float),
        np.asarray(weight_mesh, dtype=float),
    )


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


def average_2d(
    func: Callable[[ArrayLikeFloat], ArrayLikeFloat],
    *,
    n_phi: int = 128,
) -> float:
    r"""Return normalized 2D angular average over :math:`\phi\in[0,2\pi)`."""
    phi, w = _angular_grid_2d(int(n_phi))
    values = np.asarray(func(phi), dtype=float)
    return float(np.sum(w * values))


def average_3d(
    func: Callable[[ArrayLikeFloat, ArrayLikeFloat], ArrayLikeFloat],
    *,
    n_theta: int = 32,
    n_phi: int = 64,
) -> float:
    r"""Return normalized 3D spherical average over :math:`(\theta,\phi)`."""
    theta_mesh, phi_mesh, w = _angular_grid_3d(int(n_theta), int(n_phi))
    values = np.asarray(func(theta_mesh, phi_mesh), dtype=float)
    return float(np.sum(w * values))


def _normalize_gap_form_factor(
    g_abs: ArrayLikeFloat,
    w: ArrayLikeFloat,
    *,
    normalize_g: bool,
    normalization: str,
) -> ArrayLikeFloat:
    if not normalize_g:
        return g_abs

    mode = normalization.lower().strip()
    if mode == "rms":
        scale = np.sqrt(float(np.sum(w * np.square(g_abs))))
    elif mode in {"mean_abs", "mean"}:
        scale = float(np.sum(w * g_abs))
    else:
        raise ValueError("normalization must be 'rms' or 'mean_abs'")

    scale_safe = max(abs(scale), 1e-12)
    return np.asarray(g_abs / scale_safe, dtype=float)


def superfluid_density_2d(
    t_values: NDArray[np.float64] | list[float] | float,
    *,
    tc: float,
    gap_ratio: float,
    gap_function: Callable[..., ArrayLikeFloat],
    gap_params: dict[str, float] | None = None,
    n_phi: int = 128,
    reduced_gap_function: ReducedGapFunction | None = None,
    normalize_g: bool = False,
    normalization: str = "rms",
) -> ArrayLikeFloat:
    r"""Compute :math:`\rho_s(T)` for a quasi-2D gap function ``g(phi)``.

    Parameters
    ----------
    t_values
        Temperature values in K.
    tc
        Critical temperature in K.
    gap_ratio
        Dimensionless gap ratio :math:`\Delta_0/(k_B T_c)`.
    gap_function
        Callable returning angular form factor ``g(phi, **gap_params)``.
    reduced_gap_function
        Optional helper returning the reduced temperature dependence
        :math:`\delta(T/T_c)`. Defaults to :func:`asymmetry.core.fitting.sc.bcs.delta_bcs`.
    normalize_g
        If True, normalize ``|g|`` before integration.
    normalization
        ``"rms"`` or ``"mean_abs"`` normalization mode.
    """
    tc_value = float(tc)
    if tc_value <= 0.0:
        raise ValueError("Tc must be > 0")

    temps = np.asarray(t_values, dtype=float)
    rho = np.zeros_like(temps, dtype=float)
    t_reduced = temps / tc_value

    rho[t_reduced <= 0.0] = 1.0
    rho[t_reduced >= 1.0] = 0.0

    mask = (t_reduced > 0.0) & (t_reduced < 1.0)
    if not np.any(mask):
        return rho

    phi, w_phi = _angular_grid_2d(int(n_phi))
    gap_kwargs = gap_params or {}
    gap_scale = delta_bcs if reduced_gap_function is None else reduced_gap_function

    g_raw = np.asarray(gap_function(phi, **gap_kwargs), dtype=float)
    g_abs = np.asarray(np.abs(g_raw), dtype=float)
    g_eff = _normalize_gap_form_factor(
        g_abs,
        w_phi,
        normalize_g=normalize_g,
        normalization=normalization,
    )

    delta_t = np.abs(float(gap_ratio)) * np.asarray(gap_scale(t_reduced[mask]), dtype=float)

    rho_vals: list[float] = []
    for tt, dt in zip(t_reduced[mask], delta_t, strict=True):
        delta_angles = dt * g_eff
        integral_vals = thermal_integral(delta_angles / float(tt))
        fs_avg = float(np.sum(w_phi * integral_vals))
        rho_vals.append(1.0 + 2.0 * fs_avg)

    rho_mask = np.asarray(rho_vals, dtype=float)
    rho[mask] = rho_mask
    return rho


def superfluid_density_3d(
    t_values: NDArray[np.float64] | list[float] | float,
    *,
    tc: float,
    gap_ratio: float,
    gap_function: Callable[..., ArrayLikeFloat],
    gap_params: dict[str, float] | None = None,
    n_theta: int = 32,
    n_phi: int = 64,
    reduced_gap_function: ReducedGapFunction | None = None,
    normalize_g: bool = False,
    normalization: str = "rms",
) -> ArrayLikeFloat:
    r"""Compute :math:`\rho_s(T)` for a 3D gap function ``g(theta, phi)``.

    Parameters
    ----------
    t_values
        Temperature values in K.
    tc
        Critical temperature in K.
    gap_ratio
        Dimensionless gap ratio :math:`\Delta_0/(k_B T_c)`.
    gap_function
        Callable returning angular form factor ``g(theta, phi, **gap_params)``.
    gap_params
        Optional keyword arguments forwarded to ``gap_function``.
    n_theta
        Number of Gauss-Legendre nodes in polar angle.
    n_phi
        Number of Gauss-Legendre nodes in azimuthal angle.
    reduced_gap_function
        Optional helper returning the reduced temperature dependence
        :math:`\delta(T/T_c)`. Defaults to :func:`asymmetry.core.fitting.sc.bcs.delta_bcs`.
    normalize_g
        If ``True``, normalize :math:`|g|` before integration.
    normalization
        Either ``"rms"`` or ``"mean_abs"``.

    Returns
    -------
    numpy.ndarray
        Normalized superfluid density, bounded by ``[0, 1]`` up to round-off, with
        ``rho_s(0)=1`` and ``rho_s(T>=Tc)=0`` imposed for numerical stability.
    """
    tc_value = float(tc)
    if tc_value <= 0.0:
        raise ValueError("Tc must be > 0")

    temps = np.asarray(t_values, dtype=float)
    rho = np.zeros_like(temps, dtype=float)
    t_reduced = temps / tc_value

    rho[t_reduced <= 0.0] = 1.0
    rho[t_reduced >= 1.0] = 0.0

    mask = (t_reduced > 0.0) & (t_reduced < 1.0)
    if not np.any(mask):
        return rho

    theta_mesh, phi_mesh, w_mesh = _angular_grid_3d(int(n_theta), int(n_phi))
    w_flat = np.asarray(w_mesh.ravel(), dtype=float)

    gap_kwargs = gap_params or {}
    gap_scale = delta_bcs if reduced_gap_function is None else reduced_gap_function
    g_raw = np.asarray(gap_function(theta_mesh, phi_mesh, **gap_kwargs), dtype=float)
    g_abs = np.asarray(np.abs(g_raw).ravel(), dtype=float)
    g_eff = _normalize_gap_form_factor(
        g_abs,
        w_flat,
        normalize_g=normalize_g,
        normalization=normalization,
    )

    delta_t = np.abs(float(gap_ratio)) * np.asarray(gap_scale(t_reduced[mask]), dtype=float)

    rho_vals: list[float] = []
    for tt, dt in zip(t_reduced[mask], delta_t, strict=True):
        delta_angles = dt * g_eff
        integral_vals = thermal_integral(delta_angles / float(tt))
        fs_avg = float(np.sum(w_flat * integral_vals))
        rho_vals.append(1.0 + 2.0 * fs_avg)

    rho_mask = np.asarray(rho_vals, dtype=float)
    rho[mask] = rho_mask
    return rho


def superfluid_density(
    t_values: NDArray[np.float64] | list[float] | float,
    *,
    tc: float,
    gap_ratio: float,
    gap_function: Callable[..., ArrayLikeFloat],
    gap_params: dict[str, float] | None = None,
    n_phi: int = 128,
    reduced_gap_function: ReducedGapFunction | None = None,
    normalize_g: bool = False,
    normalization: str = "rms",
) -> ArrayLikeFloat:
    """Alias for :func:`superfluid_density_2d` with identical semantics.

    This convenience wrapper keeps older call sites concise when only quasi-2D
    angular averaging is required.
    """
    return superfluid_density_2d(
        t_values,
        tc=tc,
        gap_ratio=gap_ratio,
        gap_function=gap_function,
        gap_params=gap_params,
        n_phi=n_phi,
        reduced_gap_function=reduced_gap_function,
        normalize_g=normalize_g,
        normalization=normalization,
    )

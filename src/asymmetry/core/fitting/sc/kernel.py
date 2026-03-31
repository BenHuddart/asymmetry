r"""Thermal kernel and Fermi-surface averaging for superfluid density models.

Implemented kernel
------------------

.. math::

     \rho_s(T)=1+2\left\langle\int_{\Delta(T,k)}^{\infty}
     \frac{\partial f}{\partial E}\frac{E\,dE}{\sqrt{E^2-\Delta^2(T,k)}}
     \right\rangle_{FS}

with :math:`\Delta(T,k)=\Delta_0\,\delta_{BCS}(T/T_c)\,g(k)`.

Numerical strategy
------------------

- Energy and angular integrals use Gauss-Legendre quadrature.
- The lower-bound singularity in the energy integral is handled by variable
    substitution and finite-precision safeguards.
- Limiting values are imposed for stability: ``rho_s(0)=1`` and ``rho_s(T>=Tc)=0``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from asymmetry.core.fitting.sc.bcs import delta_bcs

ArrayLikeFloat = NDArray[np.float64]


@lru_cache(maxsize=16)
def _energy_nodes(n_energy: int) -> tuple[ArrayLikeFloat, ArrayLikeFloat]:
    """Return transformed quadrature nodes for x in [0, 1)."""
    n = max(int(n_energy), 8)
    nodes, weights = leggauss(n)
    # Map from [-1, 1] to [0, 1]. Endpoints are excluded by Gauss-Legendre.
    x = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    return np.asarray(x, dtype=float), np.asarray(w, dtype=float)


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


def _fermi_derivative(e_reduced: ArrayLikeFloat, t_reduced: float) -> ArrayLikeFloat:
    """Return d f / dE in reduced units with E normalized by k_B Tc."""
    t = max(float(t_reduced), 1e-12)
    arg = np.clip(e_reduced / (2.0 * t), -80.0, 80.0)
    sech2 = 1.0 / np.cosh(arg) ** 2
    return np.asarray(-0.25 * sech2 / t, dtype=float)


def energy_integral(
    delta_reduced: NDArray[np.float64] | list[float] | float,
    t_reduced: float,
    *,
    n_energy: int = 96,
) -> ArrayLikeFloat:
    r"""Evaluate the reduced thermal integral for one temperature.

    .. math::

       I(\Delta,t)=\int_{\Delta}^{\infty}
       \frac{\partial f}{\partial E}
       \frac{E\,dE}{\sqrt{E^2-\Delta^2}}

    where energies are expressed in units of :math:`k_B T_c`.

    The integral is transformed to x in [0, 1) via E = delta + x/(1-x).
    """
    deltas = np.maximum(np.asarray(delta_reduced, dtype=float), 0.0)
    t = float(t_reduced)

    out = np.zeros_like(deltas, dtype=float)
    if t <= 0.0:
        return out

    zero_mask = deltas <= 1e-12
    out[zero_mask] = -0.5

    nonzero = ~zero_mask
    if not np.any(nonzero):
        return out

    x, w = _energy_nodes(int(n_energy))
    tail = x / np.maximum(1.0 - x, 1e-15)
    jac = 1.0 / np.maximum((1.0 - x) ** 2, 1e-30)

    delta_nz = deltas[nonzero]
    e = delta_nz[:, None] + tail[None, :]
    denom = np.sqrt(np.maximum(e * e - delta_nz[:, None] ** 2, 1e-300))
    dfd_e = _fermi_derivative(e, t)

    integrand = dfd_e * e / denom * jac[None, :]
    out[nonzero] = np.sum(integrand * w[None, :], axis=1)
    return np.asarray(out, dtype=float)


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
    n_energy: int = 96,
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

    g_raw = np.asarray(gap_function(phi, **gap_kwargs), dtype=float)
    g_abs = np.asarray(np.abs(g_raw), dtype=float)
    g_eff = _normalize_gap_form_factor(
        g_abs,
        w_phi,
        normalize_g=normalize_g,
        normalization=normalization,
    )

    delta_t = np.abs(float(gap_ratio)) * delta_bcs(t_reduced[mask])

    rho_vals: list[float] = []
    for tt, dt in zip(t_reduced[mask], delta_t, strict=True):
        delta_angles = dt * g_eff
        integral_vals = energy_integral(delta_angles, float(tt), n_energy=int(n_energy))
        fs_avg = float(np.sum(w_phi * integral_vals))
        rho_vals.append(1.0 + 2.0 * fs_avg)

    rho_mask = np.asarray(rho_vals, dtype=float)
    rho[mask] = np.clip(rho_mask, 0.0, 1.0)
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
    n_energy: int = 96,
    normalize_g: bool = False,
    normalization: str = "rms",
) -> ArrayLikeFloat:
    r"""Compute :math:`\rho_s(T)` for a 3D gap function ``g(theta, phi)``."""
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
    g_raw = np.asarray(gap_function(theta_mesh, phi_mesh, **gap_kwargs), dtype=float)
    g_abs = np.asarray(np.abs(g_raw).ravel(), dtype=float)
    g_eff = _normalize_gap_form_factor(
        g_abs,
        w_flat,
        normalize_g=normalize_g,
        normalization=normalization,
    )

    delta_t = np.abs(float(gap_ratio)) * delta_bcs(t_reduced[mask])

    rho_vals: list[float] = []
    for tt, dt in zip(t_reduced[mask], delta_t, strict=True):
        delta_angles = dt * g_eff
        integral_vals = energy_integral(delta_angles, float(tt), n_energy=int(n_energy))
        fs_avg = float(np.sum(w_flat * integral_vals))
        rho_vals.append(1.0 + 2.0 * fs_avg)

    rho_mask = np.asarray(rho_vals, dtype=float)
    rho[mask] = np.clip(rho_mask, 0.0, 1.0)
    return rho


def superfluid_density(
    t_values: NDArray[np.float64] | list[float] | float,
    *,
    tc: float,
    gap_ratio: float,
    gap_function: Callable[..., ArrayLikeFloat],
    gap_params: dict[str, float] | None = None,
    n_phi: int = 128,
    n_energy: int = 96,
    normalize_g: bool = False,
    normalization: str = "rms",
) -> ArrayLikeFloat:
    """Alias for :func:`superfluid_density_2d` with identical semantics."""
    return superfluid_density_2d(
        t_values,
        tc=tc,
        gap_ratio=gap_ratio,
        gap_function=gap_function,
        gap_params=gap_params,
        n_phi=n_phi,
        n_energy=n_energy,
        normalize_g=normalize_g,
        normalization=normalization,
    )

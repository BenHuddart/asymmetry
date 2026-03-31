"""Angular gap form factors g(k) for superconducting models.

The quasiparticle excitation spectrum depends on the magnitude of Delta(k, T),
so observables that integrate over quasiparticle energies use the magnitude of
these form factors. Sign-preserving variants are still exposed for users who
want explicit control of conventions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayLikeFloat = NDArray[np.float64]


def isotropic_s(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    """Isotropic s-wave: g(phi) = 1."""
    return np.ones_like(np.asarray(phi, dtype=float), dtype=float)


def anisotropic_s_cos4(
    phi: NDArray[np.float64] | list[float] | float,
    a_anis: float,
) -> ArrayLikeFloat:
    """Anisotropic s-wave: g(phi) = 1 + a*cos(4phi).

    For abs(a) < 1 this form is nodeless. For abs(a) >= 1 accidental nodes can
    occur.
    """
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(1.0 + float(a_anis) * np.cos(4.0 * phi_arr), dtype=float)


def anisotropic_s_cos2(
    phi: NDArray[np.float64] | list[float] | float,
    a_anis: float,
) -> ArrayLikeFloat:
    """Anisotropic s-wave: g(phi) = 1 + a*cos(2phi)."""
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(1.0 + float(a_anis) * np.cos(2.0 * phi_arr), dtype=float)


def d_wave(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    """d-wave: g(phi) = cos(2phi), with line nodes in 2D."""
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.cos(2.0 * phi_arr), dtype=float)


def extended_s(
    phi: NDArray[np.float64] | list[float] | float,
    *,
    signed: bool = False,
) -> ArrayLikeFloat:
    """Extended s-wave based on cos(2phi).

    Parameters
    ----------
    signed
        If True, return cos(2phi). If False, return absolute(cos(2phi)).
    """
    base = d_wave(phi)
    if signed:
        return base
    return np.asarray(np.abs(base), dtype=float)


def anisotropic_d_with_harmonics(
    phi: NDArray[np.float64] | list[float] | float,
    b_harm: float,
) -> ArrayLikeFloat:
    """Anisotropic d-wave with higher harmonic: cos(2phi) + b*cos(6phi)."""
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.cos(2.0 * phi_arr) + float(b_harm) * np.cos(6.0 * phi_arr), dtype=float)


def nonmonotonic_d_wave(
    phi: NDArray[np.float64] | list[float] | float,
    beta_nm: float,
) -> ArrayLikeFloat:
    """Nonmonotonic d-wave: beta*cos(2phi) + (1-beta)*cos(6phi)."""
    phi_arr = np.asarray(phi, dtype=float)
    beta = float(beta_nm)
    return np.asarray(
        beta * np.cos(2.0 * phi_arr) + (1.0 - beta) * np.cos(6.0 * phi_arr),
        dtype=float,
    )


def p_wave_axial_2d(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    """2D p-wave axial form: g(phi) = cos(phi)."""
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.cos(phi_arr), dtype=float)


def p_wave_chiral_magnitude(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    """Chiral p-wave magnitude model sqrt(cos(phi)^2 + sin(phi)^2) = 1."""
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.sqrt(np.cos(phi_arr) ** 2 + np.sin(phi_arr) ** 2), dtype=float)


def point_node_3d(theta: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    """Example 3D point-node form: g(theta) = cos(theta)."""
    theta_arr = np.asarray(theta, dtype=float)
    return np.asarray(np.cos(theta_arr), dtype=float)


def line_node_3d(theta: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    """Example 3D line-node form: g(theta) = sin(theta)."""
    theta_arr = np.asarray(theta, dtype=float)
    return np.asarray(np.sin(theta_arr), dtype=float)

r"""Angular gap form factors g(k) for superconducting models.

The superfluid-density kernel uses

.. math::

    \Delta(T, \mathbf{k}) = \Delta_0\,\delta_{BCS}(T/T_c)\,g(\mathbf{k}),

so each helper in this module returns a symmetry form factor ``g``. For
quasiparticle-energy observables, :math:`|g|` typically enters the kernel,
while signed variants are kept available for explicit convention control.

References
----------
[1] R. Prozorov and R. W. Giannetta, Supercond. Sci. Technol. 19, R41 (2006).
[2] A. Carrington and F. Manzano, Physica C 385, 205 (2003).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayLikeFloat = NDArray[np.float64]


def isotropic_s(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return isotropic s-wave form factor.

    .. math::

       g(\phi) = 1.
    """
    return np.ones_like(np.asarray(phi, dtype=float), dtype=float)


def anisotropic_s_cos4(
    phi: NDArray[np.float64] | list[float] | float,
    a_anis: float,
) -> ArrayLikeFloat:
    r"""Return fourfold-anisotropic s-wave form factor.

    .. math::

       g(\phi) = 1 + a\cos(4\phi).

    For abs(a) < 1 this form is nodeless. For abs(a) >= 1 accidental nodes can
    occur.
    """
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(1.0 + float(a_anis) * np.cos(4.0 * phi_arr), dtype=float)


def anisotropic_s_cos2(
    phi: NDArray[np.float64] | list[float] | float,
    a_anis: float,
) -> ArrayLikeFloat:
    r"""Return twofold-anisotropic s-wave form factor.

    .. math::

       g(\phi) = 1 + a\cos(2\phi).
    """
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(1.0 + float(a_anis) * np.cos(2.0 * phi_arr), dtype=float)


def d_wave(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return 2D d-wave form factor with line nodes.

    .. math::

       g(\phi) = \cos(2\phi).
    """
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.cos(2.0 * phi_arr), dtype=float)


def extended_s(
    phi: NDArray[np.float64] | list[float] | float,
    *,
    signed: bool = False,
) -> ArrayLikeFloat:
    r"""Return extended-s form based on ``cos(2phi)``.

    .. math::

       g(\phi) = \cos(2\phi)\ \text{or}\ |\cos(2\phi)|.

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
    r"""Return anisotropic d-wave with higher harmonic.

    .. math::

       g(\phi) = \cos(2\phi) + b\cos(6\phi).
    """
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.cos(2.0 * phi_arr) + float(b_harm) * np.cos(6.0 * phi_arr), dtype=float)


def nonmonotonic_d_wave(
    phi: NDArray[np.float64] | list[float] | float,
    beta_nm: float,
) -> ArrayLikeFloat:
    r"""Return nonmonotonic d-wave form factor.

    .. math::

       g(\phi) = \beta\cos(2\phi) + (1-\beta)\cos(6\phi).
    """
    phi_arr = np.asarray(phi, dtype=float)
    beta = float(beta_nm)
    return np.asarray(
        beta * np.cos(2.0 * phi_arr) + (1.0 - beta) * np.cos(6.0 * phi_arr),
        dtype=float,
    )


def s_plus_g(
    theta: NDArray[np.float64] | list[float] | float,
    phi: NDArray[np.float64] | list[float] | float,
) -> ArrayLikeFloat:
    r"""Return s+g form factor used in anisotropic singlet models.

    .. math::

       g(\theta,\phi) = \frac{1 - \sin^4\theta\cos(4\phi)}{2}.

    This representation follows the weak-coupling form tabulated in Ref. [1]
    for s+g-wave phenomenology.
    """
    theta_arr = np.asarray(theta, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray((1.0 - np.sin(theta_arr) ** 4 * np.cos(4.0 * phi_arr)) / 2.0, dtype=float)


def p_wave_axial_2d(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return 2D axial p-wave form factor.

    .. math::

       g(\phi) = \cos(\phi).
    """
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.cos(phi_arr), dtype=float)


def p_wave_chiral_magnitude(phi: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return magnitude-only chiral p-wave form factor.

    .. math::

       g(\phi) = \sqrt{\cos^2\phi + \sin^2\phi} = 1.
    """
    phi_arr = np.asarray(phi, dtype=float)
    return np.asarray(np.sqrt(np.cos(phi_arr) ** 2 + np.sin(phi_arr) ** 2), dtype=float)


def point_node_3d(theta: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return example 3D point-node form factor.

    .. math::

       g(\theta) = \cos(\theta).
    """
    theta_arr = np.asarray(theta, dtype=float)
    return np.asarray(np.cos(theta_arr), dtype=float)


def line_node_3d(theta: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return example 3D line-node form factor.

    .. math::

       g(\theta) = \sin(\theta).
    """
    theta_arr = np.asarray(theta, dtype=float)
    return np.asarray(np.sin(theta_arr), dtype=float)

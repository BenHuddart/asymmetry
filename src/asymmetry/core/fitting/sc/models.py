r"""Superconducting sigma(T) models for TF-muSR penetration-depth analysis.

This module provides two layers:

- ``rho_*`` functions for normalized superfluid density :math:`\rho_s(T)`.
- ``sc_*`` functions that map :math:`\rho_s(T)` to measured Gaussian rate
  :math:`\sigma(T)`.

Conventions
-----------

Additive convention:

.. math::

    \sigma(T) = \sigma_0\,\rho_s(T) + \sigma_{bg}

Quadrature convention:

.. math::

    \sigma^2(T) = \sigma_{sc}^2\,\rho_s^2(T) + \sigma_{nm}^2

Gap magnitude can be supplied either as ``gap_ratio = Delta0/(k_B Tc)`` or as
``gap_mev``. If both are supplied, ``gap_mev`` takes precedence.

References
----------
[1] R. Prozorov and R. W. Giannetta, Supercond. Sci. Technol. 19, R41 (2006).
[2] A. Carrington and F. Manzano, Physica C 385, 205 (2003).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.sc import gaps
from asymmetry.core.fitting.sc.bcs import delta_generalized, resolve_gap_ratio
from asymmetry.core.fitting.sc.kernel import superfluid_density, superfluid_density_3d

ArrayLikeFloat = NDArray[np.float64]

_D_WAVE_SHAPE_FACTOR = 4.0 / 3.0
_S_PLUS_G_SHAPE_FACTOR = 2.0
_EXTENDED_S_SHAPE_FACTOR = _D_WAVE_SHAPE_FACTOR


def _make_generalized_reduced_gap(
    *,
    gap_ratio: float,
    shape_factor: float,
) -> Callable[[ArrayLikeFloat], ArrayLikeFloat]:
    def reduced_gap(t_reduced: ArrayLikeFloat) -> ArrayLikeFloat:
        return delta_generalized(
            t_reduced,
            gap_ratio=gap_ratio,
            shape_factor=shape_factor,
        )

    return reduced_gap


def _optional_generalized_reduced_gap(
    *,
    gap_ratio: float,
    shape_factor_a: float,
) -> Callable[[ArrayLikeFloat], ArrayLikeFloat] | None:
    shape_factor = float(shape_factor_a)
    if not np.isfinite(shape_factor) or shape_factor <= 0.0:
        return None
    return _make_generalized_reduced_gap(
        gap_ratio=gap_ratio,
        shape_factor=shape_factor,
    )


def _as_temperature_array(T: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    """Return temperature input as a float NumPy array."""
    return np.asarray(T, dtype=float)


def _sigma_additive(rho: ArrayLikeFloat, sigma_0: float, sigma_bg: float = 0.0) -> ArrayLikeFloat:
    """Apply additive sigma convention to a superfluid-density curve."""
    return np.asarray(float(sigma_0) * rho + float(sigma_bg), dtype=float)


def _sigma_quadrature(rho: ArrayLikeFloat, sigma_sc: float, sigma_nm: float) -> ArrayLikeFloat:
    """Apply quadrature sigma convention to a superfluid-density curve."""
    return np.asarray(
        np.sqrt((float(sigma_sc) * rho) ** 2 + float(sigma_nm) ** 2),
        dtype=float,
    )


def rho_s_wave(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 1.764,
    gap_mev: float | None = None,
    n_phi: int = 64,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for isotropic s-wave gap, :math:`g(\phi)=1`.

    Parameters
    ----------
    T
        Temperature in K.
    Tc
        Critical temperature in K.
    gap_ratio
        Dimensionless ratio :math:`\Delta_0/(k_B T_c)`.
    gap_mev
        Optional :math:`\Delta_0` in meV. Overrides ``gap_ratio`` when given.
    n_phi
        Number of angular quadrature points for Fermi-surface averaging.
    n_energy
        Number of Gauss-Legendre nodes for the energy integral.

    Returns
    -------
    numpy.ndarray
        Normalized superfluid density :math:`\rho_s(T)`.
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=gaps.isotropic_s,
        n_phi=n_phi,
        n_energy=n_energy,
    )


def rho_d_wave(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 2.14,
    gap_mev: float | None = None,
    n_phi: int = 64,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for d-wave :math:`g(\phi)=\cos(2\phi)`.

    This model has line nodes and therefore stronger low-temperature variation
    than isotropic s-wave, typically close to linear-in-T in clean limits [1].
    The reduced gap amplitude uses the generalized weak-coupling form with the
    d-wave shape factor :math:`a=4/3`.
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=gaps.d_wave,
        n_phi=n_phi,
        n_energy=n_energy,
        reduced_gap_function=_make_generalized_reduced_gap(
            gap_ratio=ratio,
            shape_factor=_D_WAVE_SHAPE_FACTOR,
        ),
    )


def rho_anisotropic_s_cos4(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 1.764,
    a_anis: float = 0.2,
    shape_factor_a: float = 0.0,
    gap_mev: float | None = None,
    n_phi: int = 64,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for anisotropic s-wave ``1 + a*cos(4phi)``.

    ``a_anis`` controls anisotropy. For ``abs(a_anis) < 1`` the gap remains
    nodeless; accidental nodes may appear when ``abs(a_anis) >= 1``.
    If ``shape_factor_a > 0``, use the generalized weak-coupling reduced-gap
    law with that value; otherwise fall back to the Carrington-Manzano
    interpolation.
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=gaps.anisotropic_s_cos4,
        gap_params={"a_anis": float(a_anis)},
        n_phi=n_phi,
        n_energy=n_energy,
        reduced_gap_function=_optional_generalized_reduced_gap(
            gap_ratio=ratio,
            shape_factor_a=shape_factor_a,
        ),
    )


def rho_nonmonotonic_d(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 2.14,
    beta_nm: float = 0.8,
    gap_mev: float | None = None,
    n_phi: int = 64,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for nonmonotonic d-wave.

    .. math::

        g(\phi)=\beta\cos(2\phi)+(1-\beta)\cos(6\phi).

    This form is commonly used as a phenomenological extension when a monotonic
    :math:`\cos(2\phi)` d-wave is insufficient [1]. The temperature-dependent
    gap amplitude uses the same d-wave weak-coupling shape factor
    :math:`a=4/3` as :func:`rho_d_wave`.
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=gaps.nonmonotonic_d_wave,
        gap_params={"beta_nm": float(beta_nm)},
        n_phi=n_phi,
        n_energy=n_energy,
        reduced_gap_function=_make_generalized_reduced_gap(
            gap_ratio=ratio,
            shape_factor=_D_WAVE_SHAPE_FACTOR,
        ),
    )


def rho_p_wave_axial(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 2.0,
    shape_factor_a: float = 0.0,
    gap_mev: float | None = None,
    n_phi: int = 64,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for 2D axial p-wave, :math:`g(\phi)=\cos\phi`.

    If ``shape_factor_a > 0``, use the generalized weak-coupling reduced-gap
    law with that value; otherwise fall back to the Carrington-Manzano
    interpolation.
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=gaps.p_wave_axial_2d,
        n_phi=n_phi,
        n_energy=n_energy,
        reduced_gap_function=_optional_generalized_reduced_gap(
            gap_ratio=ratio,
            shape_factor_a=shape_factor_a,
        ),
    )


def rho_extended_s(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 2.14,
    signed: bool = False,
    gap_mev: float | None = None,
    n_phi: int = 64,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for extended s-wave based on ``cos(2phi)``.

    Set ``signed=True`` to preserve sign of :math:`\cos(2\phi)`. The default
    uses magnitude because the quasiparticle excitation energy depends on
    :math:`|\Delta|`. The reduced gap amplitude uses the generalized
    weak-coupling form with :math:`a=4/3`, consistent with the
    :math:`\cos(2\phi)` basis tabulated in Ref. [1].
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=gaps.extended_s,
        gap_params={"signed": bool(signed)},
        n_phi=n_phi,
        n_energy=n_energy,
        reduced_gap_function=_make_generalized_reduced_gap(
            gap_ratio=ratio,
            shape_factor=_EXTENDED_S_SHAPE_FACTOR,
        ),
    )


def rho_p_wave_polar_3d(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 2.0,
    shape_factor_a: float = 0.0,
    gap_mev: float | None = None,
    n_theta: int = 24,
    n_phi: int = 48,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for 3D polar p-wave line-node example.

    Uses :math:`g(\theta)=\sin\theta` with spherical angular averaging.
    If ``shape_factor_a > 0``, use the generalized weak-coupling reduced-gap
    law with that value; otherwise fall back to the Carrington-Manzano
    interpolation.
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density_3d(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=lambda theta, _phi: gaps.line_node_3d(theta),
        n_theta=n_theta,
        n_phi=n_phi,
        n_energy=n_energy,
        reduced_gap_function=_optional_generalized_reduced_gap(
            gap_ratio=ratio,
            shape_factor_a=shape_factor_a,
        ),
    )


def rho_s_plus_g(
    T: NDArray[np.float64] | list[float] | float,
    *,
    Tc: float,
    gap_ratio: float = 2.77,
    gap_mev: float | None = None,
    n_theta: int = 24,
    n_phi: int = 48,
    n_energy: int = 48,
) -> ArrayLikeFloat:
    r"""Return :math:`\rho_s(T)` for s+g-wave anisotropic singlet gap.

    .. math::

       g(\theta,\phi)=\frac{1-\sin^4\theta\cos(4\phi)}{2}.

    The default ``gap_ratio=2.77`` follows the weak-coupling tabulation used
    in Ref. [1]. The reduced gap amplitude uses the generalized weak-coupling
    form with the s+g shape factor :math:`a=2`.

    Parameters
    ----------
    T
        Temperature in K.
    Tc
        Critical temperature in K.
    gap_ratio
        Dimensionless ratio :math:`\Delta_0/(k_B T_c)`.
    gap_mev
        Optional :math:`\Delta_0` in meV. Overrides ``gap_ratio`` when given.
    n_theta
        Number of polar-angle quadrature points.
    n_phi
        Number of azimuthal-angle quadrature points.
    n_energy
        Number of Gauss-Legendre nodes for the energy integral.
    """
    ratio = resolve_gap_ratio(tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return superfluid_density_3d(
        T,
        tc=float(Tc),
        gap_ratio=ratio,
        gap_function=gaps.s_plus_g,
        n_theta=n_theta,
        n_phi=n_phi,
        n_energy=n_energy,
        reduced_gap_function=_make_generalized_reduced_gap(
            gap_ratio=ratio,
            shape_factor=_S_PLUS_G_SHAPE_FACTOR,
        ),
    )


def sc_s_wave(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio: float = 1.764,
    sigma_bg: float = 0.0,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Additive isotropic s-wave model for measured :math:`\sigma(T)`.

    .. math::

       \sigma(T) = \sigma_0\,\rho_s(T) + \sigma_{bg}
    """
    rho = rho_s_wave(T, Tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_d_wave(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio: float = 2.14,
    sigma_bg: float = 0.0,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Additive d-wave model for measured :math:`\sigma(T)`."""
    rho = rho_d_wave(T, Tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_anisotropic_s_cos4(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio: float = 1.764,
    a_anis: float = 0.2,
    shape_factor_a: float = 0.0,
    sigma_bg: float = 0.0,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Additive anisotropic-s model for measured :math:`\sigma(T)`."""
    rho = rho_anisotropic_s_cos4(
        T,
        Tc=Tc,
        gap_ratio=gap_ratio,
        a_anis=a_anis,
        shape_factor_a=shape_factor_a,
        gap_mev=gap_mev,
    )
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_nonmonotonic_d(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio: float = 2.14,
    beta_nm: float = 0.8,
    sigma_bg: float = 0.0,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Additive nonmonotonic-d model for measured :math:`\sigma(T)`."""
    rho = rho_nonmonotonic_d(
        T,
        Tc=Tc,
        gap_ratio=gap_ratio,
        beta_nm=beta_nm,
        gap_mev=gap_mev,
    )
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_p_wave_axial(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio: float = 2.0,
    shape_factor_a: float = 0.0,
    sigma_bg: float = 0.0,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Additive 2D axial p-wave model for measured :math:`\sigma(T)`."""
    rho = rho_p_wave_axial(
        T, Tc=Tc, gap_ratio=gap_ratio, shape_factor_a=shape_factor_a, gap_mev=gap_mev
    )
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_extended_s(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio: float = 2.14,
    signed_gap: float = 0.0,
    sigma_bg: float = 0.0,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Additive extended-s model for measured :math:`\sigma(T)`."""
    rho = rho_extended_s(
        T,
        Tc=Tc,
        gap_ratio=gap_ratio,
        signed=bool(signed_gap),
        gap_mev=gap_mev,
    )
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_s_plus_g(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio: float = 2.77,
    sigma_bg: float = 0.0,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Additive s+g model for measured :math:`\sigma(T)`.

    .. math::

       \sigma(T) = \sigma_0\,\rho_{s+g}(T) + \sigma_{bg}.
    """
    rho = rho_s_plus_g(T, Tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_alpha_model(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    alpha_sc: float = 1.0,
    sigma_bg: float = 0.0,
) -> ArrayLikeFloat:
    """Single-gap alpha model using isotropic BCS kernel.

    alpha_sc rescales the weak-coupling s-wave ratio 1.764.
    """
    effective_ratio = max(float(alpha_sc), 1e-6) * 1.764
    rho = rho_s_wave(T, Tc=Tc, gap_ratio=effective_ratio)
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_two_gap_ss(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio_1: float = 1.2,
    gap_ratio_2: float = 2.2,
    weight: float = 0.5,
    sigma_bg: float = 0.0,
) -> ArrayLikeFloat:
    r"""Two-gap isotropic model, weighted superfluid density sum.

    .. math::

         \rho_s(T) = w\rho_1(T) + (1-w)\rho_2(T),\quad 0\le w\le 1.

     This is the standard MgB2-style alpha-model decomposition for multiband
     superconductors [2].
    """
    w = np.clip(float(weight), 0.0, 1.0)
    rho_1 = rho_s_wave(T, Tc=Tc, gap_ratio=gap_ratio_1)
    rho_2 = rho_s_wave(T, Tc=Tc, gap_ratio=gap_ratio_2)
    rho = w * rho_1 + (1.0 - w) * rho_2
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_two_gap_sd(
    T: NDArray[np.float64] | list[float] | float,
    sigma_0: float,
    Tc: float,
    gap_ratio_s: float = 1.764,
    gap_ratio_d: float = 2.14,
    weight: float = 0.5,
    sigma_bg: float = 0.0,
) -> ArrayLikeFloat:
    """Two-gap mixed-symmetry model (s + d weighted sum)."""
    w = np.clip(float(weight), 0.0, 1.0)
    rho_s = rho_s_wave(T, Tc=Tc, gap_ratio=gap_ratio_s)
    rho_d = rho_d_wave(T, Tc=Tc, gap_ratio=gap_ratio_d)
    rho = w * rho_s + (1.0 - w) * rho_d
    return _sigma_additive(rho, sigma_0=sigma_0, sigma_bg=sigma_bg)


def sc_s_wave_q(
    T: NDArray[np.float64] | list[float] | float,
    sigma_sc: float,
    sigma_nm: float,
    Tc: float,
    gap_ratio: float = 1.764,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Quadrature s-wave model for measured :math:`\sigma(T)`.

    Use this convention when independent Gaussian broadening channels combine
    at the second-moment level, motivating
    :math:`\sigma^2=(\sigma_{sc}\rho_s)^2+\sigma_{nm}^2`.
    """
    rho = rho_s_wave(T, Tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return _sigma_quadrature(rho, sigma_sc=sigma_sc, sigma_nm=sigma_nm)


def sc_d_wave_q(
    T: NDArray[np.float64] | list[float] | float,
    sigma_sc: float,
    sigma_nm: float,
    Tc: float,
    gap_ratio: float = 2.14,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Quadrature d-wave model for measured :math:`\sigma(T)`."""
    rho = rho_d_wave(T, Tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return _sigma_quadrature(rho, sigma_sc=sigma_sc, sigma_nm=sigma_nm)


def sc_s_plus_g_q(
    T: NDArray[np.float64] | list[float] | float,
    sigma_sc: float,
    sigma_nm: float,
    Tc: float,
    gap_ratio: float = 2.77,
    gap_mev: float | None = None,
) -> ArrayLikeFloat:
    r"""Quadrature s+g model for measured :math:`\sigma(T)`.

    This is useful when superconducting and non-superconducting linewidth
    channels are treated as independent Gaussian contributions that add in
    quadrature.
    """
    rho = rho_s_plus_g(T, Tc=Tc, gap_ratio=gap_ratio, gap_mev=gap_mev)
    return _sigma_quadrature(rho, sigma_sc=sigma_sc, sigma_nm=sigma_nm)


def rho_to_lambda_inv_sq(
    rho: NDArray[np.float64] | list[float] | float,
    *,
    lambda_0_nm: float,
) -> ArrayLikeFloat:
    r"""Convert normalized :math:`\rho_s(T)` to :math:`\lambda^{-2}(T)` in nm^-2."""
    lam0 = max(float(lambda_0_nm), 1e-12)
    return np.asarray(np.asarray(rho, dtype=float) / (lam0 * lam0), dtype=float)


def rho_to_lambda(
    rho: NDArray[np.float64] | list[float] | float,
    *,
    lambda_0_nm: float,
) -> ArrayLikeFloat:
    r"""Convert normalized :math:`\rho_s(T)` to :math:`\lambda(T)` in nm."""
    lam0 = max(float(lambda_0_nm), 1e-12)
    rho_arr = np.asarray(rho, dtype=float)
    out = np.full_like(rho_arr, np.inf, dtype=float)
    positive = rho_arr > 0.0
    out[positive] = lam0 / np.sqrt(rho_arr[positive])
    return out


def lambda_inv_sq_from_model(
    T: NDArray[np.float64] | list[float] | float,
    *,
    rho_function: Callable[..., ArrayLikeFloat],
    lambda_0_nm: float,
    **kwargs: float,
) -> ArrayLikeFloat:
    r"""Evaluate :math:`\lambda^{-2}(T)` from any ``rho_function`` callable."""
    _ = _as_temperature_array(T)
    rho = np.asarray(rho_function(T, **kwargs), dtype=float)
    return rho_to_lambda_inv_sq(rho, lambda_0_nm=lambda_0_nm)


def lambda_from_model(
    T: NDArray[np.float64] | list[float] | float,
    *,
    rho_function: Callable[..., ArrayLikeFloat],
    lambda_0_nm: float,
    **kwargs: float,
) -> ArrayLikeFloat:
    r"""Evaluate :math:`\lambda(T)` from any ``rho_function`` callable."""
    _ = _as_temperature_array(T)
    rho = np.asarray(rho_function(T, **kwargs), dtype=float)
    return rho_to_lambda(rho, lambda_0_nm=lambda_0_nm)

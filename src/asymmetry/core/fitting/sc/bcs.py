r"""BCS reduced-gap helpers used by superconducting sigma(T) models.

The helpers return reduced gap amplitudes :math:`\delta(T)=\Delta(T)/\Delta(0)` for use in

.. math::

    \Delta(T,k)=\Delta_0\,\delta(T/T_c)\,g(k).

This module also resolves gap magnitude conventions between
``gap_ratio = Delta0/(k_B Tc)`` and ``gap_mev`` inputs.

The implementation provides both the Carrington-Manzano interpolation [2]
used as the isotropic s-wave reference and the generalized Gross-style form
[1] used when a symmetry-dependent weak-coupling shape factor is available.

References
----------
[1] R. Prozorov and R. W. Giannetta, Supercond. Sci. Technol. 19, R41 (2006).
[2] A. Carrington and F. Manzano, Physica C 385, 205 (2003).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.sc.constants import BOLTZMANN_CONSTANT_MEV_PER_K

ArrayLikeFloat = NDArray[np.float64]


def _prepare_reduced_gap_output(
    t_reduced: NDArray[np.float64] | list[float] | float,
) -> tuple[ArrayLikeFloat, ArrayLikeFloat, NDArray[np.bool_]]:
    t_arr = np.asarray(t_reduced, dtype=float)
    out = np.zeros_like(t_arr, dtype=float)

    low_mask = t_arr <= 0.0
    high_mask = t_arr >= 1.0
    mid_mask = (~low_mask) & (~high_mask)

    out[low_mask] = 1.0
    out[high_mask] = 0.0
    return t_arr, out, mid_mask


def delta_bcs(t_reduced: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return reduced BCS gap from the Carrington-Manzano interpolation.

    This uses the Carrington-Manzano approximation:

    delta(T) = tanh(1.82 * [1.018 * (1/t - 1)]^0.51),  for 0 < t < 1

    with t = T / Tc.

    The function is pinned to:
    - delta = 1 at t <= 0
    - delta = 0 at t >= 1

    Notes
    -----
    A frequently cited alternative is a Gross-type form [1],

    .. math::

        \Delta_0(T)=\Delta_0(0)\tanh\left[\frac{\pi T_c}{\Delta_0(0)}
        \sqrt{a\left(\frac{T_c}{T}-1\right)}\right],

    where ``a`` is symmetry-dependent. Use :func:`delta_generalized` when a
    symmetry-specific weak-coupling shape factor is available.
    """
    t_arr, out, mid_mask = _prepare_reduced_gap_output(t_reduced)

    if np.any(mid_mask):
        tt = np.maximum(t_arr[mid_mask], 1e-12)
        arg = 1.82 * np.power(1.018 * (1.0 / tt - 1.0), 0.51)
        out[mid_mask] = np.tanh(arg)

    return out


def delta_generalized(
    t_reduced: NDArray[np.float64] | list[float] | float,
    *,
    gap_ratio: float,
    shape_factor: float,
) -> ArrayLikeFloat:
    r"""Return reduced gap from the generalized Gross-style expression.

    In reduced units with :math:`t = T/T_c` and
    :math:`\Delta_0(0)/(k_B T_c)=\text{gap_ratio}` this becomes

    .. math::

        \delta(t)=\tanh\left[\frac{\pi}{\text{gap_ratio}}
        \sqrt{a\left(\frac{1}{t}-1\right)}\right],

    where ``shape_factor`` corresponds to the literature parameter :math:`a`.
    The function is pinned to ``1`` for ``t <= 0`` and ``0`` for ``t >= 1``.
    """
    t_arr, out, mid_mask = _prepare_reduced_gap_output(t_reduced)
    ratio_safe = max(abs(float(gap_ratio)), 1e-12)
    shape = max(float(shape_factor), 0.0)

    if np.any(mid_mask):
        tt = np.maximum(t_arr[mid_mask], 1e-12)
        arg = (np.pi / ratio_safe) * np.sqrt(shape * np.maximum(1.0 / tt - 1.0, 0.0))
        out[mid_mask] = np.tanh(arg)

    return out


def gap_ratio_from_mev(gap_mev: float, tc: float) -> float:
    """Convert ``Delta0`` in meV to ``Delta0/(k_B Tc)``.

    Parameters
    ----------
    gap_mev
        Zero-temperature gap magnitude in meV.
    tc
        Critical temperature in K.
    """
    tc_safe = max(float(tc), 1e-12)
    return float(gap_mev) / (BOLTZMANN_CONSTANT_MEV_PER_K * tc_safe)


def resolve_gap_ratio(
    *,
    tc: float,
    gap_ratio: float | None = None,
    gap_mev: float | None = None,
) -> float:
    r"""Resolve gap ratio from either explicit ratio or meV input.

    If both are provided, ``gap_mev`` takes precedence.

    Returns
    -------
    float
        Dimensionless ratio :math:`\Delta_0/(k_B T_c)`.
    """
    if gap_mev is not None:
        return gap_ratio_from_mev(gap_mev=float(gap_mev), tc=float(tc))
    if gap_ratio is None:
        raise ValueError("Either gap_ratio or gap_mev must be provided")
    return float(gap_ratio)

r"""BCS reduced-gap helpers used by superconducting sigma(T) models.

The key helper returns :math:`\delta_{BCS}(T)=\Delta(T)/\Delta(0)` for use in

.. math::

    \Delta(T,k)=\Delta_0\,\delta_{BCS}(T/T_c)\,g(k).

This module also resolves gap magnitude conventions between
``gap_ratio = Delta0/(k_B Tc)`` and ``gap_mev`` inputs.

The implementation uses the Carrington-Manzano interpolation [2], while the
user guide also documents the Gross-style approximation [1] commonly used for
symmetry-dependent weak-coupling parameterizations.

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


def delta_bcs(t_reduced: NDArray[np.float64] | list[float] | float) -> ArrayLikeFloat:
    r"""Return reduced BCS gap delta(T) = Delta(T) / Delta(0).

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

    where ``a`` is symmetry-dependent. In practice, the Carrington-Manzano
    interpolation used here provides stable and accurate fitting behavior.
    """
    t_arr = np.asarray(t_reduced, dtype=float)
    out = np.zeros_like(t_arr, dtype=float)

    low_mask = t_arr <= 0.0
    high_mask = t_arr >= 1.0
    mid_mask = (~low_mask) & (~high_mask)

    out[low_mask] = 1.0
    out[high_mask] = 0.0

    if np.any(mid_mask):
        tt = np.maximum(t_arr[mid_mask], 1e-12)
        arg = 1.82 * np.power(1.018 * (1.0 / tt - 1.0), 0.51)
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

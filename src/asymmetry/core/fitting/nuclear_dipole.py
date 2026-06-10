"""Zero-field polarization functions for a muon dipolar-coupled to one nucleus.

Ports of WiMDA's single-dipole user functions (``dipolarfunctions.dpr``):
``Dipolar ZF PCR`` / ``Proton dip ZF PCR`` / ``Electron dip ZF PCR`` (all the
Meier spin-1/2 pair polarization with different parameterisations of the
dipolar frequency) and ``Dip gen ZF PCR`` (the Celio-Meier closed form for a
muon coupled to a single nucleus of arbitrary spin ``J`` with both dipolar and
quadrupolar interactions).

Physics references:

* P. F. Meier, Hyperfine Interact. 17-19, 427 (1984) — spin-1/2 pair; the same
  polycrystalline average appears as eqn 4.80 of Blundell, De Renzi, Lancaster
  & Pratt, *Muon Spectroscopy* (OUP, 2022) (cited below as MS-Intro):

  .. math::

      \\langle P_z(t)\\rangle = \\tfrac{1}{6}\\left[1 + \\cos\\omega_d t
        + 2\\cos\\tfrac{\\omega_d t}{2} + 2\\cos\\tfrac{3\\omega_d t}{2}\\right]

  with the dipolar frequency :math:`\\hbar\\omega_d = \\mu_0\\hbar^2
  \\gamma_\\mu\\gamma_j / 4\\pi r^3` (MS-Intro eqn 4.76).

* M. Celio and P. F. Meier, Hyperfine Interact. 17-19, 435 (1984) — general
  spin ``J`` with quadrupole splitting.

Conventions follow the rest of :mod:`asymmetry.core.fitting`: time in µs,
frequencies in MHz (the ``2π`` explicit), fields in Gauss, distances in Å,
rates in µs⁻¹. The transverse damping ``lambda_T`` applies only to the
oscillating 5/6 part of the pair polarization (the non-oscillating 1/6 term
comes from field components parallel to the muon spin, which do not dephase) —
matching WiMDA's ``(1 + e^{-\\lambda t}(\\dots))/6`` form.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.muon_fluorine.dipolar import omega_dipolar_rad_per_us
from asymmetry.core.utils.constants import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

#: Electron gyromagnetic ratio gamma_e / 2pi in MHz/T, derived from the
#: rad/(µs·G) constant so the two cannot drift apart.
ELECTRON_GYROMAGNETIC_RATIO_MHZ_PER_T = (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / (2.0 * np.pi) / GAUSS_TO_TESLA
)

_TWO_PI = 2.0 * np.pi


def dipolar_pair_kernel(
    t: NDArray[np.float64], omega_d_rad_per_us: float, lambda_T: float
) -> NDArray[np.float64]:
    """Meier spin-1/2 pair polarization with transverse damping on the oscillation.

    P(t) = (1/6) [1 + e^{-lambda_T t} (2 cos(w t/2) + cos(w t) + 2 cos(3 w t/2))]

    MS-Intro eqn 4.80 / Meier (1984); WiMDA's ``ZFdipole`` kernel.
    """
    t_arr = np.abs(np.asarray(t, dtype=float))
    wt = float(omega_d_rad_per_us) * t_arr
    damp = np.exp(np.clip(-abs(float(lambda_T)) * t_arr, -700, 0))
    return (1.0 + damp * (2.0 * np.cos(0.5 * wt) + np.cos(wt) + 2.0 * np.cos(1.5 * wt))) / 6.0


def dipolar_pair_field(
    t: NDArray[np.float64], B_dip: float, lambda_T: float
) -> NDArray[np.float64]:
    """Spin-1/2 dipole pair parameterised by the dipolar field ``B_dip`` (Gauss).

    ``omega_d = gamma_mu B_dip`` — WiMDA's ``Dipolar ZF PCR``.
    """
    omega_d = _TWO_PI * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * abs(float(B_dip))
    return dipolar_pair_kernel(t, omega_d, lambda_T)


def proton_dipole(t: NDArray[np.float64], r: float, lambda_T: float) -> NDArray[np.float64]:
    """Spin-1/2 dipole pair for a proton at distance ``r`` (Å).

    ``omega_d`` is computed from first principles (MS-Intro eqn 4.76) with the
    proton gyromagnetic ratio — WiMDA's ``Proton dip ZF PCR`` instead uses the
    empirical field constant ``c = 5.05`` its own source flags as approximate.
    """
    omega_d = omega_dipolar_rad_per_us(
        r, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T, PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    return dipolar_pair_kernel(t, omega_d, lambda_T)


def electron_dipole(t: NDArray[np.float64], r: float, lambda_T: float) -> NDArray[np.float64]:
    """Spin-1/2 dipole pair for an electron moment at distance ``r`` (Å).

    As :func:`proton_dipole` with the electron gyromagnetic ratio (WiMDA's
    ``Electron dip ZF PCR``).  Appropriate for a muon near a single localized
    electronic spin-1/2 moment in zero field.
    """
    omega_d = omega_dipolar_rad_per_us(
        r, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T, ELECTRON_GYROMAGNETIC_RATIO_MHZ_PER_T
    )
    return dipolar_pair_kernel(t, omega_d, lambda_T)


def dipolar_spin_j(
    t: NDArray[np.float64], f_dip: float, f_quad: float, J: float
) -> NDArray[np.float64]:
    """ZF polycrystalline polarization for a muon coupled to a single spin-``J`` nucleus.

    Closed-form eigen-solution of the muon + spin-``J`` system with dipolar
    coupling ``f_dip`` (MHz) and quadrupolar splitting ``f_quad`` (MHz),
    averaged as ``(P_z + 2 P_x)/3`` for a polycrystal — the Celio-Meier result
    (Hyperfine Interact. 17-19, 435 (1984)); WiMDA's ``Dip gen ZF PCR``.

    For ``J = 1/2`` the quadrupole interaction is inactive (a spin-1/2 nucleus
    has no quadrupole moment) and the function reduces to the Meier pair
    (:func:`dipolar_pair_kernel` with ``lambda_T = 0``).

    ``J`` must be a positive half-integer; non-half-integer values are rounded
    to the nearest half-integer (as WiMDA's ``round(2 p3)``).  ``J`` is intended
    to be held fixed during fits.
    """
    t_arr = np.asarray(t, dtype=float)
    wd = _TWO_PI * float(f_dip)
    wq = _TWO_PI * float(f_quad)
    j2 = max(int(round(2.0 * float(J))), 1)  # 2J
    jval = j2 / 2.0

    n_lev = j2 + 2  # index i = 0 .. 2J+1
    lam_p = np.empty(n_lev)
    lam_m = np.empty(n_lev)
    csq2a = np.empty(n_lev)
    for i in range(n_lev):
        m = -jval + i
        q1 = (wq + wd) * (2.0 * m - 1.0)
        arg = jval * (jval + 1.0) - m * (m - 1.0)
        q2 = wd * np.sqrt(max(arg, 0.0))
        qq = q1 * q1 + q2 * q2
        q3 = wq * (2.0 * m * m - 2.0 * m + 1.0) + wd
        wm = np.sqrt(qq)
        lam_p[i] = 0.5 * (q3 + wm) if i < j2 + 1 else wq * jval * jval - wd * jval
        lam_m[i] = 0.5 * (q3 - wm) if i > 0 else wq * jval * jval - wd * jval
        csq2a[i] = (q1 * q1 / qq) if qq > 0.0 else 0.0
    ssq2a = 1.0 - csq2a
    csqa = 0.5 * (1.0 + np.sqrt(csq2a))
    ssqa = 1.0 - csqa

    pz = np.ones_like(t_arr)
    for i in range(1, j2 + 1):
        pz = pz + csq2a[i] + ssq2a[i] * np.cos((lam_p[i] - lam_m[i]) * t_arr)

    px = np.zeros_like(t_arr)
    for i in range(j2 + 1):
        px = (
            px
            + csqa[i + 1] * ssqa[i] * np.cos((lam_p[i + 1] - lam_p[i]) * t_arr)
            + csqa[i + 1] * csqa[i] * np.cos((lam_p[i + 1] - lam_m[i]) * t_arr)
            + ssqa[i + 1] * ssqa[i] * np.cos((lam_m[i + 1] - lam_p[i]) * t_arr)
            + ssqa[i + 1] * csqa[i] * np.cos((lam_m[i + 1] - lam_m[i]) * t_arr)
        )

    return np.asarray((pz + 2.0 * px) / (3.0 * (j2 + 1)), dtype=float)


__all__ = [
    "ELECTRON_GYROMAGNETIC_RATIO_MHZ_PER_T",
    "dipolar_pair_kernel",
    "dipolar_pair_field",
    "proton_dipole",
    "electron_dipole",
    "dipolar_spin_j",
]

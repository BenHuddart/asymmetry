"""Muonium oscillation line-shapes, ported from WiMDA's muonium functions.

Faithful ports of WiMDA's ``TFMuonium`` / ``LowTFMuonium`` / ``ZFmuonium``
(``src/Extrafunctions/muoniumfunctions.dpr``), adapted to Asymmetry conventions:

* frequencies in MHz, time in µs, the ``2π`` factor explicit;
* **phase in radians** (WiMDA uses degrees);
* g-factors taken from :mod:`asymmetry.core.utils.constants` rather than the
  literals ``gm = 0.01355342`` / ``ge = 2.8024`` (they agree to 6 figures);
* these return the *normalised* oscillation; the leading amplitude ``A`` is
  applied by the composite-model wrapper (as for every other component).

**Positive-frequency (same-phase) convention.** WiMDA's ``TFMuonium`` uses the
*signed* transition frequency ``w12`` (which is negative), so its
``cos(2π w12 t + φ)`` puts the lower satellite at phase ``−φ``; yet WiMDA's own
``LowTFMuonium`` *negates* ``w12`` to make it positive — an internal
inconsistency. Physically the muonium precession lines share one initial phase,
so here every line uses ``|w|`` (positive frequency, ``+φ`` for all lines). This
is the deliberate, documented deviation from ``TFMuonium``'s literal signed form
(it matches ``LowTFMuonium``'s negation and same-phase data).

The transverse-field functions take the applied field ``B`` (Gauss) and the
hyperfine coupling ``A_hf`` (MHz); the central diamagnetic Mu⁺ line is **not**
included here (model it separately with ``OscillatoryField``), matching WiMDA.

In the shallow-donor regime (small ``A_hf``) ``TFMuonium`` reduces to two
satellites straddling ``ν_d = γ_µ·B`` with separation ``≈ A_hf``: the two extra
transitions carry weight ``(1−δ) → 0``. For that shallow-donor case the muonium
satellites are better fit as three independent oscillating lines (WiMDA's own
recommendation, = Asymmetry's link groups); these components target genuine
muonium where all transitions and their relative weights matter. See
``docs/porting/muonium-triplet/``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.utils.constants import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

#: Muon and electron gyromagnetic ratios in MHz/G (WiMDA's ``gm`` and ``ge``).
G_MU_MHZ_PER_G = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
G_E_MHZ_PER_G = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / (2.0 * np.pi)

_TWO_PI = 2.0 * np.pi


def _tf_levels(field: float, A_hf: float) -> tuple[float, float, float, float, float]:
    """Return ``(delta, E1, E2, E3, E4)`` for the TF muonium level structure.

    Mirrors WiMDA: ``x = (g_e+g_µ)·B/A_hf`` (with ``x → 1e20`` when ``A_hf ≤ 0``,
    as WiMDA guards), ``δ = x/√(1+x²)``, and the four energy levels ``E_i``.

    ``x`` is clamped to ``±1e20`` (WiMDA's own saturation sentinel) so that a
    pathologically small ``A_hf`` can never overflow ``x²`` to ``inf`` and yield
    a NaN curve — keeping the model finite for every trial value the minimiser
    might probe.
    """
    if A_hf > 0.0:
        x = (G_E_MHZ_PER_G + G_MU_MHZ_PER_G) * field / A_hf
    else:
        x = 1.0e20
    x = float(np.clip(x, -1.0e20, 1.0e20))
    d = (G_E_MHZ_PER_G - G_MU_MHZ_PER_G) / (G_E_MHZ_PER_G + G_MU_MHZ_PER_G)
    root = np.sqrt(1.0 + x * x)
    delta = x / root
    e1 = A_hf / 4.0 * (1.0 + 2.0 * d * x)
    e2 = A_hf / 4.0 * (-1.0 + 2.0 * root)
    e3 = A_hf / 4.0 * (1.0 - 2.0 * d * x)
    e4 = A_hf / 4.0 * (-1.0 - 2.0 * root)
    return delta, e1, e2, e3, e4


def tf_muonium(
    t: NDArray[np.float64], field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    """Normalised transverse-field muonium oscillation (WiMDA ``TFMuonium``).

    Four Mu⁰ transitions ``w12, w14, w34, w23`` with weights ``(1±δ)``; returns
    ``¼·Σ (1±δ)·cos(2π |w| t + φ)`` (positive-frequency convention, see module
    docstring). ``field`` in Gauss, ``A_hf`` in MHz.
    """
    t = np.asarray(t, dtype=float)
    delta, e1, e2, e3, e4 = _tf_levels(field, A_hf)
    w12, w14, w34, w23 = abs(e1 - e2), abs(e1 - e4), abs(e3 - e4), abs(e2 - e3)
    return 0.25 * (
        (1.0 + delta) * np.cos(_TWO_PI * w12 * t + phase)
        + (1.0 - delta) * np.cos(_TWO_PI * w14 * t + phase)
        + (1.0 + delta) * np.cos(_TWO_PI * w34 * t + phase)
        + (1.0 - delta) * np.cos(_TWO_PI * w23 * t + phase)
    )


def low_tf_muonium(
    t: NDArray[np.float64], field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    """Normalised low-field TF muonium oscillation (WiMDA ``LowTFMuonium``).

    Two transitions ``w12, w23`` (positive-frequency convention; WiMDA's
    ``LowTFMuonium`` likewise negates ``w12`` so its lower line is positive).
    """
    t = np.asarray(t, dtype=float)
    delta, e1, e2, e3, _e4 = _tf_levels(field, A_hf)
    w12, w23 = abs(e1 - e2), abs(e2 - e3)
    return 0.25 * (
        (1.0 + delta) * np.cos(_TWO_PI * w12 * t + phase)
        + (1.0 - delta) * np.cos(_TWO_PI * w23 * t + phase)
    )


def high_tf_muonium(
    t: NDArray[np.float64], field: float, A_hf: float, phase: float
) -> NDArray[np.float64]:
    """Normalised high-field TF muonium pair (WiMDA ``High TF Muonium``).

    At high transverse field only the intratriplet ``nu_12`` and ``nu_34``
    transitions carry weight (the ``1 - delta`` lines vanish as ``delta -> 1``),
    and ``nu_12 + nu_34 = A_hf`` exactly.  Both lines are given equal weight
    ``1/2`` — the WiMDA convention, exact in the ``x = B/B_0 >> 1`` limit.
    Use the full :func:`tf_muonium` when the field is low enough that the
    ``(1 ± delta)`` weights or the other two transitions matter.
    """
    t = np.asarray(t, dtype=float)
    _delta, e1, e2, e3, e4 = _tf_levels(field, A_hf)
    w12, w34 = abs(e1 - e2), abs(e3 - e4)
    return 0.5 * (np.cos(_TWO_PI * w12 * t + phase) + np.cos(_TWO_PI * w34 * t + phase))


#: Gauss-Legendre node count for the anisotropic powder average; 32 nodes keep
#: the average converged to ~1e-10 for any D/A_hf the fit will visit (WiMDA uses
#: a 15-point midpoint rule).
_ANISO_POWDER_NODES = 32


def high_tf_muonium_aniso(
    t: NDArray[np.float64], field: float, A_hf: float, D: float, phase: float
) -> NDArray[np.float64]:
    """Powder-averaged anisotropic high-field TF muonium pair (WiMDA ``PCR Hi TF Mu``).

    As :func:`high_tf_muonium`, with an axially anisotropic hyperfine component
    ``D``: for symmetry-axis angle ``theta`` to the field the two lines shift by
    ``±d/2`` with ``d = (D/2)(3 cos^2 theta - 1)`` (first-order anisotropy), and
    the polycrystalline (PCR) average is taken over ``cos theta``.  In the
    isotropic-part/traceless-part decomposition of the axial hyperfine tensor
    (MS-Intro eqn 4.68) ``A_hf`` is the isotropic coupling ``A_iso`` and ``D``
    the axial (traceless) component.  ``D = 0`` reduces exactly to
    :func:`high_tf_muonium`.

    WiMDA's parameter slot for the phase is occupied by ``D`` in this function;
    here the phase is kept as an explicit parameter.
    """
    t = np.asarray(t, dtype=float)
    _delta, e1, e2, e3, e4 = _tf_levels(field, A_hf)
    w12, w34 = abs(e1 - e2), abs(e3 - e4)
    nodes, weights = np.polynomial.legendre.leggauss(_ANISO_POWDER_NODES)
    # Map nodes from (-1, 1) to cos(theta) in (0, 1); weights normalised to 1.
    ct = 0.5 * (nodes + 1.0)
    wt = 0.5 * weights
    d_shift = 0.5 * D * (3.0 * ct * ct - 1.0)
    f_hi = w34 + 0.5 * d_shift  # (n_theta,)
    f_lo = w12 - 0.5 * d_shift
    cos_sum = 0.5 * (
        np.cos(_TWO_PI * np.outer(f_hi, t) + phase) + np.cos(_TWO_PI * np.outer(f_lo, t) + phase)
    )
    return np.asarray(wt @ cos_sum, dtype=float)


#: Vacuum-muonium hyperfine constant in MHz (WiMDA hard-codes 4464).
VACUUM_MUONIUM_A_HF_MHZ = 4463.302


def muonium_lf_relaxation(
    t: NDArray[np.float64], delta_ex: float, tau_c: float, B_LF: float, A_hf: float
) -> NDArray[np.float64]:
    """Muonium longitudinal-field T1 relaxation, exp(-lambda(B) t) (WiMDA ``Mu LF reln``).

    Spin-lattice relaxation of muonium polarization by a fluctuating coupling
    (nuclear-hyperfine modulation from muonium hopping, or electron spin
    exchange) of amplitude ``delta_ex`` (µs⁻¹) and correlation time ``tau_c``
    (µs), sampled at the intratriplet ``nu_12`` transition (BPP/Redfield form):

        lambda(B) = (1 - delta) delta_ex^2 tau_c / (1 + (omega_12 tau_c)^2),

    with ``omega_12 = 2 pi nu_12`` and ``delta = x/sqrt(1+x^2)`` the Breit-Rabi
    mixing factor.  ``nu_12`` is computed from the *exact* Breit-Rabi levels
    (:func:`_tf_levels`); WiMDA instead uses an approximate ``nu_12`` built with
    ``(gamma_e - gamma_mu)`` in both the linear and square-root terms (a
    convention also found in the quantum-diffusion literature) and contains a
    2π unit inconsistency — this implementation deliberately re-derives rather
    than transliterates (see docs/porting/wimda-fit-function-parity/).

    The ``(1 - delta)`` prefactor quenches the relaxation as the muon
    repolarizes in high longitudinal field; together with the growing
    ``omega_12`` it reproduces the LF quenching curves used to extract hop
    rates.  ``A_hf`` defaults to vacuum muonium and is normally held fixed.

    References: R. F. Kiefl et al., Phys. Rev. Lett. 62, 792 (1989);
    R. Kadono et al., Phys. Rev. Lett. 64, 665 (1990).
    """
    t = np.asarray(t, dtype=float)
    _d, e1, e2, _e3, _e4 = _tf_levels(B_LF, A_hf)
    nu12 = abs(e1 - e2)  # MHz
    omega12_tau = _TWO_PI * nu12 * abs(float(tau_c))
    lam = (1.0 - _d) * float(delta_ex) ** 2 * abs(float(tau_c)) / (1.0 + omega12_tau * omega12_tau)
    return np.exp(np.clip(-lam * np.abs(t), -700.0, 0.0))


def zf_muonium(
    t: NDArray[np.float64], A_hf: float, D: float, f_cut: float, phase: float
) -> NDArray[np.float64]:
    """Normalised zero-field axial muonium oscillation (WiMDA ``ZFmuonium``).

    Three lines ``f1=A_hf−D``, ``f2=A_hf+D/2``, ``f3=3D/2`` with weights
    ``a1,a2,a3`` (a Lorentzian roll-off above ``f_cut`` when ``f_cut > 0``),
    normalised by ``1/6``.
    """
    t = np.asarray(t, dtype=float)
    f1 = A_hf - D
    f2 = A_hf + D / 2.0
    f3 = 1.5 * D
    if f_cut > 0.0:
        a1 = 1.0 / (1.0 + (f1 / f_cut) ** 2)
        a2 = 2.0 / (1.0 + (f2 / f_cut) ** 2)
        a3 = 2.0 / (1.0 + (f3 / f_cut) ** 2)
    else:
        a1, a2, a3 = 1.0, 2.0, 2.0
    return (
        a1 * np.cos(_TWO_PI * f1 * t + phase)
        + a2 * np.cos(_TWO_PI * f2 * t + phase)
        + a3 * np.cos(_TWO_PI * f3 * t + phase)
    ) / 6.0

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


#: Gauss-Legendre nodes for the anisotropic powder average, computed once at
#: import (WiMDA uses a 15-point midpoint rule; 32-node Gauss-Legendre keeps
#: the average converged for any D/A_hf the fit will visit).
_ANISO_POWDER_NODES = 32
_ANISO_NODES, _ANISO_WEIGHTS = np.polynomial.legendre.leggauss(_ANISO_POWDER_NODES)
#: cos(theta) grid on (0, 1) and weights normalised to 1.
_ANISO_COS_THETA = 0.5 * (_ANISO_NODES + 1.0)
_ANISO_WT = 0.5 * _ANISO_WEIGHTS


# Spin-1/2 operator matrices used by the exact anisotropic-muonium solver.
_SPIN_HALF = (
    0.5 * np.array([[0, 1], [1, 0]], dtype=complex),  # S_x
    0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex),  # S_y
    0.5 * np.array([[1, 0], [0, -1]], dtype=complex),  # S_z
)
_ID2 = np.eye(2, dtype=complex)
#: sigma_x of the muon in the (electron ⊗ muon) product basis.
_SIGMA_X_MU = np.kron(_ID2, 2.0 * _SPIN_HALF[0])


def _aniso_pair_frequencies(
    field: float, A_hf: float, D: float, cos_theta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact muon-spin-flip pair frequencies for axial muonium at each angle.

    Diagonalizes the full 4-level Hamiltonian
    ``H = γ_e B S_z^e − γ_µ B S_z^µ + S^e·A(θ)·S^µ`` (frequency units, MHz)
    for every ``cos θ`` on the powder grid (batched ``eigh``), and selects per
    orientation the two transitions with the largest ``|⟨m|σ_x^µ|n⟩|²``
    amplitude — the muon-spin-flip pair that survives at high field.
    """
    ct = np.asarray(cos_theta, dtype=float)
    st = np.sqrt(np.clip(1.0 - ct * ct, 0.0, None))
    n = ct.shape[0]

    # Axial hyperfine tensor per orientation: A = (A_iso − D/2)·1 + (3D/2)·n̂n̂ᵀ.
    axes = np.stack([st, np.zeros_like(ct), ct], axis=1)  # (n, 3)
    a_tensor = (A_hf - 0.5 * D) * np.eye(3)[None, :, :] + 1.5 * D * np.einsum(
        "ki,kj->kij", axes, axes
    )

    zeeman = G_E_MHZ_PER_G * field * np.kron(_SPIN_HALF[2], _ID2) - G_MU_MHZ_PER_G * (
        field
    ) * np.kron(_ID2, _SPIN_HALF[2])
    pair_ops = np.array(
        [[np.kron(_SPIN_HALF[i], _SPIN_HALF[j]) for j in range(3)] for i in range(3)]
    )  # (3, 3, 4, 4)
    h = zeeman[None, :, :] + np.einsum("kij,ijab->kab", a_tensor, pair_ops)

    evals, evecs = np.linalg.eigh(h)  # (n, 4), (n, 4, 4)
    sig = np.einsum("kam,ab,kbn->kmn", evecs.conj(), _SIGMA_X_MU, evecs)
    weights = np.abs(sig) ** 2  # (n, 4, 4)
    omega = np.abs(evals[:, :, None] - evals[:, None, :])  # (n, 4, 4)

    # Per orientation, take the two strongest distinct transitions (m < n).
    iu = np.triu_indices(4, k=1)
    w_flat = weights[:, iu[0], iu[1]]  # (n, 6)
    f_flat = omega[:, iu[0], iu[1]]
    order = np.argsort(w_flat, axis=1)
    top2 = order[:, -2:]
    f_pair = np.take_along_axis(f_flat, top2, axis=1)  # (n, 2)
    f_lo = f_pair.min(axis=1)
    f_hi = f_pair.max(axis=1)
    assert f_lo.shape == (n,)
    return f_lo, f_hi


def high_tf_muonium_aniso(
    t: NDArray[np.float64], field: float, A_hf: float, D: float, phase: float
) -> NDArray[np.float64]:
    """Powder-averaged anisotropic high-field TF muonium pair (WiMDA ``PCR Hi TF Mu``).

    As :func:`high_tf_muonium`, with an **axially anisotropic** hyperfine
    interaction: the hyperfine tensor is written as an isotropic part
    ``A_hf`` plus a traceless axial part ``D``, and for each crystallite
    orientation ``θ`` on the powder grid the two muon-spin-flip transition
    frequencies are obtained by **exact diagonalization of the 4-level
    Hamiltonian** ``H = γ_e B S_z^e − γ_µ B S_z^µ + S^e·A(θ)·S^µ`` (batched
    over the 32-node Gauss-Legendre ``cos θ`` grid).  Both lines co-shift so
    that the orientation's pair sum tracks the secular effective coupling
    ``A_eff(θ) = A_hf + (D/2)(3cos²θ − 1)``, producing the characteristic
    asymmetric (Pake-like) powder broadening; each line keeps equal weight
    ``1/2`` (the high-field limit, as in :func:`high_tf_muonium`).

    This deliberately diverges from a literal port of WiMDA's
    ``AnisMuoniumPairRot``, which shifts its two (signed) line frequencies by
    a symmetric ``±d/2``: in our positive-frequency convention that moves
    ``ν_12`` in the wrong direction, and the symmetric split is in any case
    only approximate (see
    ``docs/porting/wimda-fit-function-parity/comparison.md``).  ``D = 0``
    reduces exactly to :func:`high_tf_muonium`.

    WiMDA's parameter slot for the phase is occupied by ``D`` in this
    function; here the phase is kept as an explicit parameter.
    """
    t = np.asarray(t, dtype=float)
    f_lo, f_hi = _aniso_pair_frequencies(field, A_hf, D, _ANISO_COS_THETA)
    cos_sum = 0.5 * (
        np.cos(_TWO_PI * np.outer(f_hi, t) + phase) + np.cos(_TWO_PI * np.outer(f_lo, t) + phase)
    )
    return np.asarray(_ANISO_WT @ cos_sum, dtype=float)


#: Vacuum-muonium hyperfine constant in MHz (WiMDA hard-codes 4464).
VACUUM_MUONIUM_A_HF_MHZ = 4463.302


def muonium_lf_relaxation(
    t: NDArray[np.float64], delta_ex: float, tau_c: float, B_LF: float, A_hf: float
) -> NDArray[np.float64]:
    """Muonium longitudinal-field T1 relaxation, exp(-lambda(B) t) (WiMDA ``Mu LF reln``).

    Spin-lattice relaxation of muonium polarization by a fluctuating coupling
    (nuclear-hyperfine modulation from muonium hopping, or electron spin
    exchange) of amplitude ``delta_ex`` (MHz ≡ µs⁻¹, no 2π) and correlation time ``tau_c``
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

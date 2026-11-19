"""Dimensionless muonium transition kernels and relaxation envelope.

Frequencies are in MHz, time in microseconds and phase in radians. Physical
constants come from :mod:`asymmetry.core.utils.constants`; composite wrappers
apply the asymmetry amplitude. Selected-transition kernels need not start at
one: low TF retains half the initial intensity, while the axial ZF component
omits an additional constant sixth in its unfiltered, zero-phase definition.

The isotropic TF functions use positive transition frequencies with a shared
cosine phase. This is a fitting convention; the independent normalized spin
observable is established at phase zero, not at arbitrary common phase.
The axial tensor and selected high-field approximations are specified by
individual functions. A diamagnetic signal is a separate component.
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
    """Selected low-field TF transition pair, with initial value 1/2 at zero phase.

    Two intratriplet transitions ``w12, w23`` use exact positive frequencies.
    Omitting the other pair does not renormalize the retained intensity.
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


def low_tf_pair_frequencies(field: float, A_hf: float) -> tuple[float, float]:
    """Return the low-TF muonium pair ``(nu_12, nu_23)`` in MHz.

    The two lines rendered by :func:`low_tf_muonium`, sorted low/high by the
    level structure (``nu_12 <= nu_23`` for positive ``field`` and ``A_hf``).
    """
    _delta, e1, e2, e3, _e4 = _tf_levels(field, A_hf)
    return abs(e1 - e2), abs(e2 - e3)


def high_tf_pair_frequencies(field: float, A_hf: float) -> tuple[float, float]:
    """Return the high-TF muonium pair ``(nu_12, nu_34)`` in MHz.

    The two lines rendered by :func:`high_tf_muonium`; they satisfy
    ``nu_12 + nu_34 == A_hf`` exactly in the Paschen–Back regime.
    """
    _delta, e1, e2, e3, e4 = _tf_levels(field, A_hf)
    return abs(e1 - e2), abs(e3 - e4)


def a_hf_from_low_tf_pair(
    field_gauss: float,
    f_lo_mhz: float,
    f_hi_mhz: float,
    *,
    a_hf_range_mhz: tuple[float, float] = (10.0, 4700.0),
) -> float | None:
    """Invert the Breit–Rabi map: hyperfine constant from a low-TF line pair.

    The low-TF pair splitting ``nu_23 - nu_12 = A_hf*(sqrt(1+x^2) - 1)`` with
    ``x = (g_e+g_mu)*B/A_hf`` decreases monotonically in ``A_hf`` (from
    ``(g_e+g_mu)*B`` towards zero), so the root is bracketed and unique when it
    exists. Returns ``None`` when the observed splitting is outside the range
    reachable within ``a_hf_range_mhz`` (wrong-pair guard rather than an error:
    callers probe candidate peak pairs speculatively).
    """
    splitting = float(f_hi_mhz) - float(f_lo_mhz)
    if field_gauss <= 0.0 or splitting <= 0.0:
        return None

    def _mismatch(a_hf: float) -> float:
        lo, hi = low_tf_pair_frequencies(field_gauss, a_hf)
        return (hi - lo) - splitting

    a_min, a_max = a_hf_range_mhz
    g_min, g_max = _mismatch(a_min), _mismatch(a_max)
    if g_min == 0.0:
        return a_min
    if g_max == 0.0:
        return a_max
    if g_min * g_max > 0.0:
        return None

    from scipy.optimize import brentq

    return float(brentq(_mismatch, a_min, a_max, xtol=1e-6, rtol=1e-10))


#: Gauss-Legendre nodes for the anisotropic powder average, computed once at
#: import. Finite angular sampling does not guarantee convergence for
#: arbitrarily large anisotropy-times-time products.
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

    The pair uses positive transition frequencies and the secular effective
    coupling for each orientation; it is not a symmetric signed-frequency
    offset. ``D = 0`` reduces exactly to :func:`high_tf_muonium`. The phase is
    an explicit parameter independent of ``D``.
    """
    t = np.asarray(t, dtype=float)
    f_lo, f_hi = _aniso_pair_frequencies(field, A_hf, D, _ANISO_COS_THETA)
    cos_sum = 0.5 * (
        np.cos(_TWO_PI * np.outer(f_hi, t) + phase) + np.cos(_TWO_PI * np.outer(f_lo, t) + phase)
    )
    return np.asarray(_ANISO_WT @ cos_sum, dtype=float)


def muonium_lf_relaxation(
    t: NDArray[np.float64], delta_ex: float, tau_c: float, B_LF: float, A_hf: float
) -> NDArray[np.float64]:
    """Intratriplet muonium T1 envelope from fluctuating nuclear hyperfine fields.

    Kadono and Ito, JPSJ 94, 064601 (2025), Eq.(22), attributed there to
    Kadono et al., PRL 64, 665 (1990): lambda=delta_ex²*tau_c /
    [1+(omega_12*tau_c)²]. The dominant intratriplet angular frequency is
    omega_12=2*pi*nu_12; nu_12 comes from exact isotropic Breit-Rabi levels.
    delta_ex is an angular-rate amplitude in inverse microseconds, tau_c in
    microseconds, field in Gauss and A_hf in MHz. There is no extra 2*pi
    conversion on delta_ex and no (1-delta) rate prefactor.

    The complete observable in Eq.(19) includes a separate repolarization
    amplitude. This function supplies only its exponential decay envelope;
    it does not describe general electron spin exchange or all transitions.
    """
    t = np.asarray(t, dtype=float)
    _d, e1, e2, _e3, _e4 = _tf_levels(B_LF, A_hf)
    nu12 = abs(e1 - e2)  # MHz
    omega12_tau = _TWO_PI * nu12 * abs(float(tau_c))
    lam = float(delta_ex) ** 2 * abs(float(tau_c)) / (1.0 + omega12_tau * omega12_tau)
    return np.exp(np.clip(-lam * np.abs(t), -700.0, 0.0))


def zf_muonium(
    t: NDArray[np.float64], A_hf: float, D: float, f_cut: float, phase: float
) -> NDArray[np.float64]:
    """Defined three-line axial muonium component, omitting an additional static 1/6.

    Three lines ``f1=A_hf−D``, ``f2=A_hf+D/2``, ``f3=3D/2`` with weights
    ``a1,a2,a3`` (a Lorentzian roll-off above ``f_cut`` when ``f_cut > 0``),
    divided by six. At phase=cutoff=0 the initial value is 5/6; adding 1/6
    gives the ideal static powder polarization for tensor (A-D,A-D,A+2D).
    The cutoff is an empirical amplitude-response approximation, not a full
    causal detector filter. High-TF anisotropic muonium uses twice this D.
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

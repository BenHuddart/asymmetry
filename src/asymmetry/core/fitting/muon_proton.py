"""Defined isotropic electron-muon-proton contact Hamiltonian and RF branches.

In cyclic frequency units H/h=A_mu*S_e.S_mu+A_p*S_e.S_p
+(g_e*S_ez-g_mu*S_muz-g_p*S_pz)*B, with g=gamma/(2*pi) in MHz/G.
The fitted branches are sorted gaps E7-E5 and E8-E6 (one-based labels).
Exact diagonalisation does not establish transition strengths or visibility:
no RF matrix elements, populations or additional resonances are computed.
McKenzie et al., J. Phys. Chem. B 117, 13614 (2013), supplies experimental
radical spectroscopy context, not a global proof of this branch selection.
"""

from __future__ import annotations

from math import hypot

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.muonium import G_E_MHZ_PER_G, G_MU_MHZ_PER_G
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

#: Proton gyromagnetic ratio γ_p/2π in MHz/G.
G_P_MHZ_PER_G = PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA

#: Defined field-scan bracket (Gauss) for the resonance root search.
_RES_FIELD_MIN_G = 1.0
_RES_FIELD_MAX_G = 2000.0
#: Coarse grid resolution used to bracket sign changes before Brent refinement.
_RES_SCAN_POINTS = 256


def _spin_half_ops() -> tuple[NDArray[np.complex128], ...]:
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz


def _build_operators() -> tuple[NDArray[np.complex128], ...]:
    """Cartesian spin operators for electron, muon and proton in the 8-dim basis.

    Ordering of the Kronecker product is electron ⊗ muon ⊗ proton.
    """
    sx, sy, sz = _spin_half_ops()
    i2 = np.eye(2, dtype=complex)

    def k3(a: NDArray, b: NDArray, c: NDArray) -> NDArray[np.complex128]:
        return np.kron(np.kron(a, b), c)

    s_e = [k3(s, i2, i2) for s in (sx, sy, sz)]
    s_mu = [k3(i2, s, i2) for s in (sx, sy, sz)]
    s_p = [k3(i2, i2, s) for s in (sx, sy, sz)]
    se_dot_smu = sum(s_e[i] @ s_mu[i] for i in range(3))
    se_dot_sp = sum(s_e[i] @ s_p[i] for i in range(3))
    return s_e[2], s_mu[2], s_p[2], se_dot_smu, se_dot_sp


_SEZ, _SMUZ, _SPZ, _SE_DOT_SMU, _SE_DOT_SP = _build_operators()
#: Field-independent Zeeman operator (MHz per Gauss): γₑSₑz − γ_µS_µz − γ_pSₚz.
#: The single source of the Zeeman term, shared by the scalar Hamiltonian and the
#: batched level solver so the two can never drift apart.
_ZEEMAN_PER_G = G_E_MHZ_PER_G * _SEZ - G_MU_MHZ_PER_G * _SMUZ - G_P_MHZ_PER_G * _SPZ


def _hyperfine_part(A_mu: float, A_p: float) -> NDArray[np.complex128]:
    """Field-independent hyperfine block ``A_µ·Sₑ·S_µ + A_p·Sₑ·S_p`` (MHz)."""
    return float(A_mu) * _SE_DOT_SMU + float(A_p) * _SE_DOT_SP


def mu_proton_hamiltonian(field: float, A_mu: float, A_p: float) -> NDArray[np.complex128]:
    """Return the 8×8 muon+electron+proton Hamiltonian (MHz) at one field.

    ``field`` in Gauss, ``A_mu`` / ``A_p`` in MHz. See the module docstring for
    the operator form. :func:`mu_proton_levels` is the batched eigenvalue path
    and shares the same hyperfine/Zeeman operators, so ``eigvalsh`` of this matrix
    equals one row of ``mu_proton_levels`` (asserted in the tests).
    """
    return _hyperfine_part(A_mu, A_p) + float(field) * _ZEEMAN_PER_G


def mu_proton_levels(
    field: NDArray[np.float64] | float, A_mu: float, A_p: float
) -> NDArray[np.float64]:
    """Sorted energy levels (MHz) of the mu+e+p system at one or many fields.

    For a scalar ``field`` returns the 8 ascending eigenvalues; for an array of
    fields returns an ``(n, 8)`` array (one ascending row per field), computed
    with a single batched :func:`numpy.linalg.eigvalsh`.
    """
    b = np.atleast_1d(np.asarray(field, dtype=float))
    h = _hyperfine_part(A_mu, A_p)[None, :, :] + b[:, None, None] * _ZEEMAN_PER_G[None, :, :]
    evals = np.linalg.eigvalsh(h)  # ascending along the last axis -> (n, 8)
    if np.isscalar(field) or np.ndim(field) == 0:
        return np.asarray(evals[0], dtype=float)
    return np.asarray(evals, dtype=float)


def rf_transition_freqs(
    field: NDArray[np.float64] | float, A_mu: float, A_p: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The two selected sorted-energy gaps (MHz) vs field.

    Returns ``(f1, f2)`` for the pairs ``E₇−E₅`` and ``E₈−E₆``
    (one-based labels); visibility is not computed. Both
    are returned as arrays matching the shape of ``field`` (0-d arrays for a
    scalar input).
    """
    levels = mu_proton_levels(field, A_mu, A_p)
    if levels.ndim == 1:
        return np.asarray(levels[6] - levels[4]), np.asarray(levels[7] - levels[5])
    return levels[:, 6] - levels[:, 4], levels[:, 7] - levels[:, 5]


def analytic_rf_transition_freqs(field: float, A_mu: float, A_p: float) -> tuple[float, float]:
    """Signed RF gaps from first-order proton perturbations, in MHz.

    Diagonalising the electron-muon part of the module's contact Hamiltonian
    gives a mixed-state separation R=hypot(A_mu, (g_e+g_mu)*B). The upper
    mixed state has electron projection (g_e+g_mu)*B/(2*R); the aligned
    |up,up> state has projection 1/2. Their unperturbed gap is
    [A_mu+(g_e-g_mu)*B-R]/2. A proton with m_p=+1/2 or -1/2 adds
    A_p*m_p times the difference of those projections. Its Zeeman term
    cancels because m_p is conserved in each transition.

    ``field`` is in Gauss; both couplings are in MHz. Return the +1/2 proton
    sector first, without absolute values or sorting. The mixed state is the
    higher m_z=0 eigenstate for either sign of A_mu. At A_mu=B=0 use the
    B=0, A_mu -> 0+ triplet continuation (zero electron projection).
    These diagonal first-order shifts do not resolve degenerate perturbations
    or predict visibility. The fitted model uses :func:`rf_transition_freqs`,
    which diagonalises the full three-spin Hamiltonian instead.
    """
    b = float(field)
    a = float(A_mu)
    zeeman_sum = (G_E_MHZ_PER_G + G_MU_MHZ_PER_G) * b
    separation = hypot(a, zeeman_sum)
    electron_polarisation = zeeman_sum / separation if separation else 0.0
    # Rationalise R-A_mu to retain the small-field gap when A_mu is positive.
    mixing_shift = (
        zeeman_sum * (zeeman_sum / (separation + a)) if a >= 0.0 and separation else separation - a
    )
    centre = ((G_E_MHZ_PER_G - G_MU_MHZ_PER_G) * b - mixing_shift) / 2.0
    splitting = float(A_p) * (1.0 - electron_polarisation) / 4.0
    return centre + splitting, centre - splitting


def _lowest_ascending_root(
    transition: NDArray[np.float64],
    grid: NDArray[np.float64],
    target: float,
    selector: int,
    A_mu: float,
    A_p: float,
) -> float:
    """First field where ``transition`` crosses ``target`` on an *ascending* branch.

    The RF transitions rise monotonically through the resonance window before
    saturating (and can turn over at very high field), so the physical resonance
    is the lowest upward crossing. Returns ``nan`` when no such crossing exists in
    the bracket (the caller then drops that Lorentzian, keeping the model finite
    for any trial parameters the minimiser probes).

    Boundary behaviour: as a coupling is lowered, a transition's peak drops below
    ``target`` and the crossing first appears at the saturating (near-vertical)
    branch — so just inside the bracketable region ``dB/dA`` is very large and just
    outside the root is ``nan``. The resonance field is therefore ill-conditioned
    in a thin strip next to the ``nan`` cliff (well away from realistic couplings).
    A finite-difference Jacobian that straddles the cliff sees one ``nan`` field,
    but :func:`rf_resonance_mup` still returns a finite curve there (it drops the
    affected Lorentzian), so the model never feeds ``nan`` to a least-squares step;
    the practical consequence is a locally flat gradient, hence the recommendation
    to start near the expected couplings.
    """
    from scipy.optimize import brentq

    shifted = transition - target
    for i in range(shifted.shape[0] - 1):
        if shifted[i] < 0.0 <= shifted[i + 1]:
            return float(
                brentq(
                    lambda bb: float(rf_transition_freqs(bb, A_mu, A_p)[selector]) - target,
                    grid[i],
                    grid[i + 1],
                    xtol=1e-10,
                    rtol=8.9e-16,
                )
            )
    return float("nan")


def rf_resonance_fields(
    A_mu: float,
    A_p: float,
    nu_RF: float,
    *,
    field_min: float = _RES_FIELD_MIN_G,
    field_max: float = _RES_FIELD_MAX_G,
    scan_points: int = _RES_SCAN_POINTS,
) -> tuple[float, float]:
    """Resonance fields ``(B₁, B₂)`` (Gauss) where the two RF transitions = ``ν_RF``.

    ``B₁`` is the resonance of the ``E₇−E₅`` transition and ``B₂`` that of
    ``E₈−E₆`` (WiMDA's ``FreqDiff1exact`` / ``FreqDiff2exact``); for the benzene
    cyclohexadienyl radical (A_µ ≈ 514.78, A_p ≈ 124.6 MHz, ν_RF = 218.5 MHz)
    these are ≈ 894 G and ≈ 797 G respectively. A field that cannot be bracketed
    is returned as ``nan``.
    """
    grid = np.linspace(float(field_min), float(field_max), int(scan_points))
    f1, f2 = rf_transition_freqs(grid, A_mu, A_p)
    target = float(nu_RF)
    b1 = _lowest_ascending_root(f1, grid, target, 0, A_mu, A_p)
    b2 = _lowest_ascending_root(f2, grid, target, 1, A_mu, A_p)
    return b1, b2


def _lorentzian_peak(
    x: NDArray[np.float64], centre: float, ampl: float, width: float
) -> NDArray[np.float64]:
    if not np.isfinite(centre):
        return np.zeros_like(x)
    w2 = max(abs(float(width)), 1e-9) ** 2
    return float(ampl) * w2 / (w2 + (x - float(centre)) ** 2)


def rf_resonance_mup(
    x: NDArray[np.float64] | float,
    A_mu: float,
    A_p: float,
    nu_RF: float,
    ampl1: float,
    wid1: float,
    ampl2: float,
    wid2: float,
    BG: float,
) -> NDArray[np.float64]:
    """Field-swept RF-µSR resonance curve for the mu+e+p radical (exact diag.).

    Follows WiMDA's ``RFresonanceMuPlusProtonExact``: two Lorentzian features at
    the exact-diagonalisation resonance fields ``B₁(A_µ, A_p, ν_RF)`` and
    ``B₂(...)`` on a flat background,

        y(B) = BG + ampl1·wid1²/(wid1² + (B − B₁)²)
                  + ampl2·wid2²/(wid2² + (B − B₂)²).

    ``x`` is the swept field in Gauss; ``A_mu``/``A_p``/``nu_RF`` in MHz; the
    Lorentzian amplitudes are in the asymmetry unit of the scan and the widths in
    Gauss. ``ampl`` may be negative to fit resonance **dips** (the usual
    Red−Green RF observable). Designed as a parameter-vs-field trend component
    (see :mod:`asymmetry.core.fitting.parameter_models`).
    """
    xx = np.asarray(x, dtype=float)
    b1, b2 = rf_resonance_fields(A_mu, A_p, nu_RF)
    out = np.full_like(xx, float(BG))
    out = out + _lorentzian_peak(xx, b1, ampl1, wid1)
    out = out + _lorentzian_peak(xx, b2, ampl2, wid2)
    return out

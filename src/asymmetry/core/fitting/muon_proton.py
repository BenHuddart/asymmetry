"""Muon + electron + proton (radical) spin Hamiltonian and RF-µSR resonance fit.

Faithful port of WiMDA's ``RigiWorkshopFit`` energy-level / RF-resonance models
(``wimda installation/user fitting/RigiWorkshopfit.dpr`` + the ``Eigenuni.pas``
Hermitian eigensolver), adapted to Asymmetry conventions:

* energies / hyperfine couplings in **MHz**, applied field in **Gauss**;
* gyromagnetic ratios taken from :mod:`asymmetry.core.utils.constants` and
  :mod:`asymmetry.core.fitting.muonium` rather than WiMDA's literals
  (``g_mu = 0.01355342``, ``g_e = 2.80249514``, ``g_p = 0.00425764`` MHz/G —
  they agree to ~5 figures);
* WiMDA's bespoke Jacobi/Householder eigensolvers (``eigen``..``eigen4``) are
  replaced by :func:`numpy.linalg.eigvalsh` — the spectrum of a Hermitian
  matrix is basis-independent, so the sorted eigenvalues (and therefore every
  level *difference* WiMDA selects) are identical to machine precision.

**The model.** A muoniated radical such as cyclohexadienyl (C₆H₆Mu) couples the
muon spin **and** one dominant proton spin to the unpaired electron. The
isotropic contact Hamiltonian in the (electron ⊗ muon ⊗ proton) product basis,
in frequency units (MHz), is

    H = A_µ · Sₑ·S_µ + A_p · Sₑ·S_p
        + (γₑ Sₑz − γ_µ S_µz − γ_p S_pz) · B,

where ``A_µ`` is the muon hyperfine coupling, ``A_p`` the proton hyperfine
coupling, and ``B`` the applied (longitudinal) field. Note the proton couples to
the *electron*, not directly to the muon — the standard radical topology. With
``A_p = 0`` the proton decouples and the 8 levels collapse pairwise onto the
4 Breit-Rabi muonium levels of :func:`asymmetry.core.fitting.muonium._tf_levels`
(plus the bare proton Zeeman splitting), which is the cross-check the unit tests
assert.

**RF-µSR resonance.** In an RF-µSR experiment a fixed-frequency RF field ``ν_RF``
is applied while the static field ``B`` is swept; a resonance (an asymmetry dip
or peak) occurs whenever an RF-driven muon-spin-flip transition matches ``ν_RF``.
For the radical there are **two** such transitions, between the sorted-eigenvalue
pairs ``E₇−E₅`` and ``E₈−E₆`` (WiMDA's ``RFresonanceMuPlusProtonExact``
selectors ``75`` and ``86``). Their resonance fields ``B₁`` and ``B₂`` bracket
the W-shaped double dip; the mean tracks ``A_µ`` and the splitting tracks
``A_p``, so a single field-swept scan determines **both** hyperfine constants
simultaneously. This is the capability flagged as parity gap PC1 against WiMDA
(``docs/testing/parity-checks.md`` on the testing branch); see
``docs/porting/rf-musr-resonance-fit/``.

The high-field-limit linear map ``B_res = (ν_RF ± ½A_µ)/γ_µ`` and even WiMDA's
own first-order **analytic** levels (:func:`analytic_rf_transition_freqs`) are
inaccurate at the low fields (hundreds of G) where these scans are taken — hence
the exact diagonalisation. Reference: I. McKenzie, R. Scheuermann, S. P.
Cottrell, J. S. Lord, and I. M. Tucker, J. Phys. Chem. B 117, 13614 (2013).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from asymmetry.core.fitting.muonium import G_E_MHZ_PER_G, G_MU_MHZ_PER_G
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

#: Proton gyromagnetic ratio γ_p/2π in MHz/G (WiMDA's ``g_p = 0.00425764``).
G_P_MHZ_PER_G = PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA

#: Default field-scan bracket (Gauss) for the resonance root search; mirrors
#: WiMDA's ``zbrent(...,0,2000,...)`` window.
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

    Ordering of the Kronecker product is (electron ⊗ muon ⊗ proton), matching the
    sign pattern of WiMDA's diagonal in ``muproton1``.
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
    """The two RF-driven muon-spin-flip transition frequencies (MHz) vs field.

    Returns ``(f1, f2)`` for the sorted-eigenvalue pairs ``E₇−E₅`` and ``E₈−E₆``
    (1-indexed; WiMDA's exact-diagonalisation selectors ``75`` and ``86``). Both
    are returned as arrays matching the shape of ``field`` (0-d arrays for a
    scalar input).
    """
    levels = mu_proton_levels(field, A_mu, A_p)
    if levels.ndim == 1:
        return np.asarray(levels[6] - levels[4]), np.asarray(levels[7] - levels[5])
    return levels[:, 6] - levels[:, 4], levels[:, 7] - levels[:, 5]


def analytic_rf_transition_freqs(field: float, A_mu: float, A_p: float) -> tuple[float, float]:
    """WiMDA's first-order **analytic** mu+p RF transitions (``MuoniumPlusProton``).

    Breit-Rabi muonium levels with a first-order-in-``A_p`` proton correction
    (WiMDA ``E11−E21`` and ``E12−E22``). Provided for cross-checking only — it is
    inaccurate at the low fields of RF-µSR scans, which is exactly why the exact
    diagonalisation in :func:`rf_transition_freqs` is the production path.
    """
    b = float(field)
    gg = G_E_MHZ_PER_G + G_MU_MHZ_PER_G
    n_mu = G_MU_MHZ_PER_G * b
    n_e = G_E_MHZ_PER_G * b
    n_p = G_P_MHZ_PER_G * b
    a = float(A_mu)
    ap = float(A_p)
    # Clamp x to WiMDA's saturation sentinel (as ``muonium._tf_levels`` does) so a
    # pathologically small A_mu cannot overflow x² to inf for any minimiser probe.
    xx = gg * b / a if a > 0.0 else 1.0e20
    xx = float(np.clip(xx, -1.0e20, 1.0e20))
    yy = np.sqrt(1.0 + xx * xx)
    delta = xx / yy
    e11 = a / 4.0 + 0.5 * (n_e - n_mu - n_p) + ap / 4.0
    e12 = e11 + n_p - ap / 2.0
    e21 = -a / 4.0 + a / 2.0 * yy - n_p / 2.0 + ap / 4.0 * delta
    e22 = e21 + n_p - ap / 2.0 * delta
    return e11 - e21, e12 - e22


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

    Port of WiMDA ``RFresonanceMuPlusProtonExact``: two Lorentzian features at
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

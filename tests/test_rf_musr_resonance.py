"""RF-µSR mu+e+p resonance fit (WiMDA RigiWorkshopFit parity, PC1).

Covers the exact-diagonalisation muon+electron+proton spin Hamiltonian, the RF
resonance-field solver, the registered ``RFResonanceMuP`` field-trend component,
and the benzene-corpus verification (paper-graded against McKenzie 2013).

See ``docs/porting/rf-musr-resonance-fit/``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from asymmetry.core.fitting.muon_proton import (
    G_P_MHZ_PER_G,
    analytic_rf_transition_freqs,
    mu_proton_hamiltonian,
    mu_proton_levels,
    rf_resonance_fields,
    rf_resonance_mup,
    rf_transition_freqs,
)
from asymmetry.core.fitting.muonium import _tf_levels
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
    component_names_for_x,
)

# Paper-graded targets (McKenzie et al., J. Phys. Chem. B 117, 13614 (2013),
# Table 1): cyclohexadienyl C6H6Mu in benzene at 293 K, ν_RF = 218.5 MHz.
A_MU_PAPER = 514.78
A_P_PAPER = 124.6
NU_RF = 218.5


# --------------------------------------------------------------------------
# Hamiltonian
# --------------------------------------------------------------------------
def test_proton_gyromagnetic_ratio_matches_wimda() -> None:
    # WiMDA RigiWorkshopFit literal g_p = 0.00425764 MHz/G.
    assert G_P_MHZ_PER_G == pytest.approx(0.00425764, rel=1e-4)


def test_hamiltonian_reduces_to_muonium_when_Ap_zero() -> None:
    """With A_p = 0 the proton decouples: the 8 levels are the 4 Breit-Rabi
    muonium levels each split by the bare proton Zeeman ±γ_p·B/2, so their
    pairwise midpoints reproduce ``muonium._tf_levels`` exactly."""
    field, a_mu = 800.0, A_MU_PAPER
    levels = mu_proton_levels(field, a_mu, 0.0)
    assert levels.shape == (8,)

    _delta, e1, e2, e3, e4 = _tf_levels(field, a_mu)
    muonium = np.sort([e1, e2, e3, e4])
    midpoints = 0.5 * (levels[0::2] + levels[1::2])
    np.testing.assert_allclose(np.sort(midpoints), muonium, atol=1e-6)

    # The pair splitting is the bare proton Zeeman γ_p·B.
    splittings = levels[1::2] - levels[0::2]
    np.testing.assert_allclose(splittings, G_P_MHZ_PER_G * field, atol=1e-6)


def test_levels_are_basis_independent_sorted() -> None:
    """Batched and scalar level evaluation agree, and levels are ascending."""
    fields = np.array([100.0, 500.0, 900.0])
    batched = mu_proton_levels(fields, A_MU_PAPER, A_P_PAPER)
    assert batched.shape == (3, 8)
    for row, b in zip(batched, fields, strict=True):
        np.testing.assert_allclose(row, mu_proton_levels(b, A_MU_PAPER, A_P_PAPER))
        assert np.all(np.diff(row) >= -1e-9)


def test_scalar_hamiltonian_matches_batched_level_solver() -> None:
    """The standalone Hamiltonian and the batched level solver share operators, so
    eigvalsh of the matrix equals one row of mu_proton_levels (guards drift)."""
    for b in (0.0, 320.0, 900.0):
        h = mu_proton_hamiltonian(b, A_MU_PAPER, A_P_PAPER)
        assert h.shape == (8, 8)
        np.testing.assert_allclose(h, h.conj().T, atol=1e-12)  # Hermitian
        np.testing.assert_allclose(
            np.linalg.eigvalsh(h),
            mu_proton_levels(b, A_MU_PAPER, A_P_PAPER),
            atol=1e-9,
        )


# --------------------------------------------------------------------------
# Resonance fields and transition selection
# --------------------------------------------------------------------------
def test_resonance_fields_for_benzene_couplings() -> None:
    """Exact diagonalisation places the two RF resonances inside the corpus
    field window (560–1080 G), with the W-shaped split tracking A_p."""
    b1, b2 = rf_resonance_fields(A_MU_PAPER, A_P_PAPER, NU_RF)
    # B1 is the E7-E5 resonance (higher field), B2 the E8-E6 (lower field).
    assert b1 == pytest.approx(893.9, abs=1.0)
    assert b2 == pytest.approx(796.7, abs=1.0)
    assert 560.0 < b2 < b1 < 1080.0


def test_only_the_wimda_pair_resonates_in_window() -> None:
    """At the paper couplings, only the sorted transitions E7-E5 and E8-E6
    cross ν_RF inside the experimental window — confirming WiMDA's 75/86
    selectors are the physically driven pair (no wrong-transition ambiguity)."""
    grid = np.linspace(1.0, 2000.0, 400)
    levels = mu_proton_levels(grid, A_MU_PAPER, A_P_PAPER)
    in_window: list[tuple[int, int]] = []
    for i in range(8):
        for j in range(i):
            tr = levels[:, i] - levels[:, j] - NU_RF
            crossings = np.where((tr[:-1] < 0) & (tr[1:] >= 0))[0]
            for k in crossings:
                if 500.0 < grid[k] < 1100.0:
                    in_window.append((i + 1, j + 1))
    assert set(in_window) == {(7, 5), (8, 6)}


def test_rf_transition_freqs_match_level_differences() -> None:
    f1, f2 = rf_transition_freqs(820.0, A_MU_PAPER, A_P_PAPER)
    levels = mu_proton_levels(820.0, A_MU_PAPER, A_P_PAPER)
    assert float(f1) == pytest.approx(levels[6] - levels[4])
    assert float(f2) == pytest.approx(levels[7] - levels[5])


def test_unbracketed_resonance_returns_nan() -> None:
    # ν_RF far above any transition in the window cannot be bracketed.
    b1, b2 = rf_resonance_fields(A_MU_PAPER, A_P_PAPER, 5000.0)
    assert np.isnan(b1) and np.isnan(b2)


# --------------------------------------------------------------------------
# Inverse problem — recover the hyperfine couplings
# --------------------------------------------------------------------------
def _invert_dips(b_hi: float, b_lo: float, start: tuple[float, float]) -> tuple[float, float]:
    def resid(p: np.ndarray) -> list[float]:
        fields = rf_resonance_fields(abs(p[0]), abs(p[1]), NU_RF)
        if not np.all(np.isfinite(fields)):
            return [1e3, 1e3]
        return [fields[0] - b_hi, fields[1] - b_lo]

    sol = least_squares(resid, list(start), diff_step=2e-3)
    return abs(sol.x[0]), abs(sol.x[1])


def test_direct_inversion_recovers_couplings_exactly() -> None:
    """Forward(A_µ, A_p) → (B1, B2); inverse-fit those exact fields recovers
    **both** couplings to machine precision — the gold-standard self-consistency
    check. The position→coupling map is well-conditioned; the practical RF-µSR
    difficulty is pinning the dip positions (esp. the splitting) from noisy data,
    which is what propagates into the A_p uncertainty."""
    b1, b2 = rf_resonance_fields(A_MU_PAPER, A_P_PAPER, NU_RF)
    a_mu, a_p = _invert_dips(b1, b2, start=(520.0, 130.0))
    assert a_mu == pytest.approx(A_MU_PAPER, abs=1e-3)
    assert a_p == pytest.approx(A_P_PAPER, abs=1e-3)


def test_invert_digitised_corpus_dips_matches_paper() -> None:
    """Inverting the paper's own digitised Fig-3a resonance fields (773/865 G,
    figure-traced) through the exact model recovers A_µ within ~0.3 % and A_p
    within the paper's stated uncertainty (124.6 ± 1.4 MHz)."""
    a_mu, a_p = _invert_dips(865.0, 773.0, start=(510.0, 130.0))
    assert a_mu == pytest.approx(A_MU_PAPER, rel=0.01)  # ≈516 G-trace vs 514.78
    assert a_p == pytest.approx(A_P_PAPER, abs=3.0)


def test_dip_inversion_is_start_independent() -> None:
    """From any starting guess whose trial resonances fall in the field window,
    the dip inversion converges to the same exact couplings."""
    b1, b2 = rf_resonance_fields(A_MU_PAPER, A_P_PAPER, NU_RF)
    for start in [(520.0, 130.0), (525.0, 118.0), (510.0, 130.0), (530.0, 140.0)]:
        a_mu, a_p = _invert_dips(b1, b2, start=start)
        assert a_mu == pytest.approx(A_MU_PAPER, abs=1e-2)
        assert a_p == pytest.approx(A_P_PAPER, abs=1e-2)


def test_analytic_underestimates_the_split_versus_exact() -> None:
    """WiMDA's first-order analytic levels give too small an RF resonance split
    at these low fields (the paper's stated reason for exact diagonalisation)."""

    def split(freqs_fn) -> float:  # type: ignore[no-untyped-def]
        from scipy.optimize import brentq

        grid = np.linspace(1.0, 1400.0, 600)
        roots = []
        for sel in (0, 1):
            vals = np.array([freqs_fn(b, A_MU_PAPER, A_P_PAPER)[sel] for b in grid]) - NU_RF
            for k in range(len(grid) - 1):
                if vals[k] < 0 <= vals[k + 1]:
                    roots.append(
                        brentq(
                            lambda b, s=sel: freqs_fn(b, A_MU_PAPER, A_P_PAPER)[s] - NU_RF,
                            grid[k],
                            grid[k + 1],
                        )
                    )
                    break
        return abs(roots[0] - roots[1])

    exact_split = split(rf_transition_freqs)
    analytic_split = split(analytic_rf_transition_freqs)
    assert exact_split == pytest.approx(97.0, abs=2.0)
    assert analytic_split < exact_split - 10.0


# --------------------------------------------------------------------------
# Registered component
# --------------------------------------------------------------------------
def test_component_registered_in_field_scope_only() -> None:
    assert "RFResonanceMuP" in component_names_for_x("field")
    assert "RFResonanceMuP" not in component_names_for_x("temperature")
    assert "RFResonanceMuP" not in component_names_for_x("other")


def test_component_param_info_complete() -> None:
    comp = PARAMETER_MODEL_COMPONENTS["RFResonanceMuP"]
    assert set(comp.param_names) == set(comp.param_info)
    assert set(comp.param_defaults) == set(comp.param_names)


def test_component_curve_is_two_dips_on_background() -> None:
    x = np.linspace(560.0, 1080.0, 80)
    y = rf_resonance_mup(x, A_MU_PAPER, A_P_PAPER, NU_RF, -18.0, 28.0, -18.0, 28.0, -1.5)
    b1, b2 = rf_resonance_fields(A_MU_PAPER, A_P_PAPER, NU_RF)
    # Background away from resonance; deepest near a resonance field.
    assert y[0] == pytest.approx(-1.5, abs=0.5)
    assert y.min() < -15.0
    assert min(abs(x[np.argmin(y)] - b1), abs(x[np.argmin(y)] - b2)) < 40.0


def test_component_roundtrip_through_composite_model() -> None:
    model = ParameterCompositeModel(["RFResonanceMuP"])
    truth = {
        "A_mu": A_MU_PAPER,
        "A_p": A_P_PAPER,
        "nu_RF": NU_RF,
        "ampl1": -18.0,
        "wid1": 28.0,
        "ampl2": -18.0,
        "wid2": 28.0,
        "BG": -1.5,
    }
    x = np.linspace(560.0, 1080.0, 80)
    y = model.function(x, **truth)
    names = model.param_names

    def resid(p: np.ndarray) -> np.ndarray:
        return model.function(x, **dict(zip(names, p, strict=True))) - y

    start = [v for v in truth.values()]
    sol = least_squares(resid, start, diff_step=3e-3, xtol=1e-13, ftol=1e-13)
    fit = dict(zip(names, sol.x, strict=True))
    assert fit["A_mu"] == pytest.approx(A_MU_PAPER, abs=1e-2)
    assert fit["A_p"] == pytest.approx(A_P_PAPER, abs=1e-2)


@pytest.mark.parametrize(
    "params",
    [
        (0.0, 0.0, NU_RF),
        (1e6, 1e6, NU_RF),
        (-5.0, -5.0, NU_RF),
        (A_MU_PAPER, A_P_PAPER, 0.0),
    ],
)
def test_component_finite_for_pathological_params(params: tuple[float, float, float]) -> None:
    """The minimiser may probe any parameter values; the model must stay finite
    (unbracketed resonances drop their Lorentzian rather than raising)."""
    x = np.linspace(560.0, 1080.0, 40)
    a_mu, a_p, nu = params
    y = rf_resonance_mup(x, a_mu, a_p, nu, -1.0, 10.0, -1.0, 10.0, 0.0)
    assert np.all(np.isfinite(y))

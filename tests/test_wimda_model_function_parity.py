"""Tests for the WiMDA Model-layer (parameter-trending) parity components.

Verification criteria from docs/porting/model-function-parity/: numerical
oracles transcribed from WiMDA's ``fitfunctions.pas``, exact round-trips,
composite-recipe identities (with each documented divergence asserted as a
behavioural test), and registry/doc hygiene. Oracle values: test-data.md.
"""

from __future__ import annotations

import re
from math import comb

import numpy as np
import pytest

from asymmetry.core.fitting.component_docs import (
    PARAMETER_MODEL_APPLICABILITY,
    get_component_applicability,
    get_component_references,
)
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
    _arrhenius,
    _critical_divergence,
    _lcr_lorentzian,
    _mu_repolarisation,
    _order_parameter,
    _polynomial,
    _power_law_quad_bg,
    component_names_for_x,
    fit_parameter_model,
    isotropic_mu_b0_gauss,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.utils.constants import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

NEW_COMPONENTS = ("Polynomial", "PowerLawQuadBG", "MuRepolarisation")

# (γₑ + γ_μ)/2π in MHz/G, assembled from the same constants the component uses.
GAMMA_SUM_MHZ_PER_G = (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / (2.0 * np.pi)
    + MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
)

VACUUM_MU_A_HF_MHZ = 4463.302765


# ---------------------------------------------------------------------------
# WiMDA-transcribed oracles (fitfunctions.pas)
# ---------------------------------------------------------------------------


def test_polynomial_matches_wimda_func0_oracle() -> None:
    coeffs = dict(c0=0.5, c1=-1.2, c2=0.8, c3=-0.05, c4=0.002, c5=-0.0001)
    x = np.array([0.0, 1.0, 2.5, 7.0, 10.0])
    expected = [0.5, 0.0519, 1.787109375, 17.2713, 28.5]
    np.testing.assert_allclose(_polynomial(x, **coeffs), expected, rtol=1e-12)


def test_power_law_quad_bg_matches_wimda_oracle() -> None:
    x = np.array([0.5, 1.0, 4.0, 9.0])
    expected = [
        3.082207001484488,
        3.605551275463989,
        16.278820596099706,
        54.08326913195984,
    ]
    np.testing.assert_allclose(_power_law_quad_bg(x, 2.0, 1.5, 3.0), expected, rtol=1e-12)


def test_power_law_quad_bg_limits() -> None:
    # y(0) -> |BG| (the |x| floor makes the power-law term negligible)
    np.testing.assert_allclose(_power_law_quad_bg(np.array([0.0]), 2.0, 1.5, 3.0), [3.0])
    # BG = 0 reduces to the plain power law
    x = np.array([0.3, 2.0, 7.0])
    np.testing.assert_allclose(_power_law_quad_bg(x, 2.0, 1.5, 0.0), 2.0 * x**1.5, rtol=1e-12)


def test_mu_repolarisation_matches_wimda_muonrep_oracle() -> None:
    # WiMDA muonrep fits B0 directly; identical curve with B0 = A_hf/(γₑ+γ_μ).
    a_mu, a_dia = 15.0, 8.0
    b0 = isotropic_mu_b0_gauss(VACUUM_MU_A_HF_MHZ)
    x = np.array([0.0, 100.0, b0, 5000.0, 20000.0])
    r2 = (x / b0) ** 2
    wimda = a_mu * (0.5 + r2) / (1.0 + r2) + a_dia  # transcription of muonrep
    np.testing.assert_allclose(
        _mu_repolarisation(x, a_mu, VACUUM_MU_A_HF_MHZ, a_dia), wimda, rtol=1e-14
    )
    # Spot values from test-data.md
    np.testing.assert_allclose(wimda[0], 15.5, rtol=1e-12)  # a_Mu/2 + a_Dia
    np.testing.assert_allclose(wimda[1], 15.529737440954086, rtol=1e-10)
    np.testing.assert_allclose(wimda[2], 19.25, rtol=1e-12)  # 3/4 point at B0


def test_mu_repolarisation_b0_from_constants() -> None:
    # Vacuum muonium: B0 = A/(γₑ+γ_μ) ≈ 1585 G ≈ 0.1585 T.
    b0 = isotropic_mu_b0_gauss(VACUUM_MU_A_HF_MHZ)
    np.testing.assert_allclose(b0, VACUUM_MU_A_HF_MHZ / GAMMA_SUM_MHZ_PER_G, rtol=1e-14)
    np.testing.assert_allclose(b0, 1584.952, atol=0.01)


def test_mu_repolarisation_limits() -> None:
    a_mu, a_dia = 12.0, 3.0
    y = _mu_repolarisation(np.array([0.0, 1.0e7]), a_mu, VACUUM_MU_A_HF_MHZ, a_dia)
    np.testing.assert_allclose(y[0], a_mu / 2.0 + a_dia, rtol=1e-12)
    np.testing.assert_allclose(y[1], a_mu + a_dia, rtol=1e-6)


# ---------------------------------------------------------------------------
# Exact fit round-trips
# ---------------------------------------------------------------------------


def test_polynomial_exact_round_trip_quintic() -> None:
    true = dict(c0=0.5, c1=-1.2, c2=0.8, c3=-0.05, c4=0.002, c5=-0.0001)
    x = np.linspace(-3.0, 9.0, 60)
    y = _polynomial(x, **true)
    yerr = np.full_like(x, 0.01)
    model = ParameterCompositeModel(["Polynomial"])
    params = ParameterSet([Parameter(name, value=0.1) for name in true])
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    for name, value in true.items():
        np.testing.assert_allclose(fitted[name], value, rtol=1e-5, atol=1e-8)


def test_polynomial_quadratic_with_fixed_tail() -> None:
    x = np.linspace(0.0, 10.0, 30)
    y = _polynomial(x, c0=2.0, c1=-0.5, c2=0.25)
    yerr = np.full_like(x, 0.01)
    model = ParameterCompositeModel(["Polynomial"])
    params = ParameterSet(
        [
            Parameter("c0", value=0.0),
            Parameter("c1", value=0.0),
            Parameter("c2", value=0.0),
            Parameter("c3", value=0.0, fixed=True),
            Parameter("c4", value=0.0, fixed=True),
            Parameter("c5", value=0.0, fixed=True),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(
        [fitted["c0"], fitted["c1"], fitted["c2"]], [2.0, -0.5, 0.25], atol=1e-6
    )
    assert fitted["c3"] == 0.0 and fitted["c4"] == 0.0 and fitted["c5"] == 0.0


def test_mu_repolarisation_recovers_hyperfine_constant() -> None:
    true = dict(a_Mu=14.0, A_hf=2000.0, a_Dia=6.0)
    x = np.geomspace(5.0, 20000.0, 40)
    y = _mu_repolarisation(x, **true)
    yerr = np.full_like(x, 0.02)
    model = ParameterCompositeModel(["MuRepolarisation"])
    params = ParameterSet(
        [
            Parameter("a_Mu", value=10.0),
            Parameter("A_hf", value=4000.0, min=1.0, max=1.0e5),
            Parameter("a_Dia", value=3.0),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(fitted["A_hf"], true["A_hf"], rtol=1e-5)
    np.testing.assert_allclose(fitted["a_Mu"], true["a_Mu"], rtol=1e-6)
    np.testing.assert_allclose(fitted["a_Dia"], true["a_Dia"], rtol=1e-6)


# ---------------------------------------------------------------------------
# Composite recipes (documented divergences asserted, comparison.md D2/D4/D5)
# ---------------------------------------------------------------------------


def test_arrhenius_recipe_matches_wimda_func2_with_constant_correction() -> None:
    """D4: WiMDA's e/k is 0.089 % low; meV + CODATA k_B reproduce func2 exactly
    once the energy is rescaled by the constant ratio."""
    c_wimda = 1.60e-19 / 1.38e-23
    kb_mev = 8.617333262e-2  # CODATA, as used by _arrhenius
    c_codata = 1.0 / (kb_mev * 1.0e-3)
    t = np.array([50.0, 100.0, 200.0, 300.0])
    amp1, ea1_ev, amp2, ea2_ev = 5.0, 0.10, 50.0, 0.25
    wimda = amp1 * np.exp(-ea1_ev * c_wimda / t) + amp2 * np.exp(-ea2_ev * c_wimda / t)
    scale = c_wimda / c_codata  # = 0.99911...; Ea[meV] = 1000*Ea[eV]*scale
    ours = _arrhenius(t, amp1, 1000.0 * ea1_ev * scale) + _arrhenius(
        t, amp2, 1000.0 * ea2_ev * scale
    )
    np.testing.assert_allclose(ours, wimda, rtol=1e-12)
    # The naive 1000x conversion is measurably off: the 0.089 % constant error
    # is amplified by the Boltzmann exponent (≈ Ea·c/T · 8.9e-4 in y).
    naive = _arrhenius(t, amp1, 1000.0 * ea1_ev) + _arrhenius(t, amp2, 1000.0 * ea2_ev)
    assert 1e-4 < np.max(np.abs(naive / wimda - 1.0)) < 0.1
    np.testing.assert_allclose(c_codata / c_wimda - 1.0, 8.897e-4, rtol=1e-3)


def test_order_parameter_plus_constant_matches_wimda_func5() -> None:
    """D5: on the physical domain (T ≥ 0, positive exponents) the WiMDA form
    is identical to OrderParameter + Constant, including the clamp above Tc."""
    b0, tc, alpha, beta, b_bg = 29.9, 69.2, 1.23, 0.417, 3.0
    t = np.array([5.0, 30.0, 60.0, 69.19, 69.2, 80.0])

    def wimda_func5(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for i, xv in enumerate(x):
            if xv > tc:
                out[i] = b_bg
            else:
                q2 = 1.0 - np.abs(xv / tc) ** np.abs(alpha)
                out[i] = b0 * np.abs(q2) ** beta + b_bg
        return out

    ours = _order_parameter(t, y0=b0, Tc=tc, beta=beta, alpha=alpha) + b_bg
    np.testing.assert_allclose(ours, wimda_func5(t), rtol=1e-12)
    # Clamp explicitly: zero order parameter at and above Tc.
    np.testing.assert_allclose(ours[-2:], [b_bg, b_bg], atol=1e-12)


def test_lorentzian_lcr_matches_wimda_peak_term() -> None:
    """The WiMDA '2 Lorentzians + cubic BG' peak Ampl·Wid²/(Wid²+(x−Pos)²) is
    algebraically LorentzianLCR(f=Ampl, B0=Pos, Bwid=Wid)."""
    ampl, pos, wid = 3.5, 1200.0, 80.0
    x = np.linspace(800.0, 1600.0, 81)
    wimda = ampl * wid**2 / (wid**2 + (x - pos) ** 2)
    np.testing.assert_allclose(_lcr_lorentzian(x, ampl, pos, wid), wimda, rtol=1e-12)


def test_recentred_cubic_background_is_a_polynomial() -> None:
    """D2: WiMDA's cubic BG in powers of (x − Pos1) equals Polynomial with
    re-centred coefficients — same model space, different coefficient values."""
    pos1 = 1200.0
    bg = [0.7, -0.01, 5e-5, -2e-8]  # WiMDA BG_0..BG_3 about Pos 1
    x = np.linspace(800.0, 1600.0, 41)
    wimda_bg = sum(b * (x - pos1) ** k for k, b in enumerate(bg))
    # Expand sum_k b_k (x - p)^k into absolute-x coefficients via binomials.
    coeffs = np.zeros(6)
    for k, b in enumerate(bg):
        for j in range(k + 1):
            coeffs[j] += b * comb(k, j) * (-pos1) ** (k - j)
    ours = _polynomial(x, *coeffs)
    np.testing.assert_allclose(ours, wimda_bg, rtol=1e-9)
    assert not np.allclose(coeffs[:4], bg)  # coefficients do NOT transfer 1:1


def test_critical_divergence_matches_widthdiv_off_the_singular_point() -> None:
    """D6: WiMDA WidthDiv ≡ CriticalDivergence (a=scaling, nu=alpha, c=MinRate)
    everywhere except exactly at T = Tc."""
    tc, alpha, min_rate, scaling = 69.2, 0.7, 0.05, 2.0
    t = np.array([40.0, 60.0, 68.0, 70.5, 90.0])
    wimda = min_rate + scaling / np.abs(t - tc) ** alpha
    np.testing.assert_allclose(
        _critical_divergence(t, a=scaling, Tc=tc, nu=alpha, c=min_rate), wimda, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# EuO regression (real-data fixture)
# ---------------------------------------------------------------------------

# ν(T) trend of the EuO ZF series (PSI GPS runs 2928–2943, WiMDA Muon School
# corpus), produced headlessly on 2026-06-10 with the core API: per-run
# Oscillatory * Exponential + Constant fits on t = 0–8 µs (loader-default
# grouping), frequency ± Hesse error per run. Regenerating the table requires
# the corpus; the fixture freezes it so this regression runs anywhere.
EUO_NU_T_TREND: tuple[tuple[float, float, float], ...] = (
    (17.0, 29.221222, 0.058808),
    (24.0, 27.972966, 0.051047),
    (30.0, 26.610159, 0.070142),
    (36.0, 24.881154, 0.065332),
    (41.0, 23.402752, 0.067416),
    (46.0, 21.579090, 0.042962),
    (50.0, 20.049828, 0.066756),
    (52.5, 18.791178, 0.063968),
    (57.5, 16.455647, 0.069198),
    (61.0, 14.236831, 0.068251),
    (63.0, 12.832770, 0.101919),
    (64.5, 11.506486, 0.123985),
    (65.5, 10.642407, 0.099857),
    (66.5, 9.338051, 0.122190),
    (67.5, 7.624838, 0.122544),
    (68.3, 5.493185, 0.170581),
)


def test_order_parameter_recipe_reproduces_euo_beta_extraction() -> None:
    """The OrderParameter recipe on the real EuO ν(T) trend reproduces the
    PR #15 GUI-verified extraction (Tc = 69.2(1) K, β = 0.417(7), α = 1.23(5))
    within combined uncertainties."""
    t = np.array([row[0] for row in EUO_NU_T_TREND])
    nu = np.array([row[1] for row in EUO_NU_T_TREND])
    err = np.array([row[2] for row in EUO_NU_T_TREND])

    model = ParameterCompositeModel(["OrderParameter"])
    params = ParameterSet(
        [
            Parameter("y0", value=30.0, min=0.0, max=100.0),
            Parameter("Tc", value=70.0, min=50.0, max=90.0),
            Parameter("beta", value=0.4, min=0.0, max=2.0),
            Parameter("alpha", value=1.0, min=0.1, max=5.0),
        ]
    )
    result = fit_parameter_model(t, nu, err, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}

    # Regression against this scripted pipeline's own frozen outcome…
    np.testing.assert_allclose(fitted["Tc"], 69.244, atol=0.05)
    np.testing.assert_allclose(fitted["beta"], 0.4085, atol=0.005)
    np.testing.assert_allclose(fitted["alpha"], 1.177, atol=0.03)

    # …and agreement with the PR #15 GUI extraction within 2σ combined.
    for name, ref_value, ref_err in (
        ("Tc", 69.2, 0.1),
        ("beta", 0.417, 0.007),
        ("alpha", 1.23, 0.05),
    ):
        sigma = np.hypot(ref_err, result.uncertainties.get(name, 0.0))
        assert abs(fitted[name] - ref_value) < 2.0 * sigma, (name, fitted[name])


# ---------------------------------------------------------------------------
# Registry and documentation hygiene
# ---------------------------------------------------------------------------


def test_new_components_registered_with_correct_scopes() -> None:
    assert PARAMETER_MODEL_COMPONENTS["Polynomial"].scopes == ("common",)
    assert PARAMETER_MODEL_COMPONENTS["PowerLawQuadBG"].scopes == ("common",)
    assert PARAMETER_MODEL_COMPONENTS["MuRepolarisation"].scopes == ("field",)
    for x_key in ("field", "temperature", "run"):
        names = component_names_for_x(x_key)
        assert "Polynomial" in names
        assert "PowerLawQuadBG" in names
    assert "MuRepolarisation" in component_names_for_x("field")
    assert "MuRepolarisation" not in component_names_for_x("temperature")


def test_new_components_have_param_info_and_defaults() -> None:
    for name in NEW_COMPONENTS:
        comp = PARAMETER_MODEL_COMPONENTS[name]
        assert set(comp.param_names) == set(comp.param_defaults)
        assert set(comp.param_names) == set(comp.param_info)
        # formula_template must format cleanly with parameter values only.
        rendered = comp.formula_template.format(
            **{k: f"{v:g}" for k, v in comp.param_defaults.items()}
        )
        assert rendered
        assert comp.latex_equation


def test_new_components_have_explicit_applicability_text() -> None:
    for name in NEW_COMPONENTS:
        text = PARAMETER_MODEL_APPLICABILITY.get(name, "")
        assert len(text) > 80, name
        assert text == get_component_applicability(name, kind="parameter_model")
        lowered = text.lower()
        for forbidden in ("eqn", "eq.", "ms-intro", "phys. rev."):
            assert forbidden not in lowered, (name, forbidden)


def test_mu_repolarisation_references_are_aps_style() -> None:
    refs = get_component_references("MuRepolarisation", kind="parameter_model")
    assert refs
    assert any("Muon Spectroscopy" in ref for ref in refs)
    for ref in refs:
        # APS style ends with "(year)." (journal) or "year)." (book).
        assert re.search(r"\(.*\d{4}\)\.$", ref), ref


def test_a_hf_lower_bound_keeps_b0_finite() -> None:
    # The registry minimum allows A_hf -> 0 during minimisation; the kernel
    # must stay finite (B0 floor) rather than dividing by zero.
    y = _mu_repolarisation(np.array([100.0]), 10.0, 0.0, 2.0)
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y[0], 12.0)  # fully repolarised limit


@pytest.mark.parametrize("name", NEW_COMPONENTS)
def test_new_components_evaluate_finite_on_scope_grid(name: str) -> None:
    comp = PARAMETER_MODEL_COMPONENTS[name]
    x = np.linspace(0.0, 5000.0, 11) if "field" in comp.scopes else np.linspace(0.0, 300.0, 11)
    y = comp.function(x, **comp.param_defaults)
    assert y.shape == x.shape
    assert np.isfinite(y).all()

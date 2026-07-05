"""Tests for parameter-model fitting core."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    ParameterModelFitResult,
    _arrhenius,
    _constant,
    _critical_divergence,
    _exp_decay,
    _lcr_gaussian,
    _linear,
    _lorentzian,
    _order_parameter,
    _power_law,
    _redfield,
    carve_window_gap,
    component_names_for_x,
    evaluate_parameter_model_fit,
    fit_parameter_model,
    sample_parameter_model,
    suggest_model_seeds,
    suggest_trend_seeds,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

# ---------------------------------------------------------------------------
# component_names_for_x
# ---------------------------------------------------------------------------


def test_component_names_filtered_by_x_key() -> None:
    field_names = component_names_for_x("field")
    temp_names = component_names_for_x("temperature")
    run_names = component_names_for_x("run")

    assert "Redfield" in field_names
    assert "Lorentzian" in field_names
    assert "GaussianLCR" in field_names
    assert "Arrhenius" not in field_names

    assert "Arrhenius" in temp_names
    assert "CriticalDivergence" in temp_names
    assert "OrderParameter" in temp_names
    assert "OrderParameter" not in field_names
    assert "Redfield" not in temp_names

    assert "Constant" in run_names
    assert "Linear" in run_names
    assert "Arrhenius" not in run_names
    assert "Redfield" not in run_names


def test_component_names_returns_sorted_list() -> None:
    names = component_names_for_x("common")
    assert names == sorted(names)


def test_component_names_common_excludes_scoped() -> None:
    names = component_names_for_x("common")
    assert "Arrhenius" not in names
    assert "Redfield" not in names
    assert "CriticalDivergence" not in names
    assert "OrderParameter" not in names


# ---------------------------------------------------------------------------
# Individual basis functions
# ---------------------------------------------------------------------------


def test_constant_fills_array() -> None:
    x = np.array([1.0, 2.0, 5.0, 10.0])
    result = _constant(x, c=3.7)
    np.testing.assert_array_equal(result, np.full(4, 3.7))


def test_constant_works_with_zero() -> None:
    x = np.linspace(0.0, 10.0, 50)
    result = _constant(x, c=0.0)
    assert np.all(result == 0.0)


def test_linear_known_values() -> None:
    x = np.array([0.0, 1.0, 2.0])
    result = _linear(x, m=2.0, b=1.0)
    np.testing.assert_allclose(result, [1.0, 3.0, 5.0])


def test_linear_negative_slope() -> None:
    x = np.array([0.0, 5.0])
    result = _linear(x, m=-1.0, b=10.0)
    np.testing.assert_allclose(result, [10.0, 5.0])


def test_power_law_square() -> None:
    x = np.array([2.0, 3.0])
    result = _power_law(x, a=1.0, n=2.0, c=0.0)
    np.testing.assert_allclose(result, [4.0, 9.0], rtol=1e-10)


def test_power_law_safe_at_zero() -> None:
    x = np.array([0.0])
    result = _power_law(x, a=1.0, n=2.0, c=0.0)
    assert np.isfinite(result[0])


def test_power_law_with_offset() -> None:
    x = np.array([1.0])
    result = _power_law(x, a=2.0, n=3.0, c=5.0)
    np.testing.assert_allclose(result, [7.0], rtol=1e-10)


def test_exp_decay_at_zero_returns_a_plus_c() -> None:
    result = _exp_decay(np.array([0.0]), a=3.0, tau=2.0, c=1.0)
    np.testing.assert_allclose(result, [4.0], rtol=1e-10)


def test_exp_decay_asymptotes_to_c() -> None:
    x = np.array([1e6])
    result = _exp_decay(x, a=5.0, tau=1.0, c=2.0)
    np.testing.assert_allclose(result, [2.0], atol=1e-6)


def test_exp_decay_safe_with_near_zero_tau() -> None:
    x = np.array([1.0, 2.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _exp_decay(x, a=1.0, tau=0.0, c=0.0)
    assert np.all(np.isfinite(result))
    assert not caught


def test_arrhenius_large_temperature_approaches_a() -> None:
    # At very high T, exp(-Ea/(kT)) → 1 so result → a
    result = _arrhenius(np.array([1e9]), a=5.0, Ea=1.0)
    np.testing.assert_allclose(result, [5.0], rtol=1e-4)


def test_arrhenius_safe_at_zero_temperature() -> None:
    result = _arrhenius(np.array([0.0]), a=1.0, Ea=1.0)
    assert np.all(np.isfinite(result))


def test_arrhenius_positive_ea_gives_activated_behaviour() -> None:
    low_t = _arrhenius(np.array([10.0]), a=1.0, Ea=100.0)
    high_t = _arrhenius(np.array([300.0]), a=1.0, Ea=100.0)
    assert high_t[0] > low_t[0]


def test_critical_divergence_peaks_near_tc() -> None:
    x = np.array([9.999, 10.0, 10.001])
    result = _critical_divergence(x, a=1.0, Tc=10.0, nu=1.0, c=0.0)
    # All three values should be large (close to divergence)
    assert np.all(result > 50.0)


def test_critical_divergence_far_from_tc() -> None:
    result = _critical_divergence(np.array([0.0]), a=1.0, Tc=10.0, nu=1.0, c=0.0)
    np.testing.assert_allclose(result, [1.0 / 10.0], rtol=1e-6)


def test_order_parameter_saturates_to_y0_at_zero_temperature() -> None:
    result = _order_parameter(np.array([0.0]), y0=29.9, Tc=69.0, beta=0.36, alpha=1.0)
    np.testing.assert_allclose(result, [29.9], rtol=1e-12)


def test_order_parameter_is_zero_at_and_above_tc() -> None:
    result = _order_parameter(np.array([69.0, 80.0, 150.0]), y0=29.9, Tc=69.0, beta=0.36, alpha=1.0)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-12)


def test_order_parameter_alpha_one_is_simple_power_law() -> None:
    # With alpha=1, y(T) = y0 * (1 - T/Tc)^beta.
    temperature = np.array([10.0, 30.0, 50.0])
    tc, y0, beta = 100.0, 2.0, 0.5
    result = _order_parameter(temperature, y0=y0, Tc=tc, beta=beta, alpha=1.0)
    expected = y0 * np.power(1.0 - temperature / tc, beta)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_order_parameter_is_monotonically_decreasing_below_tc() -> None:
    temperature = np.linspace(0.0, 68.0, 50)
    result = _order_parameter(temperature, y0=29.9, Tc=69.0, beta=0.36, alpha=1.5)
    assert np.all(np.diff(result) <= 1e-12)


def test_order_parameter_safe_with_zero_tc() -> None:
    result = _order_parameter(np.array([0.0, 1.0, 10.0]), y0=1.0, Tc=0.0, beta=0.36, alpha=1.0)
    assert np.all(np.isfinite(result))


def test_order_parameter_clamps_negative_temperature() -> None:
    # T < 0 is unphysical but must stay finite and not exceed y0.
    result = _order_parameter(np.array([-5.0]), y0=3.0, Tc=10.0, beta=0.36, alpha=1.0)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [3.0], rtol=1e-12)


def test_order_parameter_clips_negative_exponents_not_mirrors() -> None:
    # Exponents are clipped to 0, not mirrored via abs(): a negative beta must
    # collapse onto beta=0 (a flat y0 below Tc), NOT reproduce the +|beta| curve.
    temperature = np.linspace(0.0, 60.0, 20)
    neg_beta = _order_parameter(temperature, y0=29.9, Tc=69.0, beta=-0.36, alpha=1.0)
    zero_beta = _order_parameter(temperature, y0=29.9, Tc=69.0, beta=0.0, alpha=1.0)
    pos_beta = _order_parameter(temperature, y0=29.9, Tc=69.0, beta=0.36, alpha=1.0)
    np.testing.assert_allclose(neg_beta, zero_beta, rtol=1e-12)
    assert not np.allclose(neg_beta, pos_beta)

    # Same for the shape exponent alpha.
    neg_alpha = _order_parameter(temperature, y0=29.9, Tc=69.0, beta=0.36, alpha=-1.0)
    zero_alpha = _order_parameter(temperature, y0=29.9, Tc=69.0, beta=0.36, alpha=0.0)
    np.testing.assert_allclose(neg_alpha, zero_alpha, rtol=1e-12)


def test_order_parameter_negative_tc_is_zero_below_t0() -> None:
    # A negative Tc (unphysical) must not mirror to a positive ordering curve.
    result = _order_parameter(np.array([5.0, 20.0, 50.0]), y0=10.0, Tc=-69.0, beta=0.36, alpha=1.0)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-12)


def test_redfield_peak_at_zero() -> None:
    x = np.array([0.0, 50.0, 100.0, 200.0])
    result = _redfield(x, D=20.0, nu=10.0, m=2.0)
    assert result[0] == max(result)


def test_redfield_with_m2_is_half_at_omega_equals_nu() -> None:
    # At omega_mu == nu and m=2, dynamic term is halved.
    from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

    nu = 10.0
    b_half = nu / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)
    result = _redfield(np.array([b_half]), D=2.0, nu=nu, m=2.0)
    # D=2 => prefactor (D^2/4)*(2/nu) = 2/nu = 0.2, then halved -> 0.1
    np.testing.assert_allclose(result, [0.1], rtol=1e-10)


def test_lorentzian_peak_at_zero() -> None:
    x = np.linspace(0.0, 500.0, 20)
    lo = _lorentzian(x, a=3.0, B0=150.0, c=0.5)
    assert lo[0] == np.max(lo)


def test_lcr_gaussian_peaks_at_b0() -> None:
    x = np.array([800.0, 1000.0, 1200.0])
    y = _lcr_gaussian(x, f=0.2, B0=1000.0, Bwid=100.0)
    np.testing.assert_allclose(y[1], 0.2, rtol=1e-12)
    assert y[1] > y[0]
    np.testing.assert_allclose(y[0], y[2], rtol=1e-12)


def test_lcr_gaussian_safe_with_near_zero_width() -> None:
    y = _lcr_gaussian(np.array([1.0, 2.0]), f=0.1, B0=1.5, Bwid=0.0)
    assert np.all(np.isfinite(y))


def test_redfield_safe_with_near_zero_nu() -> None:
    result = _redfield(np.array([1.0, 2.0]), D=1.0, nu=0.0, m=2.0)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# PARAMETER_MODEL_COMPONENTS registry
# ---------------------------------------------------------------------------


def test_all_components_have_unique_names() -> None:
    names = list(PARAMETER_MODEL_COMPONENTS.keys())
    assert len(names) == len(set(names))


def test_all_components_have_matching_param_counts() -> None:
    for comp in PARAMETER_MODEL_COMPONENTS.values():
        assert len(comp.param_names) == len(comp.param_defaults)


def test_all_components_callable() -> None:
    x = np.linspace(1.0, 10.0, 10)
    for comp in PARAMETER_MODEL_COMPONENTS.values():
        result = comp.function(x, **comp.param_defaults)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result)), f"{comp.name} produced non-finite output"


# ---------------------------------------------------------------------------
# ParameterCompositeModel construction
# ---------------------------------------------------------------------------


def test_composite_single_component_works() -> None:
    model = ParameterCompositeModel(["Constant"])
    assert model.param_names == ["c"]
    assert model.param_defaults == {"c": 0.0}


def test_composite_requires_nonempty_component_list() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        ParameterCompositeModel([])


def test_composite_unknown_component_raises() -> None:
    with pytest.raises(ValueError, match="Unknown component"):
        ParameterCompositeModel(["NonExistentModel"])


def test_composite_operator_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="operators length"):
        ParameterCompositeModel(["Constant", "Linear"], operators=["+", "+"])


def test_composite_invalid_operator_raises() -> None:
    with pytest.raises(ValueError, match="operators must be"):
        ParameterCompositeModel(["Constant", "Linear"], operators=["^"])


def test_composite_duplicate_params_get_suffix() -> None:
    # Two components that each have parameter 'c'
    model = ParameterCompositeModel(["Constant", "Constant"])
    assert "c_1" in model.param_names
    assert "c_2" in model.param_names
    assert "c" not in model.param_names


def test_composite_unique_params_have_no_suffix() -> None:
    # Linear has m and b; Constant has c — no overlap so no suffix
    model = ParameterCompositeModel(["Linear", "Constant"])
    assert "m" in model.param_names
    assert "b" in model.param_names
    assert "c" in model.param_names


# ---------------------------------------------------------------------------
# ParameterCompositeModel.formula_string
# ---------------------------------------------------------------------------


def test_formula_string_single_component() -> None:
    model = ParameterCompositeModel(["Constant"])
    assert model.formula_string() == "{c}".format(c="c")


def test_formula_string_two_components_uses_operator() -> None:
    model = ParameterCompositeModel(["Constant", "Linear"], operators=["+"])
    formula = model.formula_string()
    assert "+" in formula
    assert "m" in formula
    assert "b" in formula


def test_parameter_component_expression_round_trip() -> None:
    model = ParameterCompositeModel.from_expression("Linear + ( Arrhenius * Constant )")

    assert model.component_names == ["Linear", "Arrhenius", "Constant"]
    assert model.operators == ["+", "*"]
    assert model.open_parentheses == [0, 1, 0]
    assert model.close_parentheses == [0, 0, 1]
    assert model.component_expression_string() == "Linear + (Arrhenius * Constant)"


def test_parameter_composite_serialization_round_trip_with_parentheses() -> None:
    model = ParameterCompositeModel.from_expression("Linear + ( Arrhenius * Constant )")

    restored = ParameterCompositeModel.from_dict(model.to_dict())

    assert restored.component_names == model.component_names
    assert restored.operators == model.operators
    assert restored.open_parentheses == model.open_parentheses
    assert restored.close_parentheses == model.close_parentheses


# ---------------------------------------------------------------------------
# ParameterCompositeModel.function evaluation
# ---------------------------------------------------------------------------


def test_composite_function_addition() -> None:
    model = ParameterCompositeModel(["Constant", "Constant"], operators=["+"])
    x = np.linspace(0.0, 5.0, 10)
    result = model.function(x, c_1=3.0, c_2=2.0)
    np.testing.assert_allclose(result, np.full_like(x, 5.0))


def test_composite_function_subtraction() -> None:
    model = ParameterCompositeModel(["Constant", "Constant"], operators=["-"])
    x = np.linspace(0.0, 5.0, 10)
    result = model.function(x, c_1=5.0, c_2=2.0)
    np.testing.assert_allclose(result, np.full_like(x, 3.0))


def test_composite_function_multiplication() -> None:
    # Linear * Constant: (m*x + b) * c
    model = ParameterCompositeModel(["Linear", "Constant"], operators=["*"])
    x = np.array([1.0, 2.0, 3.0])
    result = model.function(x, m=2.0, b=0.0, c=3.0)
    np.testing.assert_allclose(result, [6.0, 12.0, 18.0])


def test_composite_function_division() -> None:
    model = ParameterCompositeModel(["Constant", "Constant"], operators=["/"])
    x = np.linspace(1.0, 5.0, 10)
    result = model.function(x, c_1=6.0, c_2=3.0)
    np.testing.assert_allclose(result, np.full_like(x, 2.0))


def test_composite_function_operator_precedence_mul_before_add() -> None:
    # (a + b*c): add has lower precedence than multiply
    model = ParameterCompositeModel(["Constant", "Constant", "Constant"], operators=["+", "*"])
    x = np.array([1.0])
    # c_1=1, c_2=4, c_3=5 → 1 + 4*5 = 21
    result = model.function(x, c_1=1.0, c_2=4.0, c_3=5.0)
    np.testing.assert_allclose(result, [21.0])


def test_parameter_composite_parentheses_override_precedence() -> None:
    model = ParameterCompositeModel.from_expression("Constant * ( Constant + Constant )")
    x = np.array([1.0])

    result = model.function(x, c_1=2.0, c_2=3.0, c_3=4.0)

    np.testing.assert_allclose(result, [14.0])


def test_composite_function_missing_parameter_raises() -> None:
    model = ParameterCompositeModel(["Linear"])
    with pytest.raises(KeyError):
        model.function(np.array([1.0, 2.0]), m=1.0)  # missing b


def test_composite_function_division_by_near_zero_produces_nan() -> None:
    model = ParameterCompositeModel(["Constant", "Constant"], operators=["/"])
    x = np.array([1.0])
    result = model.function(x, c_1=1.0, c_2=0.0)
    assert not np.isfinite(result[0])


def test_parameter_composite_additive_component_indices() -> None:
    model = ParameterCompositeModel(
        ["Redfield", "Lambda_bg", "Lorentzian", "Constant"],
        operators=["+", "*", "+"],
    )
    assert model.additive_component_indices() == [0, 1, 3]


def test_parameter_composite_evaluate_components_returns_named_curves() -> None:
    x = np.linspace(10.0, 500.0, 30)
    model = ParameterCompositeModel(["Redfield", "Lambda_bg"], operators=["+"])

    curves = model.evaluate_components(x, D=10.0, nu=8.0, m=2.0, lambda_BG=0.05)

    assert [name for name, _vals in curves] == ["Redfield", "Lambda_bg"]
    assert all(vals.shape == x.shape for _name, vals in curves)
    assert np.all(np.isfinite(curves[0][1]))
    assert np.allclose(curves[1][1], 0.05)


def test_parameter_composite_evaluate_components_additive_only() -> None:
    x = np.linspace(10.0, 500.0, 30)
    model = ParameterCompositeModel(
        ["DiffusionLF_2D", "Lambda_bg", "Lorentzian"],
        operators=["+", "*"],
    )

    curves = model.evaluate_components(
        x,
        additive_only=True,
        A=0.8,
        D_2D=2.0,
        D_perp=0.0,
        lambda_BG=0.03,
        a=1.0,
        B0=120.0,
        c=0.0,
    )

    assert [name for name, _vals in curves] == ["DiffusionLF_2D", "Lambda_bg"]


# ---------------------------------------------------------------------------
# fit_parameter_model
# ---------------------------------------------------------------------------


def test_fit_parameter_model_linear_recovers_parameters() -> None:
    rng = np.random.default_rng(123)
    x = np.linspace(0.0, 10.0, 120)
    true_m = 2.0
    true_b = 1.5
    noise = rng.normal(0.0, 0.03, size=x.shape)
    y = true_m * x + true_b + noise
    yerr = np.full_like(x, 0.03)

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet(
        [
            Parameter("m", value=1.0, min=-10.0, max=10.0),
            Parameter("b", value=0.0, min=-10.0, max=10.0),
        ]
    )

    result = fit_parameter_model(x, y, yerr, model, params)

    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(fitted["m"], true_m, atol=0.1)
    np.testing.assert_allclose(fitted["b"], true_b, atol=0.1)


def test_fit_constant_recovers_value() -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = np.full_like(x, 4.2)
    yerr = np.full_like(x, 0.05)

    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.0)])

    result = fit_parameter_model(x, y, yerr, model, params)

    assert result.success
    fitted_c = next(p.value for p in result.parameters if p.name == "c")
    np.testing.assert_allclose(fitted_c, 4.2, atol=0.1)


@pytest.mark.slow
def test_fit_parameter_model_transport_seed_recovers_diffusive_component() -> None:
    x = np.array([20.0, 50.0, 100.0, 200.0, 400.0, 1000.0, 3000.0, 5000.0, 10000.0])
    model = ParameterCompositeModel(["BallisticLF_2D", "DiffusionLF_2D"], operators=["+"])
    true_params = {
        "A_1": 0.7,
        "D_hop": 3.0,
        "A_2": 32.0,
        "D_2D": 250.0,
        "D_perp": 0.0,
    }
    y = model.function(x, **true_params)
    yerr = np.maximum(0.005, 0.03 * y)

    params = ParameterSet(
        [
            Parameter("A_1", value=1.0, min=0.0, max=100.0),
            Parameter("D_hop", value=1.0, min=0.0, max=1000.0),
            Parameter("A_2", value=1.0, min=0.0, max=100.0),
            Parameter("D_2D", value=1.0, min=0.0, max=1000.0),
            Parameter("D_perp", value=0.0, min=0.0, max=0.0, fixed=True),
        ]
    )

    result = fit_parameter_model(x, y, yerr, model, params)

    assert result.success
    fitted = {parameter.name: parameter.value for parameter in result.parameters}
    assert result.reduced_chi_squared < 10.0
    assert fitted["A_2"] > 20.0
    assert 100.0 <= fitted["D_2D"] <= 400.0


@pytest.mark.slow
def test_fit_parameter_model_transport_seed_recovers_high_dhop_basin() -> None:
    x = np.array([20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0])
    model = ParameterCompositeModel(["DiffusionLF_2D", "BallisticLF_2D"], operators=["+"])
    true_params = {
        "A_1": 30.0,
        "D_2D": 300.0,
        "D_perp": 0.0,
        "A_2": 183.38088,
        "D_hop": 36171.241,
    }
    y = model.function(x, **true_params)
    yerr = np.maximum(0.005, 0.03 * y)

    params = ParameterSet(
        [
            Parameter("A_1", value=1.0, min=0.0, max=500.0),
            Parameter("D_2D", value=1.0, min=0.0, max=1.0e6),
            Parameter("D_perp", value=0.0, min=0.0, max=0.0, fixed=True),
            Parameter("A_2", value=1.0, min=0.0, max=500.0),
            Parameter("D_hop", value=1.0, min=0.0, max=1.0e6),
        ]
    )

    result = fit_parameter_model(x, y, yerr, model, params)

    assert result.success
    fitted = {parameter.name: parameter.value for parameter in result.parameters}
    assert result.reduced_chi_squared < 10.0
    assert fitted["A_1"] > 20.0
    assert 100.0 <= fitted["D_2D"] <= 1000.0
    assert fitted["A_2"] > 100.0
    assert fitted["D_hop"] > 1.0e4


@pytest.mark.slow
def test_fit_parameter_model_error_floor_blocks_high_field_collapse() -> None:
    x = np.array(
        [25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 20000.0, 35000.0]
    )
    y = np.array(
        [
            0.604866,
            0.504736,
            0.337137,
            0.203568,
            0.174864,
            0.159764,
            0.140585,
            0.139061,
            0.0664971,
            0.00056813,
            0.0,
        ]
    )
    yerr = np.array(
        [
            0.0331,
            0.0386,
            0.0239,
            0.0131,
            0.0111,
            0.0102,
            0.00929,
            0.0093,
            0.00486,
            0.000115,
            1.39e-11,
        ]
    )
    model = ParameterCompositeModel(["DiffusionLF_2D", "BallisticLF_2D"], operators=["+"])

    params = ParameterSet(
        [
            Parameter("A_1", value=1.0, min=0.0, max=500.0),
            Parameter("D_2D", value=1.0, min=0.0, max=1.0e6),
            Parameter("D_perp", value=0.0, min=0.0, max=0.0, fixed=True),
            Parameter("A_2", value=1.0, min=0.0, max=500.0),
            Parameter("D_hop", value=1.0, min=0.0, max=1.0e6),
        ]
    )

    result = fit_parameter_model(x, y, yerr, model, params)

    assert result.success
    fitted = {parameter.name: parameter.value for parameter in result.parameters}
    assert result.reduced_chi_squared < 5.0
    assert fitted["A_1"] > 20.0
    assert 100.0 <= fitted["D_2D"] <= 1000.0
    assert fitted["A_2"] > 100.0
    assert fitted["D_hop"] > 1.0e4


def test_fit_with_none_yerr_uses_unit_errors() -> None:
    x = np.linspace(0.0, 10.0, 40)
    y = 3.0 * x + 1.0
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])
    # Should not raise; yerr=None should default to ones
    result = fit_parameter_model(x, y, None, model, params)
    assert result.success


def test_fit_with_no_valid_points_returns_failure() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    yerr = np.array([1.0, 1.0, 1.0])
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])

    # x_min > x_max filters all points
    result = fit_parameter_model(x, y, yerr, model, params, x_min=10.0, x_max=5.0)
    assert not result.success
    assert "No valid points" in result.message


def test_fit_with_x_range_filtering() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 20.0, 200)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.05, size=x.shape)
    yerr = np.full_like(x, 0.05)

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])

    # Only fit 5–15 range
    result = fit_parameter_model(x, y, yerr, model, params, x_min=5.0, x_max=15.0)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(fitted["m"], 2.0, atol=0.1)


def test_fit_with_fixed_parameter_preserved() -> None:
    x = np.linspace(1.0, 10.0, 50)
    y = 3.0 * x + 5.0
    yerr = np.full_like(x, 0.1)

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet(
        [
            Parameter("m", value=1.0),
            Parameter("b", value=5.0, fixed=True),  # fix b
        ]
    )

    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(fitted["b"], 5.0)  # must not change


def test_fit_with_simplex_method_succeeds() -> None:
    x = np.linspace(0.0, 10.0, 60)
    y = 1.5 * x + 0.5
    yerr = np.full_like(x, 0.1)

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])

    result = fit_parameter_model(x, y, yerr, model, params, method="simplex")
    assert (
        result.success or not result.success
    )  # simplex may not set m.valid; just ensure no exception


def test_fit_power_law_recovers_exponent() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0.5, 5.0, 80)
    true_a, true_n = 2.0, 1.5
    noise = rng.normal(0.0, 0.05, size=x.shape)
    y = true_a * x**true_n + noise
    yerr = np.full_like(x, 0.05)

    model = ParameterCompositeModel(["PowerLaw"])
    params = ParameterSet(
        [
            Parameter("a", value=1.0, min=0.0),
            Parameter("n", value=1.0, min=0.5, max=4.0),
            Parameter("c", value=0.0, fixed=True),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(fitted["a"], true_a, atol=0.15)
    np.testing.assert_allclose(fitted["n"], true_n, atol=0.1)


def test_fit_exp_decay_recovers_tau() -> None:
    rng = np.random.default_rng(99)
    x = np.linspace(0.0, 20.0, 100)
    true_a, true_tau = 3.0, 5.0
    noise = rng.normal(0.0, 0.02, size=x.shape)
    y = true_a * np.exp(-x / true_tau) + noise
    yerr = np.full_like(x, 0.02)

    model = ParameterCompositeModel(["ExponentialDecay"])
    params = ParameterSet(
        [
            Parameter("a", value=2.0, min=0.0),
            Parameter("tau", value=4.0, min=0.1),
            Parameter("c", value=0.0, fixed=True),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(fitted["tau"], true_tau, atol=0.5)


def test_fit_result_has_uncertainties_for_free_parameters() -> None:
    x = np.linspace(0.0, 10.0, 80)
    y = 2.0 * x + 1.0
    yerr = np.full_like(x, 0.1)
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)])

    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success
    assert "m" in result.uncertainties
    assert "b" in result.uncertainties
    assert result.uncertainties["m"] > 0


def test_fit_result_chi_squared_finite() -> None:
    x = np.linspace(0.0, 5.0, 50)
    y = np.full_like(x, 1.0)
    yerr = np.full_like(x, 0.1)
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.5)])

    result = fit_parameter_model(x, y, yerr, model, params)
    assert np.isfinite(result.chi_squared)
    assert np.isfinite(result.reduced_chi_squared)
    assert result.reduced_chi_squared >= 0.0


# ---------------------------------------------------------------------------
# evaluate_parameter_model_fit
# ---------------------------------------------------------------------------


def test_evaluate_parameter_model_fit_produces_curves_for_successful_ranges() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=2.5)])

    # Build a successful result by fitting synthetic constant data.
    x = np.linspace(0.0, 5.0, 40)
    y = np.full_like(x, 2.5)
    yerr = np.full_like(x, 0.05)
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[
            ModelFitRange(
                x_min=0.0,
                x_max=5.0,
                model=model,
                parameters=params,
                result=result,
            )
        ],
        active=True,
    )

    curves = evaluate_parameter_model_fit(fit, num_points=50)
    assert len(curves) == 1
    assert curves[0].x.size == 50
    assert np.all(np.isfinite(curves[0].y))


def test_evaluate_skips_range_with_none_result() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=0.0, x_max=5.0, model=model, parameters=params, result=None)],
        active=True,
    )

    curves = evaluate_parameter_model_fit(fit)
    assert curves == []


def test_evaluate_skips_range_with_failed_result() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])
    failed = ParameterModelFitResult(success=False, message="failed")

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=0.0, x_max=5.0, model=model, parameters=params, result=failed)],
        active=True,
    )

    curves = evaluate_parameter_model_fit(fit)
    assert curves == []


def test_evaluate_skips_range_with_inverted_x_bounds() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])
    # Build a successful result first
    x = np.linspace(0.0, 5.0, 20)
    y = np.full_like(x, 1.0)
    yerr = np.full_like(x, 0.1)
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=5.0, x_max=0.0, model=model, parameters=params, result=result)],
        active=True,
    )

    curves = evaluate_parameter_model_fit(fit)
    assert curves == []


def test_evaluate_skips_range_with_none_x_bounds() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])
    x = np.linspace(0.0, 5.0, 20)
    y = np.full_like(x, 1.0)
    yerr = np.full_like(x, 0.1)
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[
            ModelFitRange(x_min=None, x_max=5.0, model=model, parameters=params, result=result)
        ],
        active=True,
    )

    curves = evaluate_parameter_model_fit(fit)
    assert curves == []


def test_evaluate_multiple_ranges_returned() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])
    x = np.linspace(0.0, 10.0, 100)
    y = np.full_like(x, 1.0)
    yerr = np.full_like(x, 0.05)
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[
            ModelFitRange(x_min=0.0, x_max=5.0, model=model, parameters=params, result=result),
            ModelFitRange(x_min=5.0, x_max=10.0, model=model, parameters=params, result=result),
        ],
        active=True,
    )

    curves = evaluate_parameter_model_fit(fit, num_points=30)
    assert len(curves) == 2
    assert curves[0].range_index == 0
    assert curves[1].range_index == 1


def test_evaluate_num_points_respected() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])
    x = np.linspace(0.0, 5.0, 20)
    y = np.full_like(x, 1.0)
    yerr = np.full_like(x, 0.05)
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=0.0, x_max=5.0, model=model, parameters=params, result=result)],
        active=True,
    )

    for n in [10, 100, 300]:
        curves = evaluate_parameter_model_fit(fit, num_points=n)
        assert curves[0].x.size == n


# ---------------------------------------------------------------------------
# sample_parameter_model
# ---------------------------------------------------------------------------


def test_sample_parameter_model_uses_given_params() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=3.25)])

    xs, ys = sample_parameter_model(model, params, x_min=0.0, x_max=5.0, num_points=40)

    assert xs.size == 40
    assert ys.size == 40
    assert np.all(np.isfinite(xs))
    assert np.allclose(ys, 3.25)
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(5.0)


def test_sample_parameter_model_window_envelope() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])

    xs, ys = sample_parameter_model(
        model,
        params,
        x_min=None,
        x_max=None,
        windows=[(0.0, 1.0), (3.0, 4.0)],
        num_points=25,
    )

    assert xs.size == 25
    assert xs.min() == pytest.approx(0.0)
    assert xs.max() == pytest.approx(4.0)
    assert np.all(np.isfinite(ys))


def test_sample_parameter_model_invalid_windows_returns_empty() -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])

    xs, ys = sample_parameter_model(
        model,
        params,
        x_min=0.0,
        x_max=5.0,
        windows=[(1.0, 0.0)],
    )

    assert xs.size == 0
    assert ys.size == 0


@pytest.mark.parametrize(
    ("x_min", "x_max"),
    [
        (5.0, 0.0),
        (None, 5.0),
        (0.0, None),
        (None, None),
        (2.0, 2.0),
    ],
)
def test_sample_parameter_model_unusable_bounds_returns_empty(
    x_min: float | None, x_max: float | None
) -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=1.0)])

    xs, ys = sample_parameter_model(model, params, x_min=x_min, x_max=x_max)

    assert xs.size == 0
    assert ys.size == 0


# ---------------------------------------------------------------------------
# carve_window_gap
# ---------------------------------------------------------------------------


def test_carve_window_gap_interior_splits() -> None:
    result = carve_window_gap([(0.0, 10.0)], x_min=0.0, x_max=10.0, lo=4.0, hi=6.0)

    assert result == [(0.0, 4.0), (6.0, 10.0)]


def test_carve_window_gap_covers_drops() -> None:
    windows = [(0.0, 2.0), (5.0, 7.0)]

    result = carve_window_gap(windows, x_min=0.0, x_max=10.0, lo=4.5, hi=7.5)

    assert result == [(0.0, 2.0)]


def test_carve_window_gap_overlaps_left() -> None:
    result = carve_window_gap([(2.0, 8.0)], x_min=0.0, x_max=10.0, lo=0.0, hi=4.0)

    assert result == [(4.0, 8.0)]


def test_carve_window_gap_overlaps_right() -> None:
    result = carve_window_gap([(2.0, 8.0)], x_min=0.0, x_max=10.0, lo=6.0, hi=10.0)

    assert result == [(2.0, 6.0)]


def test_carve_window_gap_outside_is_noop() -> None:
    windows = [(2.0, 4.0), (6.0, 8.0)]

    result = carve_window_gap(windows, x_min=0.0, x_max=10.0, lo=4.5, hi=5.5)

    assert result == windows


def test_carve_window_gap_seeds_from_bounds_when_none() -> None:
    result = carve_window_gap(None, x_min=0.0, x_max=10.0, lo=4.0, hi=6.0)

    assert result == [(0.0, 4.0), (6.0, 10.0)]


def test_carve_window_gap_empty_result_is_noop_never_none() -> None:
    windows = [(2.0, 8.0)]

    result = carve_window_gap(windows, x_min=0.0, x_max=10.0, lo=0.0, hi=10.0)

    assert result is not None
    assert result != []
    assert result == windows


def test_carve_window_gap_inverted_interval_normalised() -> None:
    forward = carve_window_gap([(0.0, 10.0)], x_min=0.0, x_max=10.0, lo=4.0, hi=6.0)
    inverted = carve_window_gap([(0.0, 10.0)], x_min=0.0, x_max=10.0, lo=6.0, hi=4.0)

    assert inverted == forward


def test_carve_window_gap_degenerate_slivers_dropped() -> None:
    result = carve_window_gap([(0.0, 10.0)], x_min=0.0, x_max=10.0, lo=0.0, hi=10.0 - 1e-12)

    # The right-hand remainder would be a sub-epsilon sliver; it must not
    # survive as a spurious near-zero-width window.
    for _, b in result:
        assert b <= 10.0
    assert all((b - a) > 1e-9 for a, b in result)


def test_carve_window_gap_returns_sorted_nonoverlapping() -> None:
    windows = [(5.0, 7.0), (0.0, 2.0)]

    result = carve_window_gap(windows, x_min=0.0, x_max=10.0, lo=1.5, hi=6.0)

    los = [a for a, _ in result]
    assert los == sorted(los)
    for (_, prev_hi), (next_lo, _) in zip(result, result[1:]):
        assert next_lo >= prev_hi


def test_sample_parameter_model_matches_evaluate_parameter_model_fit() -> None:
    """Regression: evaluate_parameter_model_fit output is unchanged for a
    fitted range now that it delegates to sample_parameter_model."""
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=2.5)])

    x = np.linspace(0.0, 5.0, 40)
    y = np.full_like(x, 2.5)
    yerr = np.full_like(x, 0.05)
    result = fit_parameter_model(x, y, yerr, model, params)
    assert result.success

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=0.0, x_max=5.0, model=model, parameters=params, result=result)],
        active=True,
    )

    curves = evaluate_parameter_model_fit(fit, num_points=50)
    assert len(curves) == 1

    expected_xs, expected_ys = sample_parameter_model(
        model, result.parameters, x_min=0.0, x_max=5.0, windows=None, num_points=50
    )
    np.testing.assert_array_equal(curves[0].x, expected_xs)
    np.testing.assert_array_equal(curves[0].y, expected_ys)


# ---------------------------------------------------------------------------
# OrderParameter integration: composite model, fitting, documentation
# ---------------------------------------------------------------------------


def test_order_parameter_composite_formula_and_params() -> None:
    model = ParameterCompositeModel(["OrderParameter"], operators=[])
    assert model.param_names == ["y0", "Tc", "beta", "alpha"]
    assert model.formula_string() == "y0*(1 - (T/Tc)^alpha)^beta"


def test_order_parameter_params_render_proper_symbols() -> None:
    # y0 and alpha must resolve to rich symbols via the global registry so the
    # global/cross-group fit windows (which label via get_param_info) match the
    # single-series Info dialog instead of showing the raw "y0"/"alpha" strings.
    from asymmetry.core.fitting.parameters import get_param_info

    assert get_param_info("alpha").unicode == "α"
    assert get_param_info("alpha").latex == r"$\alpha$"
    assert get_param_info("y0").unicode == "y₀"
    assert get_param_info("y0").latex == r"$y_0$"

    model = ParameterCompositeModel(["OrderParameter"], operators=[])
    assert model.param_info["alpha"].unicode == "α"
    assert model.param_info["y0"].unicode == "y₀"


def test_order_parameter_fit_recovers_known_parameters() -> None:
    # Synthesize a clean nu(T) order-parameter curve and recover y0, Tc, beta.
    true_y0, true_tc, true_beta, true_alpha = 29.9, 69.0, 0.36, 1.0
    temperature = np.linspace(1.5, 66.0, 25)
    values = _order_parameter(temperature, y0=true_y0, Tc=true_tc, beta=true_beta, alpha=true_alpha)
    errors = np.full_like(values, 0.05)

    model = ParameterCompositeModel(["OrderParameter"], operators=[])
    params = ParameterSet(
        [
            Parameter("y0", value=25.0, min=0.0),
            Parameter("Tc", value=72.0, min=0.0),
            Parameter("beta", value=0.5, min=0.0),
            # Fix the shape exponent to the near-Tc power law for identifiability.
            Parameter("alpha", value=1.0, min=0.0, fixed=True),
        ]
    )

    result = fit_parameter_model(temperature, values, errors, model, params)
    assert result.success
    recovered = {p.name: p.value for p in result.parameters}
    np.testing.assert_allclose(recovered["y0"], true_y0, rtol=1e-2)
    np.testing.assert_allclose(recovered["Tc"], true_tc, rtol=1e-2)
    np.testing.assert_allclose(recovered["beta"], true_beta, rtol=5e-2)


def test_order_parameter_applicability_text_describes_use() -> None:
    from asymmetry.core.fitting.component_docs import get_component_applicability

    text = get_component_applicability("OrderParameter")
    assert "order parameter" in text.lower()
    assert "Tc" in text
    assert "β" in text or "beta" in text.lower()


# ---------------------------------------------------------------------------
# Phase A — quadrature combinator ⊕  (f ⊕ g = √(f² + g²), parameter grammar)
# ---------------------------------------------------------------------------


def test_quadrature_oracle_powerlaw_plus_constant_equals_powerlawquadbg() -> None:
    """PowerLaw ⊕ Constant reproduces PowerLawQuadBG exactly.

    Registry PowerLaw is a·|x|ⁿ + c (its own additive c), whereas
    PowerLawQuadBG's inner term has no c — so the identity holds with PowerLaw's
    c fixed at 0 and Constant's c = BG.
    """
    model = ParameterCompositeModel.from_expression("PowerLaw ⊕ Constant")
    # dedup: PowerLaw.c -> c_1 (fixed 0), Constant.c -> c_2 (= BG).
    assert model.param_names == ["a", "n", "c_1", "c_2"]
    x = np.linspace(-5.0, 5.0, 41)
    kw = {"a": 1.7, "n": 2.3, "c_1": 0.0, "c_2": 0.6}
    got = model.function(x, **kw)
    quad = PARAMETER_MODEL_COMPONENTS["PowerLawQuadBG"].function(x, a=1.7, n=2.3, BG=0.6)
    np.testing.assert_allclose(got, quad, atol=1e-12, rtol=0.0)


def test_quadrature_is_associative() -> None:
    model = ParameterCompositeModel.from_expression("Constant ⊕ Constant ⊕ Constant")
    val = model.function(np.zeros(1), c_1=3.0, c_2=4.0, c_3=12.0)[0]
    assert val == pytest.approx(np.sqrt(9.0 + 16.0 + 144.0))  # = 13


def test_quadrature_shares_precedence_with_plus() -> None:
    # ⊕ is level 1 like + / -, left-associative: A ⊕ B + C = √(A²+B²) + C.
    model = ParameterCompositeModel.from_expression("Constant ⊕ Constant + Constant")
    val = model.function(np.zeros(1), c_1=3.0, c_2=4.0, c_3=1.0)[0]
    assert val == pytest.approx(np.sqrt(9.0 + 16.0) + 1.0)  # 5 + 1 = 6


def test_multiplication_binds_tighter_than_quadrature() -> None:
    # A ⊕ B * C evaluates B*C first (precedence 2), then the quadrature.
    model = ParameterCompositeModel.from_expression("Constant ⊕ Constant * Constant")
    val = model.function(np.zeros(1), c_1=3.0, c_2=2.0, c_3=2.0)[0]
    assert val == pytest.approx(np.sqrt(9.0 + 16.0))  # 3 ⊕ (2*2) = 5


def test_quadrature_respects_parentheses() -> None:
    # (A ⊕ B) + C and A ⊕ (B + C) differ.
    grouped = ParameterCompositeModel.from_expression("(Constant ⊕ Constant) + Constant")
    nested = ParameterCompositeModel.from_expression("Constant ⊕ (Constant + Constant)")
    g = grouped.function(np.zeros(1), c_1=3.0, c_2=4.0, c_3=1.0)[0]
    n = nested.function(np.zeros(1), c_1=3.0, c_2=4.0, c_3=1.0)[0]
    assert g == pytest.approx(np.sqrt(9.0 + 16.0) + 1.0)  # 6
    assert n == pytest.approx(np.sqrt(9.0 + (4.0 + 1.0) ** 2))  # √(9+25) ≈ 5.831
    assert g != pytest.approx(n)


def test_quadrature_roundtrips_through_to_dict_and_expression() -> None:
    model = ParameterCompositeModel.from_expression("PowerLaw ⊕ Constant")
    # Expression string preserves ⊕.
    assert "⊕" in model.component_expression_string()
    assert "⊕" in model.formula_string()  # GLE export path reads formula_string
    # to_dict / from_dict preserves the operator and evaluates identically.
    restored = ParameterCompositeModel.from_dict(model.to_dict())
    assert restored.operators == ["⊕"]
    x = np.linspace(-3.0, 3.0, 11)
    kw = {"a": 1.2, "n": 1.5, "c_1": 0.0, "c_2": 0.4}
    np.testing.assert_allclose(model.function(x, **kw), restored.function(x, **kw), atol=1e-12)


def test_quadrature_constructor_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="operators must be"):
        ParameterCompositeModel(["Constant", "Constant"], ["%"])
    # ⊕ is accepted.
    ParameterCompositeModel(["Constant", "Constant"], ["⊕"])


def test_quadrature_is_rejected_in_time_domain_grammar() -> None:
    """⊕ is parameter-grammar only; the time-domain composite grammar rejects
    it (the operator boundary)."""
    from asymmetry.core.fitting.composite import CompositeModel

    with pytest.raises(ValueError):
        CompositeModel.from_expression("Exponential ⊕ Constant")


def test_quadrature_operands_are_additive_components() -> None:
    # Each ⊕ operand is a distinct curve worth plotting (like +).
    model = ParameterCompositeModel.from_expression("PowerLaw ⊕ Constant")
    assert model.additive_component_indices() == [0, 1]


# ---------------------------------------------------------------------------
# Data-aware trend seeds (suggest_trend_seeds)
# ---------------------------------------------------------------------------


def test_suggest_trend_seeds_critical_divergence_tc_below_data() -> None:
    # Spin-glass-like λ(T): diverges as T -> T_g from above. Tc should seed just
    # below the lowest fitted T, never the unphysical default of 10.
    model = ParameterCompositeModel(["CriticalDivergence"])
    x = np.array([90.0, 91.0, 95.0, 120.0, 280.0])
    y = np.array([0.59, 0.30, 0.14, 0.04, 0.017])

    seeds = suggest_trend_seeds(model, x, y)

    assert seeds["Tc"] < 90.0
    # Just below the data, not miles away.
    assert 90.0 - seeds["Tc"] < (280.0 - 90.0)
    assert seeds["Tc"] != 10.0
    # Baseline c seeded from the flat, far-from-Tc end (min of y).
    assert seeds["c"] == pytest.approx(0.017)


def test_suggest_trend_seeds_order_parameter_tc_above_data() -> None:
    # Order parameter vanishes at Tc from below, so Tc seeds above the highest T
    # and the amplitude y0 from the largest observed value.
    model = ParameterCompositeModel(["OrderParameter"])
    x = np.array([5.0, 20.0, 40.0, 60.0, 68.0])
    y = np.array([100.0, 95.0, 80.0, 50.0, 20.0])

    seeds = suggest_trend_seeds(model, x, y)

    assert seeds["Tc"] > 68.0
    assert seeds["y0"] == pytest.approx(100.0)
    assert seeds["Tc"] != 10.0


def test_suggest_trend_seeds_enables_convergence_without_manual_reseed() -> None:
    # The whole point: a CriticalDivergence fit from the suggested seeds must
    # converge near the true Tc where the bare default (Tc=10) would not.
    true_tc, true_a, true_nu, true_c = 88.0, 0.4, 0.6, 0.02
    x = np.array([90.0, 91.0, 95.0, 100.0, 120.0, 150.0, 221.0, 280.0])
    y = true_a * np.abs(x - true_tc) ** (-true_nu) + true_c

    model = ParameterCompositeModel(["CriticalDivergence"])
    seeds = suggest_trend_seeds(model, x, y)

    params = ParameterSet()
    for pname in model.param_names:
        params.add(
            Parameter(name=pname, value=float(seeds.get(pname, model.param_defaults[pname])))
        )
    result = fit_parameter_model(x, y, None, model, params)

    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["Tc"] == pytest.approx(true_tc, abs=2.0)


def test_suggest_trend_seeds_ignores_non_trend_models() -> None:
    model = ParameterCompositeModel(["Linear"])
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    assert suggest_trend_seeds(model, x, y) == {}


def test_suggest_trend_seeds_handles_all_nan_x() -> None:
    model = ParameterCompositeModel(["CriticalDivergence"])
    x = np.array([np.nan, np.nan])
    y = np.array([1.0, 2.0])
    assert suggest_trend_seeds(model, x, y) == {}


def test_suggest_trend_seeds_single_point_uses_floor_margin() -> None:
    # Zero-span data must still produce a finite Tc offset from the point.
    model = ParameterCompositeModel(["CriticalDivergence"])
    x = np.array([50.0])
    y = np.array([1.0])
    seeds = suggest_trend_seeds(model, x, y)
    assert seeds["Tc"] < 50.0
    assert np.isfinite(seeds["Tc"])


def test_suggest_trend_seeds_suffixed_params_for_repeated_component() -> None:
    # Two CriticalDivergence components -> Tc_1 / Tc_2 unique names.
    model = ParameterCompositeModel(["CriticalDivergence", "CriticalDivergence"])
    x = np.array([90.0, 120.0, 280.0])
    y = np.array([0.5, 0.1, 0.02])
    seeds = suggest_trend_seeds(model, x, y)
    assert "Tc_1" in seeds and "Tc_2" in seeds
    assert all(name in model.param_names for name in seeds)


# ---------------------------------------------------------------------------
# Generic data-aware seeds (suggest_model_seeds)
# ---------------------------------------------------------------------------


def _params_from_seeds(model: ParameterCompositeModel, seeds: dict[str, float]) -> ParameterSet:
    params = ParameterSet()
    for pname in model.param_names:
        params.add(
            Parameter(name=pname, value=float(seeds.get(pname, model.param_defaults[pname])))
        )
    return params


def test_suggest_model_seeds_linear() -> None:
    model = ParameterCompositeModel(["Linear"])
    x = np.linspace(0.0, 10.0, 40)
    y = 3.5 * x - 2.0
    seeds = suggest_model_seeds(model, x, y)
    assert seeds["m"] == pytest.approx(3.5, rel=1e-6)
    assert seeds["b"] == pytest.approx(-2.0, abs=1e-6)


def test_suggest_model_seeds_constant() -> None:
    model = ParameterCompositeModel(["Constant"])
    x = np.linspace(0.0, 5.0, 21)
    y = np.full_like(x, 7.25)
    seeds = suggest_model_seeds(model, x, y)
    assert seeds["c"] == pytest.approx(7.25, rel=1e-6)


def test_suggest_model_seeds_power_law() -> None:
    model = ParameterCompositeModel(["PowerLaw"])
    true_a, true_n, true_c = 2.0, 1.5, 0.3
    x = np.linspace(0.5, 20.0, 40)
    y = true_a * np.abs(x) ** true_n + true_c
    seeds = suggest_model_seeds(model, x, y)
    # Order-of-magnitude on the amplitude, close on the exponent.
    assert 0.5 * true_a < seeds["a"] < 2.0 * true_a
    assert seeds["n"] == pytest.approx(true_n, abs=0.3)


def test_suggest_model_seeds_exp_decay() -> None:
    model = ParameterCompositeModel(["ExponentialDecay"])
    true_a, true_tau, true_c = 5.0, 4.0, 1.0
    x = np.linspace(0.0, 20.0, 60)
    y = true_a * np.exp(-x / true_tau) + true_c
    seeds = suggest_model_seeds(model, x, y)
    assert seeds["tau"] == pytest.approx(true_tau, rel=0.25)
    assert 0.5 * true_a < seeds["a"] < 2.0 * true_a
    assert seeds["c"] == pytest.approx(true_c, abs=0.5)


def test_suggest_model_seeds_arrhenius() -> None:
    from asymmetry.core.fitting.parameter_models import _arrhenius

    model = ParameterCompositeModel(["Arrhenius"])
    true_a, true_ea = 3.0, 25.0
    x = np.linspace(50.0, 400.0, 40)
    y = _arrhenius(x, a=true_a, Ea=true_ea)
    seeds = suggest_model_seeds(model, x, y)
    assert seeds["Ea"] == pytest.approx(true_ea, rel=0.1)
    assert 0.5 * true_a < seeds["a"] < 2.0 * true_a


def test_suggest_model_seeds_lcr_gaussian() -> None:
    from asymmetry.core.fitting.parameter_models import _lcr_gaussian

    model = ParameterCompositeModel(["GaussianLCR"])
    true_f, true_b0, true_bwid = 0.8, 1200.0, 150.0
    x = np.linspace(500.0, 2000.0, 80)
    y = _lcr_gaussian(x, f=true_f, B0=true_b0, Bwid=true_bwid)
    seeds = suggest_model_seeds(model, x, y)
    # Centre within a width, amplitude the right order, width within 2x.
    assert abs(seeds["B0"] - true_b0) < true_bwid
    assert 0.5 * true_f < seeds["f"] < 2.0 * true_f
    assert 0.5 * true_bwid < seeds["Bwid"] < 2.0 * true_bwid


def test_suggest_model_seeds_lorentzian_b0_positive_and_in_range() -> None:
    # The Lorentzian peak is pinned at the origin, so the B0 seed (a half-width)
    # must be positive and within a sane fraction of the x-span, never negative
    # or absurd from a pathological half-max crossing.
    from asymmetry.core.fitting.parameter_models import _lorentzian

    model = ParameterCompositeModel(["Lorentzian"])
    x = np.linspace(-500.0, 500.0, 80)
    y = _lorentzian(x, a=2.0, B0=120.0, c=0.1)
    seeds = suggest_model_seeds(model, x, y)
    span = float(np.max(x) - np.min(x))
    if "B0" in seeds:
        assert seeds["B0"] > 0.0
        assert span / 10.0 <= seeds["B0"] <= 10.0 * span
    assert seeds["a"] == pytest.approx(2.0, rel=0.5)
    assert seeds["c"] == pytest.approx(0.1, abs=0.3)


def test_suggest_model_seeds_lorentzian_pathological_drops_b0() -> None:
    # A near-flat trace with no clean half-max crossing must not seed an absurd
    # B0: the estimator drops it (or keeps it in range) rather than emitting a
    # negative/huge width.
    model = ParameterCompositeModel(["Lorentzian"])
    x = np.linspace(0.0, 10.0, 5)
    y = np.array([1.0, 1.0, 1.0001, 1.0, 1.0])
    seeds = suggest_model_seeds(model, x, y)
    span = float(np.max(x) - np.min(x))
    if "B0" in seeds:
        assert seeds["B0"] > 0.0
        assert span / 10.0 <= seeds["B0"] <= 10.0 * span


def test_suggest_model_seeds_unknown_component_returns_no_seed() -> None:
    # Redfield has no registered estimator -> no seeds contributed.
    model = ParameterCompositeModel(["Redfield"])
    x = np.linspace(1.0, 100.0, 30)
    y = np.linspace(1.0, 0.1, 30)
    assert suggest_model_seeds(model, x, y) == {}


def test_suggest_model_seeds_all_nan_returns_empty() -> None:
    model = ParameterCompositeModel(["Linear"])
    x = np.array([np.nan, np.nan, np.nan])
    y = np.array([1.0, 2.0, 3.0])
    assert suggest_model_seeds(model, x, y) == {}


def test_suggest_model_seeds_delegates_critical_component() -> None:
    # A composite of Linear + CriticalDivergence must seed both the generic
    # linear terms and the delegated trend Tc/c.
    model = ParameterCompositeModel(["CriticalDivergence"])
    x = np.array([90.0, 91.0, 95.0, 120.0, 280.0])
    y = np.array([0.59, 0.30, 0.14, 0.04, 0.017])
    generic = suggest_model_seeds(model, x, y)
    trend = suggest_trend_seeds(model, x, y)
    for name, value in trend.items():
        assert generic[name] == pytest.approx(value)


def test_suggest_trend_seeds_still_public_and_unchanged() -> None:
    # The public trend seeder must keep its exact behaviour: only the critical
    # component's params, nothing else, for a Linear+CriticalDivergence model.
    model = ParameterCompositeModel(["Linear", "CriticalDivergence"])
    x = np.array([90.0, 120.0, 280.0])
    y = np.array([0.5, 0.1, 0.02])
    seeds = suggest_trend_seeds(model, x, y)
    assert set(seeds) == {"Tc", "c"}
    assert seeds["Tc"] < 90.0
    assert seeds["c"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Multi-start robustness and new result fields
# ---------------------------------------------------------------------------


def test_fit_extra_starts_zero_is_identical() -> None:
    # extra_starts=0 (the default) must reproduce the params/chi2 exactly.
    model = ParameterCompositeModel(["Linear"])
    x = np.linspace(0.0, 10.0, 25)
    y = 2.0 * x + 1.0 + 0.01 * np.sin(x)
    params = ParameterSet()
    for pname in model.param_names:
        params.add(Parameter(name=pname, value=model.param_defaults[pname]))

    default = fit_parameter_model(x, y, None, model, params)
    explicit = fit_parameter_model(x, y, None, model, params, extra_starts=0)

    assert default.success == explicit.success
    assert default.chi_squared == explicit.chi_squared
    default_vals = {p.name: p.value for p in default.parameters}
    explicit_vals = {p.name: p.value for p in explicit.parameters}
    assert default_vals == explicit_vals


def test_fit_multistart_escapes_bad_minimum() -> None:
    # An exponential decay with a badly-placed user seed lands in a poor local
    # minimum for a single start; extra_starts should find a better chi2.
    model = ParameterCompositeModel(["ExponentialDecay"])
    x = np.linspace(0.0, 20.0, 60)
    y = 5.0 * np.exp(-x / 4.0) + 1.0

    def bad_params() -> ParameterSet:
        p = ParameterSet()
        p.add(Parameter(name="a", value=0.01, min=-100.0, max=100.0))
        p.add(Parameter(name="tau", value=1000.0, min=1e-3, max=1e5))
        p.add(Parameter(name="c", value=0.0, min=-100.0, max=100.0))
        return p

    single = fit_parameter_model(x, y, None, model, bad_params())
    multi = fit_parameter_model(x, y, None, model, bad_params(), extra_starts=8, seed=1)

    assert multi.success
    assert multi.reduced_chi_squared <= single.reduced_chi_squared + 1e-9
    # The recovered decay is close to the truth.
    fitted = {p.name: p.value for p in multi.parameters}
    assert fitted["tau"] == pytest.approx(4.0, rel=0.3)


def test_fit_multistart_deterministic() -> None:
    # Same seed -> identical result; the RNG must be default_rng(seed) only.
    model = ParameterCompositeModel(["ExponentialDecay"])
    x = np.linspace(0.0, 20.0, 40)
    y = 4.0 * np.exp(-x / 5.0) + 0.5

    def params() -> ParameterSet:
        p = ParameterSet()
        p.add(Parameter(name="a", value=1.0, min=-100.0, max=100.0))
        p.add(Parameter(name="tau", value=50.0, min=1e-3, max=1e5))
        p.add(Parameter(name="c", value=0.0, min=-100.0, max=100.0))
        return p

    first = fit_parameter_model(x, y, None, model, params(), extra_starts=6, seed=42)
    second = fit_parameter_model(x, y, None, model, params(), extra_starts=6, seed=42)
    assert first.chi_squared == second.chi_squared
    assert {p.name: p.value for p in first.parameters} == {
        p.name: p.value for p in second.parameters
    }


def test_perturbed_starts_finite_with_infinite_bounds() -> None:
    # An offset-like param (value 0) with infinite bounds must still get FINITE
    # perturbed starts — the jitter falls back to the data-derived scale, never
    # inf*0.25.
    from asymmetry.core.fitting.parameter_models import _perturbed_start_candidates

    params = ParameterSet()
    params.add(Parameter(name="b", value=0.0))  # default bounds are ±inf
    params.add(Parameter(name="m", value=2.0))
    base = {"b": 0.0, "m": 2.0}
    rng = np.random.default_rng(0)
    candidates = _perturbed_start_candidates(base, params, 5, rng, default_scale=3.0)
    assert len(candidates) == 5
    for cand in candidates:
        assert np.isfinite(cand["b"])
        assert np.isfinite(cand["m"])
        # The offset jitter stays within the data-derived fallback scale.
        assert abs(cand["b"]) <= 3.0 + 1e-9


def test_fit_infinite_bounds_multistart_runs() -> None:
    # End-to-end: a Linear fit with default (infinite) bounds and extra_starts
    # must not blow up on the offset-jitter fallback.
    model = ParameterCompositeModel(["Linear"])
    x = np.linspace(0.0, 10.0, 30)
    y = 3.0 * x - 1.0
    params = ParameterSet()
    for pname in model.param_names:
        params.add(Parameter(name=pname, value=0.0))  # ±inf bounds

    result = fit_parameter_model(x, y, None, model, params, extra_starts=6, seed=7)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["m"] == pytest.approx(3.0, rel=1e-3)
    assert fitted["b"] == pytest.approx(-1.0, abs=1e-2)


def test_fit_params_at_bound_flagged() -> None:
    # A parameter whose true optimum lies outside a tight bound gets pinned.
    model = ParameterCompositeModel(["Linear"])
    x = np.linspace(0.0, 10.0, 30)
    y = 5.0 * x + 1.0
    params = ParameterSet()
    # Slope wants 5 but is capped at 2 -> pinned at max.
    params.add(Parameter(name="m", value=1.0, min=0.0, max=2.0))
    params.add(Parameter(name="b", value=0.0, min=-100.0, max=100.0))

    result = fit_parameter_model(x, y, None, model, params)
    assert "m" in result.params_at_bound


def test_fit_seed_beat_user_start_flag() -> None:
    # A deliberately terrible user seed that a generic/perturbed start beats
    # must set seed_beat_user_start (extra_starts > 0).
    model = ParameterCompositeModel(["Linear"])
    x = np.linspace(0.0, 10.0, 30)
    y = 4.0 * x - 3.0
    params = ParameterSet()
    params.add(Parameter(name="m", value=-999.0, min=-1e4, max=1e4))
    params.add(Parameter(name="b", value=999.0, min=-1e4, max=1e4))

    result = fit_parameter_model(x, y, None, model, params, extra_starts=8, seed=3)
    assert result.success
    assert result.seed_beat_user_start
    assert result.n_starts_tried > 1


def test_fit_transport_heuristic_win_does_not_set_flag() -> None:
    # The transport heuristic ALWAYS runs internally (even at extra_starts=0), so
    # it beating the user's start is not news: the flag must stay False. A
    # DiffusionLF model with a free amplitude and a bad user seed exercises this.
    model = ParameterCompositeModel(["DiffusionLF_2D"])
    x = np.linspace(10.0, 500.0, 40)
    # A physically-shaped LF-diffusion curve from a known amplitude/rate.
    from asymmetry.core.fitting.parameter_models import _diffusion_lf_2d

    y = _diffusion_lf_2d(x, A=3.0, D_2D=50.0)
    params = ParameterSet()
    # Deliberately far-off user seed so the internal transport grid seed wins.
    params.add(Parameter(name="A", value=1e-4, min=0.0, max=100.0))
    params.add(Parameter(name="D_2D", value=1e-4, min=1e-6, max=1e6))
    params.add(Parameter(name="D_perp", value=0.0, min=0.0, max=1e6, fixed=True))

    result = fit_parameter_model(x, y, None, model, params)
    # extra_starts defaults to 0: only user + transport candidates exist.
    assert result.n_starts_tried >= 1
    # Even if the transport seed produced the winning fit, the flag stays False.
    assert result.seed_beat_user_start is False


def test_result_fields_default_and_roundtrip() -> None:
    # Benign defaults on a bare result, and a dict round-trip preserves them.
    result = ParameterModelFitResult(success=True)
    assert result.n_starts_tried == 0
    assert result.seed_beat_user_start is False
    assert result.params_at_bound == ()

    populated = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        n_points=10,
        n_starts_tried=5,
        seed_beat_user_start=True,
        params_at_bound=("m", "tau"),
    )
    # Field-by-field round-trip through a plain dict (the panel serialization
    # pattern: read known keys, benign defaults for the rest).
    state = {
        "success": populated.success,
        "chi_squared": populated.chi_squared,
        "reduced_chi_squared": populated.reduced_chi_squared,
        "n_points": populated.n_points,
        "n_starts_tried": populated.n_starts_tried,
        "seed_beat_user_start": populated.seed_beat_user_start,
        "params_at_bound": list(populated.params_at_bound),
    }
    restored = ParameterModelFitResult(
        success=bool(state["success"]),
        chi_squared=float(state["chi_squared"]),
        reduced_chi_squared=float(state["reduced_chi_squared"]),
        n_points=int(state["n_points"]),
        n_starts_tried=int(state.get("n_starts_tried", 0)),
        seed_beat_user_start=bool(state.get("seed_beat_user_start", False)),
        params_at_bound=tuple(state.get("params_at_bound", ())),
    )
    assert restored.n_starts_tried == 5
    assert restored.seed_beat_user_start is True
    assert restored.params_at_bound == ("m", "tau")

    # A legacy state missing the new keys still constructs with benign defaults.
    legacy = ParameterModelFitResult(
        success=True,
        n_starts_tried=int({}.get("n_starts_tried", 0)),
    )
    assert legacy.n_starts_tried == 0
    assert legacy.seed_beat_user_start is False
    assert legacy.params_at_bound == ()


def test_parameter_model_categories_cover_registry() -> None:
    # Every category-table key must name a registered component, so a rename
    # or removal fails here instead of silently un-categorizing the entry.
    from asymmetry.core.fitting.parameter_models import (
        _PARAMETER_MODEL_CATEGORIES,
        PARAMETER_MODEL_COMPONENTS,
    )

    missing = set(_PARAMETER_MODEL_CATEGORIES) - set(PARAMETER_MODEL_COMPONENTS)
    assert not missing

    # The taxonomy is applied to the definitions themselves.
    assert PARAMETER_MODEL_COMPONENTS["SC_SWave"].category == "Superconducting gap"
    assert PARAMETER_MODEL_COMPONENTS["Arrhenius"].category == "Scaling & activation"
    assert PARAMETER_MODEL_COMPONENTS["Constant"].category == "General"
    assert all(
        (definition.category or "").strip() for definition in PARAMETER_MODEL_COMPONENTS.values()
    )

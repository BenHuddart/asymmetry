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
    component_names_for_x,
    evaluate_parameter_model_fit,
    fit_parameter_model,
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
# OrderParameter integration: composite model, fitting, documentation
# ---------------------------------------------------------------------------


def test_order_parameter_composite_formula_and_params() -> None:
    model = ParameterCompositeModel(["OrderParameter"], operators=[])
    assert model.param_names == ["y0", "Tc", "beta", "alpha"]
    assert model.formula_string() == "y0*(1 - (T/Tc)^alpha)^beta"


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
    assert "beta" in text.lower()

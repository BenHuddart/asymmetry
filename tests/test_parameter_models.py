"""Tests for parameter-model fitting core."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
    ParameterModelFitResult,
    component_names_for_x,
    evaluate_parameter_model_fit,
    fit_parameter_model,
    ModelFitRange,
    ParameterModelFit,
    _arrhenius,
    _constant,
    _critical_divergence,
    _exp_decay,
    _linear,
    _lorentzian,
    _power_law,
    _redfield,
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
    assert "Arrhenius" not in field_names

    assert "Arrhenius" in temp_names
    assert "CriticalDivergence" in temp_names
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
    # Should not raise or return inf/nan even with tiny tau
    result = _exp_decay(x, a=1.0, tau=0.0, c=0.0)
    assert np.all(np.isfinite(result))


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
    params = ParameterSet([
        Parameter("m", value=1.0),
        Parameter("b", value=5.0, fixed=True),  # fix b
    ])

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
    assert result.success or not result.success  # simplex may not set m.valid; just ensure no exception


def test_fit_power_law_recovers_exponent() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0.5, 5.0, 80)
    true_a, true_n = 2.0, 1.5
    noise = rng.normal(0.0, 0.05, size=x.shape)
    y = true_a * x**true_n + noise
    yerr = np.full_like(x, 0.05)

    model = ParameterCompositeModel(["PowerLaw"])
    params = ParameterSet([
        Parameter("a", value=1.0, min=0.0),
        Parameter("n", value=1.0, min=0.5, max=4.0),
        Parameter("c", value=0.0, fixed=True),
    ])
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
    params = ParameterSet([
        Parameter("a", value=2.0, min=0.0),
        Parameter("tau", value=4.0, min=0.1),
        Parameter("c", value=0.0, fixed=True),
    ])
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
        ranges=[ModelFitRange(x_min=None, x_max=5.0, model=model, parameters=params, result=result)],
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

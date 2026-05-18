"""Tests for composite fit-function model construction and evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


def test_single_component_matches_baseline_free_model() -> None:
    t = np.linspace(0.0, 5.0, 100)
    model = CompositeModel(["Exponential"])

    out = model.function(t, A_1=2.0, Lambda=0.3)
    expected = COMPONENTS["Exponential"].function(t, A=2.0, Lambda=0.3)

    assert np.allclose(out, expected)


def test_addition_and_multiplication_work() -> None:
    t = np.linspace(0.0, 2.0, 20)

    add_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    add_out = add_model.function(t, A_1=1.0, Lambda=0.5, A_bg=0.2)
    assert np.allclose(add_out, COMPONENTS["Exponential"].function(t, A=1.0, Lambda=0.5) + 0.2)

    mul_model = CompositeModel(["Exponential", "Gaussian"], operators=["*"])
    mul_out = mul_model.function(t, A_1=0.8, Lambda=0.5, sigma=0.25)
    exp_vals = COMPONENTS["Exponential"].function(t, A=1.0, Lambda=0.5)
    gauss_vals = COMPONENTS["Gaussian"].function(t, A=1.0, sigma=0.25)
    expected = 0.8 * exp_vals * gauss_vals
    assert np.allclose(mul_out, expected)


def test_operator_precedence_is_respected() -> None:
    t = np.linspace(0.0, 1.0, 10)
    model = CompositeModel(["Constant", "Constant", "Constant"], operators=["+", "*"])
    out = model.function(t, A_bg_1=1.0, A_bg_2=2.0, A_bg_3=3.0)

    # 1 + (2 * 3)
    assert np.allclose(out, np.full_like(t, 7.0))


def test_duplicate_components_have_unique_parameter_names() -> None:
    model = CompositeModel(["Exponential", "Exponential"], operators=["+"])

    assert "A_1" in model.param_names
    assert "Lambda_1" in model.param_names
    assert "A_2" in model.param_names
    assert "Lambda_2" in model.param_names


def test_unique_symbols_are_not_indexed_except_amplitude() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    assert "A_1" in model.param_names
    assert "Lambda" in model.param_names
    assert "A_bg" in model.param_names


def test_multiplicative_chain_uses_single_amplitude_parameter() -> None:
    model = CompositeModel(["Exponential", "Gaussian", "Oscillatory"], operators=["*", "*"])

    assert "A_1" in model.param_names
    assert "A_2" not in model.param_names
    assert "A_3" not in model.param_names


def test_formula_and_serialization_round_trip() -> None:
    model = CompositeModel(["Oscillatory", "Constant"], operators=["-"])
    formula = model.formula_string()

    assert "frequency" in formula
    assert "A_bg" in formula

    payload = model.to_dict()
    restored = CompositeModel.from_dict(payload)

    assert restored.component_names == model.component_names
    assert restored.operators == model.operators


def test_component_expression_round_trip() -> None:
    model = CompositeModel.from_expression("Gaussian * ( Constant + Constant )")

    assert model.component_names == ["Gaussian", "Constant", "Constant"]
    assert model.operators == ["*", "+"]
    assert model.open_parentheses == [0, 1, 0]
    assert model.close_parentheses == [0, 0, 1]
    assert model.component_expression_string() == "Gaussian * (Constant + Constant)"


def test_fraction_group_expression_round_trip() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")

    assert model.component_names == ["Exponential", "Gaussian", "Constant"]
    assert model.operators == ["+", "+"]
    assert model.open_parentheses == [1, 0, 0]
    assert model.close_parentheses == [0, 0, 1]
    assert model.fraction_groups == [(0, 2)]
    assert model.component_expression_string() == "(Exponential + Gaussian + Constant){frac}"


def test_fraction_group_rejects_non_additive_group() -> None:
    with pytest.raises(ValueError, match="additive"):
        CompositeModel.from_expression("( Exponential * Gaussian ){frac}")


def test_fraction_group_accepts_sum_of_products() -> None:
    t = np.linspace(0.0, 1.0, 5)
    model = CompositeModel.from_expression(
        "( Exponential * Gaussian + Exponential * Gaussian ){frac}"
    )

    assert model.fraction_parameter_groups() == [["fraction_1", "fraction_2"]]
    assert model.param_names == [
        "A_1",
        "Lambda_1",
        "fraction_1",
        "sigma_1",
        "Lambda_2",
        "fraction_2",
        "sigma_2",
    ]
    assert "Lambda_1" in model.formula_string()
    assert "sigma_1" in model.formula_string()
    assert "Lambda_2" in model.formula_string()
    assert "sigma_2" in model.formula_string()

    out = model.function(
        t,
        A_1=2.0,
        Lambda_1=0.3,
        sigma_1=0.2,
        Lambda_2=0.6,
        sigma_2=0.4,
        fraction_1=0.25,
        fraction_2=0.75,
    )
    expected = 2.0 * (
        0.25 * np.exp(-0.3 * t) * np.exp(-(0.2 * t) ** 2)
        + 0.75 * np.exp(-0.6 * t) * np.exp(-(0.4 * t) ** 2)
    )

    assert np.allclose(out, expected)


def test_with_default_fraction_groups_wraps_top_level_additive_expression() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])

    grouped_model = model.with_default_fraction_groups()

    assert grouped_model is not model
    assert grouped_model.fraction_groups == [(0, 1)]
    assert grouped_model.component_expression_string() == "(Exponential + Constant){frac}"
    assert grouped_model.param_names == ["A_1", "Lambda", "fraction_1", "fraction_2"]


def test_to_model_definition_callable() -> None:
    t = np.linspace(0.0, 1.0, 8)
    model = CompositeModel(["Constant"])
    definition = model.to_model_definition()

    out = definition.function(t, A_bg=0.4)
    assert np.allclose(out, np.full_like(t, 0.4))


def test_additive_component_indices_excludes_non_additive_terms() -> None:
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian", "Constant"],
        operators=["+", "*", "+"],
    )
    assert model.additive_component_indices() == [0, 1, 3]


def test_evaluate_components_returns_named_component_curves() -> None:
    t = np.linspace(0.0, 2.0, 25)
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian"],
        operators=["+", "*"],
    )

    curves = model.evaluate_components(
        t,
        A_1=1.2,
        Lambda=0.3,
        A_bg=0.1,
        A_2=0.8,
        sigma=0.4,
    )

    assert [name for name, _vals in curves] == ["Exponential", "Constant", "Gaussian"]
    assert all(vals.shape == t.shape for _name, vals in curves)


def test_evaluate_components_additive_only_filters_multiplicative_terms() -> None:
    t = np.linspace(0.0, 2.0, 25)
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian", "Constant"],
        operators=["+", "*", "+"],
    )

    curves = model.evaluate_components(
        t,
        additive_only=True,
        A_1=1.0,
        Lambda=0.2,
        A_bg_2=0.15,
        A_3=0.9,
        sigma=0.5,
        A_bg_4=0.03,
    )

    assert [name for name, _vals in curves] == ["Exponential", "Constant", "Constant"]


def test_formula_string_shows_single_amplitude_for_multiplicative_chain() -> None:
    model = CompositeModel(["Exponential", "Gaussian"], operators=["*"])
    formula = model.formula_string()

    assert "A_1*exp(-Lambda*t)" in formula
    assert "exp(-(sigma*t)^2)" in formula
    assert "1*exp(-(sigma*t)^2)" not in formula


def test_oscillatory_field_matches_frequency_component() -> None:
    t = np.linspace(0.0, 6.0, 200)
    field_gauss = 750.0
    amplitude = 0.7
    phase = 0.2

    frequency_mhz = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * field_gauss

    by_field = COMPONENTS["OscillatoryField"].function(
        t,
        A=amplitude,
        field=field_gauss,
        phase=phase,
    )
    by_frequency = COMPONENTS["Oscillatory"].function(
        t,
        A=amplitude,
        frequency=frequency_mhz,
        phase=phase,
    )

    assert np.allclose(by_field, by_frequency)


def test_parenthesized_expression_changes_evaluation_order() -> None:
    t = np.linspace(0.0, 1.0, 10)

    # Under suppression rules for multiplied singletons: 1 * (3 + 4) = 7
    model = CompositeModel(
        ["Constant", "Constant", "Constant"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )
    out = model.function(t, A_bg_2=3.0, A_bg_3=4.0)

    assert np.allclose(out, np.full_like(t, 7.0))


def test_parenthesized_model_serialization_round_trip() -> None:
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )

    restored = CompositeModel.from_dict(model.to_dict())
    assert restored.component_names == model.component_names
    assert restored.operators == model.operators
    assert restored.open_parentheses == model.open_parentheses
    assert restored.close_parentheses == model.close_parentheses


def test_fraction_group_serialization_round_trip() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")

    restored = CompositeModel.from_dict(model.to_dict())
    assert restored.fraction_groups == [(0, 1)]
    assert restored.component_expression_string() == "(Exponential + Gaussian){frac}"


def test_fraction_group_uses_group_amplitude_and_fraction_parameters() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")

    assert "A_1" in model.param_names
    assert "Lambda" in model.param_names
    assert "sigma" in model.param_names
    assert "fraction_1" in model.param_names
    assert "fraction_2" in model.param_names
    assert "A_2" not in model.param_names


def test_fraction_group_evaluation_normalizes_component_weights() -> None:
    t = np.linspace(0.0, 1.0, 5)
    model = CompositeModel.from_expression("( Exponential + Exponential + Constant ){frac}")

    out = model.function(
        t,
        A_1=12.0,
        Lambda_1=0.2,
        Lambda_2=0.5,
        fraction_1=1.0,
        fraction_2=1.0,
        fraction_3=2.0,
    )
    expected = 12.0 * (0.25 * np.exp(-0.2 * t) + 0.25 * np.exp(-0.5 * t) + 0.5 * np.ones_like(t))

    assert np.allclose(out, expected)


def test_fraction_group_preserves_outer_multiplicative_amplitude_suppression() -> None:
    model = CompositeModel.from_expression("Oscillatory * ( Exponential + Gaussian ){frac}")

    assert "A_1" not in model.param_names
    assert "A_2" in model.param_names
    assert "fraction_1" in model.param_names
    assert "fraction_2" in model.param_names


def test_parenthesized_product_suppresses_leading_amplitude() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )

    assert "A_1" not in model.param_names
    assert "A_2" in model.param_names
    assert "A_3" in model.param_names

    t = np.linspace(0.0, 1.0, 5)
    kwargs = {
        "Lambda_1": 0.2,
        "A_2": 2.0,
        "Lambda_2": 0.3,
        "A_3": 3.0,
        "Lambda_3": 0.4,
    }
    out = model.function(t, **kwargs)
    exp1 = np.exp(-0.2 * t)
    exp2 = np.exp(-0.3 * t)
    exp3 = np.exp(-0.4 * t)
    expected = exp1 * (2.0 * exp2 + 3.0 * exp3)
    assert np.allclose(out, expected)


def test_parenthesized_product_suppresses_trailing_amplitude() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential"],
        operators=["+", "*"],
        open_parentheses=[1, 0, 0],
        close_parentheses=[0, 1, 0],
    )

    assert "A_1" in model.param_names
    assert "A_2" in model.param_names
    assert "A_3" not in model.param_names

    t = np.linspace(0.0, 1.0, 5)
    kwargs = {
        "A_1": 2.0,
        "Lambda_1": 0.2,
        "A_2": 3.0,
        "Lambda_2": 0.3,
        "Lambda_3": 0.4,
    }
    out = model.function(t, **kwargs)
    exp1 = np.exp(-0.2 * t)
    exp2 = np.exp(-0.3 * t)
    exp3 = np.exp(-0.4 * t)
    expected = (2.0 * exp1 + 3.0 * exp2) * exp3
    assert np.allclose(out, expected)


def test_multiplying_two_additive_groups_keeps_group_amplitudes() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential", "Exponential"],
        operators=["+", "*", "+"],
        open_parentheses=[1, 0, 1, 0],
        close_parentheses=[0, 1, 0, 1],
    )

    assert "A_1" in model.param_names
    assert "A_2" in model.param_names
    assert "A_3" in model.param_names
    assert "A_4" in model.param_names


def test_lhs_additive_group_with_constant_suppresses_rhs_amplitude() -> None:
    model = CompositeModel(
        ["Gaussian", "Constant", "OscillatoryField"],
        operators=["+", "*"],
        open_parentheses=[1, 0, 0],
        close_parentheses=[0, 1, 0],
    )

    assert "A_1" in model.param_names
    assert "A_bg" in model.param_names
    assert "A_3" not in model.param_names

    formula = model.formula_string()
    assert "(A_1*exp(-(sigma*t)^2) + A_bg) * cos(2*pi*gamma_mu*field*t + phase)" in formula


def test_lhs_additive_group_suppresses_rhs_constant_amplitude() -> None:
    model = CompositeModel(
        ["Gaussian", "Constant", "Constant"],
        operators=["+", "*"],
        open_parentheses=[1, 0, 0],
        close_parentheses=[0, 1, 0],
    )

    assert "A" in model.param_names
    assert "A_bg_2" in model.param_names
    assert "A_bg_3" not in model.param_names

    formula = model.formula_string()
    assert "* 1" not in formula


def test_rhs_additive_group_suppresses_lhs_constant_amplitude() -> None:
    model = CompositeModel(
        ["Constant", "Gaussian", "Constant"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )

    assert "A_bg_1" not in model.param_names
    assert "A" in model.param_names
    assert "A_bg_3" in model.param_names

    formula = model.formula_string()
    assert "1 *" not in formula

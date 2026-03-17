"""Tests for composite fit-function model construction and evaluation."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel


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

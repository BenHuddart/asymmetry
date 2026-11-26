"""Tests for safe composite-parameter expressions and uncertainty propagation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.fitting.composite_parameters import (
    CompositeExpression,
    CompositeExpressionError,
    validate_composite_expression,
)


def test_validate_expression_rejects_unknown_symbol() -> None:
    ok, message = validate_composite_expression(
        "A0 + Missing",
        allowed_symbols=["A0", "Lambda"],
    )
    assert ok is False
    assert message is not None
    assert "Missing" in message


def test_validate_expression_accepts_known_symbols_and_functions() -> None:
    ok, message = validate_composite_expression(
        "sin(A0) + Lambda^2 + pi",
        allowed_symbols=["A0", "Lambda"],
    )
    assert ok is True
    assert message is None


def test_expression_operator_precedence_and_parentheses() -> None:
    expr = CompositeExpression("A0 + Lambda * 2")
    value, grad = expr.evaluate({"A0": 1.0, "Lambda": 3.0})
    assert value == pytest.approx(7.0)
    assert grad["A0"] == pytest.approx(1.0)
    assert grad["Lambda"] == pytest.approx(2.0)

    expr_paren = CompositeExpression("(A0 + Lambda) * 2")
    value_paren, grad_paren = expr_paren.evaluate({"A0": 1.0, "Lambda": 3.0})
    assert value_paren == pytest.approx(8.0)
    assert grad_paren["A0"] == pytest.approx(2.0)
    assert grad_paren["Lambda"] == pytest.approx(2.0)


def test_expression_supports_unary_and_power_alias() -> None:
    expr = CompositeExpression("-A0 + Lambda^2")
    value, grad = expr.evaluate({"A0": 2.0, "Lambda": 4.0})
    assert value == pytest.approx(14.0)
    assert grad["A0"] == pytest.approx(-1.0)
    assert grad["Lambda"] == pytest.approx(8.0)


def test_expression_trig_and_log_functions() -> None:
    expr = CompositeExpression("sin(A0) + log(Lambda)")
    value, grad = expr.evaluate({"A0": 0.2, "Lambda": 2.5})
    assert value == pytest.approx(math.sin(0.2) + math.log(2.5))
    assert grad["A0"] == pytest.approx(math.cos(0.2))
    assert grad["Lambda"] == pytest.approx(1.0 / 2.5)


def test_invalid_function_raises_error() -> None:
    with pytest.raises(CompositeExpressionError, match="Unsupported function"):
        CompositeExpression("custom(A0)")


def test_division_by_zero_raises_error() -> None:
    expr = CompositeExpression("A0 / Lambda")
    with pytest.raises(CompositeExpressionError, match="Division by zero"):
        expr.evaluate({"A0": 1.0, "Lambda": 0.0})


def test_log_domain_error_raises_error() -> None:
    expr = CompositeExpression("log(A0)")
    with pytest.raises(CompositeExpressionError, match="positive"):
        expr.evaluate({"A0": -1.0})


def test_uncertainty_propagation_diagonal_fallback() -> None:
    expr = CompositeExpression("A0 + 2*Lambda")
    evaluated = expr.evaluate_with_uncertainty(
        {"A0": 3.0, "Lambda": 4.0},
        {"A0": 0.1, "Lambda": 0.2},
    )
    expected_sigma = math.sqrt((0.1**2) + ((2.0 * 0.2) ** 2))
    assert evaluated.value == pytest.approx(11.0)
    assert evaluated.uncertainty == pytest.approx(expected_sigma)


def test_uncertainty_propagation_with_covariance_array() -> None:
    expr = CompositeExpression("A0 + Lambda")
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    evaluated = expr.evaluate_with_uncertainty(
        {"A0": 1.0, "Lambda": 2.0},
        covariance=covariance,
        covariance_order=["A0", "Lambda"],
    )
    # var = 1*0.04 + 1*0.09 + 2*0.01
    assert evaluated.uncertainty == pytest.approx(math.sqrt(0.15))


@pytest.mark.parametrize("errors", [{}, {"A0": math.nan}])
def test_uncertainty_propagation_with_covariance_mapping(errors) -> None:
    expr = CompositeExpression("A0 - Lambda")
    evaluated = expr.evaluate_with_uncertainty(
        {"A0": 1.0, "Lambda": 2.0},
        errors,
        covariance={
            "A0": {"A0": 0.04, "Lambda": 0.01},
            "Lambda": {"A0": 0.01, "Lambda": 0.09},
        },
    )
    # var = [1, -1] [[0.04, 0.01], [0.01, 0.09]] [1, -1]^T = 0.11
    assert evaluated.uncertainty == pytest.approx(math.sqrt(0.11))


def test_covariance_order_required_for_ndarray() -> None:
    expr = CompositeExpression("A0 + Lambda")
    with pytest.raises(CompositeExpressionError, match="covariance_order"):
        expr.evaluate_with_uncertainty(
            {"A0": 1.0, "Lambda": 2.0},
            {"A0": 0.2, "Lambda": 0.3},
            covariance=np.eye(2),
        )


def test_variable_exponent_derivative_requires_positive_base() -> None:
    expr = CompositeExpression("A0^Lambda")
    with pytest.raises(CompositeExpressionError, match="positive base"):
        expr.evaluate({"A0": -1.0, "Lambda": 0.5})


@pytest.mark.parametrize("sigma", [float("nan"), float("inf")])
def test_explicit_unavailable_uncertainty_keeps_value_and_zero_derivative(sigma):
    value = CompositeExpression("A+B").evaluate_with_uncertainty(
        {"A": 2.0, "B": 1.0}, {"A": sigma, "B": 0.0}
    )
    assert value.value == 3.0
    assert np.isnan(value.uncertainty)
    cancelled = CompositeExpression("A-A+B").evaluate_with_uncertainty(
        {"A": 2.0, "B": 1.0}, {"A": sigma, "B": 0.0}
    )
    assert cancelled.uncertainty == 0.0


@pytest.mark.parametrize("errors", [{}, {"a": 0.2}, {"b": 0.3}, {"a": 0.2, "b": math.inf}])
def test_missing_marginal_keeps_finite_central_value(errors):
    evaluated = CompositeExpression("a-b").evaluate_with_uncertainty({"a": 5.1, "b": 5.0}, errors)

    assert evaluated.value == pytest.approx(0.1)
    assert math.isnan(evaluated.uncertainty)


def test_algebraic_cancellation_needs_no_marginal_error():
    evaluated = CompositeExpression("a-a+1").evaluate_with_uncertainty({"a": 5.1})

    assert evaluated.value == 1.0
    assert evaluated.uncertainty == 0.0


@pytest.mark.parametrize("value", [5.1, 1e-200])
def test_ratio_of_a_parameter_to_itself_needs_no_marginal_error(value):
    evaluated = CompositeExpression("a/a").evaluate_with_uncertainty({"a": value})

    assert evaluated.value == 1.0
    assert evaluated.gradient["a"] == 0.0
    assert evaluated.uncertainty == 0.0


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(a-a)", 0.0),
        ("(a-a)^0.5", 0.0),
        ("asin(a-a+1)", math.pi / 2),
        ("acos(a-a+1)", 0.0),
    ],
)
def test_cancelled_chain_at_a_singular_endpoint_is_exact(expression, expected):
    evaluated = CompositeExpression(expression).evaluate_with_uncertainty({"a": 5.1})

    assert evaluated.value == pytest.approx(expected)
    assert evaluated.gradient["a"] == 0.0
    assert evaluated.uncertainty == 0.0


@pytest.mark.parametrize(
    ("expression", "value", "expected"),
    [
        ("sqrt(a)", 0.0, 0.0),
        ("sqrt(a*a)", 0.0, 0.0),
        ("a^0.5", 0.0, 0.0),
        ("asin(a)", 1.0, math.pi / 2),
        ("acos(a)", 1.0, 0.0),
    ],
)
def test_active_singular_derivative_keeps_the_central_value(expression, value, expected):
    evaluated = CompositeExpression(expression).evaluate_with_uncertainty({"a": value}, {"a": 0.1})

    assert evaluated.value == pytest.approx(expected)
    assert math.isnan(evaluated.uncertainty)


def test_singular_chain_distinguishes_cancelled_and_stationary_inputs():
    evaluated = CompositeExpression("sqrt(b*b+(a-a))").evaluate_with_uncertainty(
        {"a": 5.1, "b": 0.0}, {"b": 0.1}
    )

    assert evaluated.value == 0.0
    assert evaluated.gradient["a"] == 0.0
    assert math.isnan(evaluated.uncertainty)


@pytest.mark.parametrize(
    ("expression", "value"), [("sqrt(a)", -1.0), ("asin(a)", 1.1), ("acos(a)", -1.1)]
)
def test_out_of_domain_composite_value_is_rejected(expression, value):
    with pytest.raises(CompositeExpressionError):
        CompositeExpression(expression).evaluate_with_uncertainty({"a": value})


def test_fixed_input_is_exact_even_without_a_marginal():
    evaluated = CompositeExpression("a-b").evaluate_with_uncertainty(
        {"a": 5.1, "b": 5.0}, {"a": 0.2}, fixed_parameters={"b"}
    )

    assert evaluated.value == pytest.approx(0.1)
    assert evaluated.uncertainty == pytest.approx(0.2)


def test_invalid_covariance_diagonal_uses_a_supplied_marginal():
    evaluated = CompositeExpression("a-b").evaluate_with_uncertainty(
        {"a": 5.1, "b": 5.0},
        {"a": 0.2},
        covariance={"a": {"a": -1.0, "b": 0.01}, "b": {"b": 0.09}},
    )

    assert evaluated.uncertainty == pytest.approx(math.sqrt(0.04 + 0.09 - 2.0 * 0.01))

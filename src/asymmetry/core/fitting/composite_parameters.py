"""Safe composite-parameter expression parsing and uncertainty propagation.

This module evaluates scalar expressions built from fitted parameter values,
allow-listed math functions, and numeric constants. It also propagates
uncertainties using analytic first derivatives and a covariance matrix when
available.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np


class CompositeExpressionError(ValueError):
    """Raised when a composite-parameter expression is invalid."""


@dataclass(frozen=True)
class CompositeParameterDefinition:
    """Serializable definition for a user-created composite parameter."""

    name: str
    expression: str


@dataclass(frozen=True)
class CompositeEvaluation:
    """Evaluated value, propagated uncertainty, and derivative metadata."""

    value: float
    uncertainty: float
    gradient: dict[str, float]
    referenced_symbols: tuple[str, ...]


@dataclass(frozen=True)
class FunctionSpec:
    """Value and derivative callbacks for unary allow-listed functions."""

    value_fn: Callable[[float], float]
    derivative_fn: Callable[[float], float]


def _log_derivative(x: float) -> float:
    if x <= 0.0:
        raise CompositeExpressionError("log requires a positive argument")
    return 1.0 / x


def _log10_derivative(x: float) -> float:
    if x <= 0.0:
        raise CompositeExpressionError("log10 requires a positive argument")
    return 1.0 / (x * math.log(10.0))


def _safe_log(x: float) -> float:
    if x <= 0.0:
        raise CompositeExpressionError("log requires a positive argument")
    return math.log(x)


def _safe_log10(x: float) -> float:
    if x <= 0.0:
        raise CompositeExpressionError("log10 requires a positive argument")
    return math.log10(x)


def _sqrt_derivative(x: float) -> float:
    if x < 0.0:
        raise CompositeExpressionError("sqrt requires a non-negative argument")
    if x == 0.0:
        return float("inf")
    return 0.5 / math.sqrt(x)


def _safe_sqrt(x: float) -> float:
    if x < 0.0:
        raise CompositeExpressionError("sqrt requires a non-negative argument")
    return math.sqrt(x)


def _asin_derivative(x: float) -> float:
    if x <= -1.0 or x >= 1.0:
        raise CompositeExpressionError("asin requires -1 < x < 1")
    return 1.0 / math.sqrt(1.0 - x * x)


def _safe_asin(x: float) -> float:
    if x < -1.0 or x > 1.0:
        raise CompositeExpressionError("asin requires -1 <= x <= 1")
    return math.asin(x)


def _acos_derivative(x: float) -> float:
    if x <= -1.0 or x >= 1.0:
        raise CompositeExpressionError("acos requires -1 < x < 1")
    return -1.0 / math.sqrt(1.0 - x * x)


def _safe_acos(x: float) -> float:
    if x < -1.0 or x > 1.0:
        raise CompositeExpressionError("acos requires -1 <= x <= 1")
    return math.acos(x)


def _abs_derivative(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    # The derivative is undefined at zero; use 0 as a stable local approximation.
    return 0.0


DEFAULT_FUNCTIONS: Final[dict[str, FunctionSpec]] = {
    "sin": FunctionSpec(math.sin, math.cos),
    "cos": FunctionSpec(math.cos, lambda x: -math.sin(x)),
    "tan": FunctionSpec(math.tan, lambda x: 1.0 / (math.cos(x) ** 2)),
    "asin": FunctionSpec(_safe_asin, _asin_derivative),
    "acos": FunctionSpec(_safe_acos, _acos_derivative),
    "atan": FunctionSpec(math.atan, lambda x: 1.0 / (1.0 + x * x)),
    "sinh": FunctionSpec(math.sinh, math.cosh),
    "cosh": FunctionSpec(math.cosh, math.sinh),
    "tanh": FunctionSpec(math.tanh, lambda x: 1.0 - math.tanh(x) ** 2),
    "exp": FunctionSpec(math.exp, math.exp),
    "log": FunctionSpec(_safe_log, _log_derivative),
    "log10": FunctionSpec(_safe_log10, _log10_derivative),
    "sqrt": FunctionSpec(_safe_sqrt, _sqrt_derivative),
    "abs": FunctionSpec(abs, _abs_derivative),
}

DEFAULT_CONSTANTS: Final[dict[str, float]] = {
    "pi": math.pi,
    "e": math.e,
}


@dataclass(frozen=True)
class _EvalResult:
    value: float
    gradient: dict[str, float]


class CompositeExpression:
    """Parsed and validated scalar expression with analytic derivatives."""

    def __init__(
        self,
        expression: str,
        *,
        functions: Mapping[str, FunctionSpec] | None = None,
        constants: Mapping[str, float] | None = None,
    ) -> None:
        self.expression = str(expression).strip()
        if not self.expression:
            raise CompositeExpressionError("Expression cannot be empty")

        self._functions = dict(DEFAULT_FUNCTIONS)
        if functions is not None:
            self._functions.update(functions)

        self._constants = dict(DEFAULT_CONSTANTS)
        if constants is not None:
            self._constants.update(constants)

        self._ast = self._parse(self.expression)
        self.referenced_symbols = tuple(sorted(self._collect_symbols(self._ast)))

    def unknown_symbols(self, allowed_symbols: Sequence[str] | set[str]) -> list[str]:
        allowed = set(allowed_symbols)
        return [name for name in self.referenced_symbols if name not in allowed]

    def evaluate(self, symbol_values: Mapping[str, float]) -> tuple[float, dict[str, float]]:
        result = self._eval_node(self._ast, symbol_values)
        if not np.isfinite(result.value):
            raise CompositeExpressionError("Expression evaluated to a non-finite value")
        grad: dict[str, float] = {
            name: float(val) for name, val in result.gradient.items() if np.isfinite(val)
        }
        return float(result.value), grad

    def evaluate_with_uncertainty(
        self,
        symbol_values: Mapping[str, float],
        symbol_uncertainties: Mapping[str, float] | None = None,
        *,
        covariance: np.ndarray | Mapping[str, Mapping[str, float]] | None = None,
        covariance_order: Sequence[str] | None = None,
    ) -> CompositeEvaluation:
        value, gradient = self.evaluate(symbol_values)
        variance = self._propagate_variance(
            gradient,
            symbol_uncertainties or {},
            covariance=covariance,
            covariance_order=covariance_order,
        )
        uncertainty = float(math.sqrt(max(0.0, variance)))
        return CompositeEvaluation(
            value=float(value),
            uncertainty=uncertainty,
            gradient=gradient,
            referenced_symbols=self.referenced_symbols,
        )

    def _parse(self, expression: str) -> ast.AST:
        normalized = expression.replace("^", "**")
        try:
            parsed = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise CompositeExpressionError(f"Invalid expression syntax: {exc.msg}") from exc

        self._validate_node(parsed.body)

        return parsed.body

    def _validate_node(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise CompositeExpressionError("Only numeric constants are allowed")
            return

        if isinstance(node, ast.Name):
            return

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise CompositeExpressionError("Unsupported unary operator")
            self._validate_node(node.operand)
            return

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise CompositeExpressionError("Unsupported arithmetic operator")
            self._validate_node(node.left)
            self._validate_node(node.right)
            return

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise CompositeExpressionError("Only direct function calls are allowed")
            if node.func.id not in self._functions:
                raise CompositeExpressionError(f"Unsupported function: {node.func.id}")
            if len(node.args) != 1 or node.keywords:
                raise CompositeExpressionError("Functions must be unary and positional")
            self._validate_node(node.args[0])
            return

        raise CompositeExpressionError("Unsupported expression element")

    def _collect_symbols(self, node: ast.AST) -> set[str]:
        symbols: set[str] = set()

        class _Collector(ast.NodeVisitor):
            def visit_Name(self, subnode: ast.Name) -> None:  # noqa: N802
                if (
                    subnode.id not in self_outer._functions
                    and subnode.id not in self_outer._constants
                ):
                    symbols.add(subnode.id)

        self_outer = self
        _Collector().visit(node)
        return symbols

    def _eval_node(self, node: ast.AST, symbol_values: Mapping[str, float]) -> _EvalResult:
        if isinstance(node, ast.Constant):
            return _EvalResult(float(node.value), {})

        if isinstance(node, ast.Name):
            if node.id in self._constants:
                return _EvalResult(float(self._constants[node.id]), {})
            if node.id not in symbol_values:
                raise CompositeExpressionError(f"Unknown parameter reference: {node.id}")
            value = float(symbol_values[node.id])
            if not np.isfinite(value):
                raise CompositeExpressionError(f"Non-finite value for parameter '{node.id}'")
            return _EvalResult(value, {node.id: 1.0})

        if isinstance(node, ast.UnaryOp):
            item = self._eval_node(node.operand, symbol_values)
            if isinstance(node.op, ast.UAdd):
                return item
            return _EvalResult(-item.value, {name: -val for name, val in item.gradient.items()})

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, symbol_values)
            right = self._eval_node(node.right, symbol_values)
            if isinstance(node.op, ast.Add):
                return _EvalResult(
                    left.value + right.value,
                    _merge_gradients(left.gradient, right.gradient, 1.0, 1.0),
                )
            if isinstance(node.op, ast.Sub):
                return _EvalResult(
                    left.value - right.value,
                    _merge_gradients(left.gradient, right.gradient, 1.0, -1.0),
                )
            if isinstance(node.op, ast.Mult):
                grad = {}
                for name, val in left.gradient.items():
                    grad[name] = grad.get(name, 0.0) + val * right.value
                for name, val in right.gradient.items():
                    grad[name] = grad.get(name, 0.0) + val * left.value
                return _EvalResult(left.value * right.value, grad)
            if isinstance(node.op, ast.Div):
                if right.value == 0.0:
                    raise CompositeExpressionError("Division by zero")
                denom = right.value * right.value
                grad = {}
                for name, val in left.gradient.items():
                    grad[name] = grad.get(name, 0.0) + val / right.value
                for name, val in right.gradient.items():
                    grad[name] = grad.get(name, 0.0) - (left.value * val) / denom
                return _EvalResult(left.value / right.value, grad)
            if isinstance(node.op, ast.Pow):
                return self._eval_power(left, right)
            raise CompositeExpressionError("Unsupported arithmetic operator")

        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            spec = self._functions[node.func.id]
            arg = self._eval_node(node.args[0], symbol_values)
            try:
                value = float(spec.value_fn(arg.value))
                derivative = float(spec.derivative_fn(arg.value))
            except CompositeExpressionError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise CompositeExpressionError(
                    f"Function '{node.func.id}' failed for value {arg.value!r}"
                ) from exc
            grad = {name: derivative * val for name, val in arg.gradient.items()}
            return _EvalResult(value, grad)

        raise CompositeExpressionError("Unsupported expression element")

    def _eval_power(self, left: _EvalResult, right: _EvalResult) -> _EvalResult:
        left_grad = left.gradient
        right_grad = right.gradient

        if right_grad and left.value <= 0.0:
            raise CompositeExpressionError(
                "Variable exponent requires a positive base for derivative"
            )

        try:
            value = float(left.value**right.value)
        except Exception as exc:  # pragma: no cover - defensive
            raise CompositeExpressionError("Invalid power operation") from exc

        if not right_grad:
            # Constant exponent: d(u^c) = c*u^(c-1)*du
            if left.value == 0.0 and right.value < 1.0:
                raise CompositeExpressionError("Power derivative undefined at zero base")
            factor = right.value * (left.value ** (right.value - 1.0))
            return _EvalResult(value, {name: factor * val for name, val in left_grad.items()})

        grad: dict[str, float] = {}
        ln_base = math.log(left.value)
        for name, val in left_grad.items():
            grad[name] = grad.get(name, 0.0) + value * right.value * val / left.value
        for name, val in right_grad.items():
            grad[name] = grad.get(name, 0.0) + value * ln_base * val
        return _EvalResult(value, grad)

    def _propagate_variance(
        self,
        gradient: Mapping[str, float],
        symbol_uncertainties: Mapping[str, float],
        *,
        covariance: np.ndarray | Mapping[str, Mapping[str, float]] | None,
        covariance_order: Sequence[str] | None,
    ) -> float:
        symbols = [name for name in self.referenced_symbols if name in gradient]
        if not symbols:
            return 0.0

        jac = np.array([float(gradient[name]) for name in symbols], dtype=float)
        cov = np.zeros((len(symbols), len(symbols)), dtype=float)

        # Start with diagonal fallback from provided uncertainties.
        for i, name in enumerate(symbols):
            sigma = float(symbol_uncertainties.get(name, 0.0))
            if np.isfinite(sigma) and sigma > 0.0:
                cov[i, i] = sigma * sigma

        if covariance is not None:
            if isinstance(covariance, np.ndarray):
                if covariance_order is None:
                    raise CompositeExpressionError(
                        "covariance_order is required for ndarray covariance"
                    )
                self._overlay_ndarray_covariance(cov, symbols, covariance, covariance_order)
            else:
                self._overlay_mapping_covariance(cov, symbols, covariance)

        variance = float(jac @ cov @ jac.T)
        if not np.isfinite(variance):
            raise CompositeExpressionError("Propagated variance is non-finite")
        return max(0.0, variance)

    def _overlay_ndarray_covariance(
        self,
        out: np.ndarray,
        symbols: list[str],
        covariance: np.ndarray,
        covariance_order: Sequence[str],
    ) -> None:
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise CompositeExpressionError("covariance must be a square matrix")
        if len(covariance_order) != cov.shape[0]:
            raise CompositeExpressionError("covariance_order length does not match covariance")

        index_by_name = {name: idx for idx, name in enumerate(covariance_order)}
        for i, name_i in enumerate(symbols):
            idx_i = index_by_name.get(name_i)
            if idx_i is None:
                continue
            for j, name_j in enumerate(symbols):
                idx_j = index_by_name.get(name_j)
                if idx_j is None:
                    continue
                val = float(cov[idx_i, idx_j])
                if np.isfinite(val):
                    out[i, j] = val

    def _overlay_mapping_covariance(
        self,
        out: np.ndarray,
        symbols: list[str],
        covariance: Mapping[str, Mapping[str, float]],
    ) -> None:
        for i, name_i in enumerate(symbols):
            row = covariance.get(name_i)
            if row is None:
                continue
            for j, name_j in enumerate(symbols):
                if name_j in row:
                    val = float(row[name_j])
                    if np.isfinite(val):
                        out[i, j] = val


def _merge_gradients(
    left: Mapping[str, float],
    right: Mapping[str, float],
    left_scale: float,
    right_scale: float,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for name, val in left.items():
        merged[name] = merged.get(name, 0.0) + left_scale * val
    for name, val in right.items():
        merged[name] = merged.get(name, 0.0) + right_scale * val
    return merged


def validate_composite_expression(
    expression: str,
    *,
    allowed_symbols: Sequence[str],
    functions: Mapping[str, FunctionSpec] | None = None,
    constants: Mapping[str, float] | None = None,
) -> tuple[bool, str | None]:
    """Validate expression syntax and parameter references.

    Returns ``(True, None)`` when valid, otherwise ``(False, message)``.
    """

    try:
        parsed = CompositeExpression(expression, functions=functions, constants=constants)
    except CompositeExpressionError as exc:
        return False, str(exc)

    unknown = parsed.unknown_symbols(allowed_symbols)
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        return False, f"Unknown parameter reference(s): {unknown_text}"

    return True, None

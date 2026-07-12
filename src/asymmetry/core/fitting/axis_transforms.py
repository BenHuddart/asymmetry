"""Per-axis transforms for parameter-trend plots (GUI-free).

An :class:`AxisTransform` maps a plotted axis quantity -- either an independent
abscissa (field, temperature, run) or a fitted-parameter ordinate -- through a
scalar function, propagating 1-sigma uncertainties analytically.  Presets cover
the common muSR linearisations:

* **Redfield** -- plot ``1/lambda`` against ``(mu0 H)^2`` (reciprocal on the
  ordinate, square on the abscissa); a straight-line trend fit then yields the
  Redfield slope/intercept directly.
* **Arrhenius** -- plot ``ln lambda`` against ``1/T`` (log on the ordinate,
  reciprocal on the abscissa); the slope is the activation energy.

Any single-variable expression (in the variable :data:`AXIS_VARIABLE`) is
available via :data:`CUSTOM`, reusing the same safe evaluator and
uncertainty-propagation machinery as composite parameters
(:mod:`asymmetry.core.fitting.composite_parameters`).

The transform is applied at the trend panel's data-assembly boundary, so the
plotted points, the error bars, *and* the Model-Fit trend line all operate on
the transformed coordinates -- fitting a ``Linear`` model to a Redfield-
transformed curve is exactly the Redfield line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np

from asymmetry.core.fitting.composite_parameters import (
    CompositeExpression,
    CompositeExpressionError,
)

#: The single free variable a transform expression may reference.
AXIS_VARIABLE: Final = "x"

# Preset kind identifiers (also the serialised ``kind`` values).
IDENTITY: Final = "identity"
RECIPROCAL: Final = "reciprocal"
SQUARE: Final = "square"
LOG: Final = "log"
LOG10: Final = "log10"
SQRT: Final = "sqrt"
CUSTOM: Final = "custom"

#: Canonical expression (in :data:`AXIS_VARIABLE`) backing each non-identity
#: preset.  ``CUSTOM`` carries its own expression on the instance.
_PRESET_EXPRESSIONS: Final[dict[str, str]] = {
    RECIPROCAL: "1/x",
    SQUARE: "x**2",
    LOG: "log(x)",
    LOG10: "log10(x)",
    SQRT: "sqrt(x)",
}

#: Menu labels for the preset chooser (identity first, custom last).
PRESET_LABELS: Final[dict[str, str]] = {
    IDENTITY: "None",
    RECIPROCAL: "1/x  (reciprocal)",
    SQUARE: "x²  (square)",
    LOG: "ln x",
    LOG10: "log₁₀ x",
    SQRT: "√x",
    CUSTOM: "Custom…",
}

#: Ordered preset kinds for populating a chooser.
PRESET_ORDER: Final[tuple[str, ...]] = (
    IDENTITY,
    RECIPROCAL,
    SQUARE,
    LOG,
    LOG10,
    SQRT,
    CUSTOM,
)

_ALL_KINDS: Final[frozenset[str]] = frozenset(PRESET_ORDER)

# Match the axis variable only as a whole word, so ``exp`` / ``max`` are left
# alone when substituting a base label into a custom expression.
_VARIABLE_TOKEN: Final = re.compile(rf"\b{re.escape(AXIS_VARIABLE)}\b")

# A base label is "atomic" (no bracketing needed) when it is a single run of
# letters/digits/greek/sub-super-scripts with no spaces or operators.
_ATOMIC_LABEL: Final = re.compile(r"^[\w°-￿]+$")

# A "simple" unit is a single atomic token optionally suffixed ``⁻¹`` (covers
# every unit ``get_param_info`` emits: G, K, T, MHz, meV, µs⁻¹, …). Only such
# units admit the trivial reciprocal/square unit algebra; anything else is
# rendered by bracketing the whole dimensioned quantity.
_SIMPLE_UNIT: Final = re.compile(r"^([^\s()²³⁴⁵⁶⁷⁸⁹⁻]+)(⁻¹)?$")


def _simple_unit(unit: str) -> tuple[str, bool] | None:
    """Return ``(token, inverted)`` for a simple unit, else ``None``."""
    match = _SIMPLE_UNIT.match(unit.strip())
    if not match:
        return None
    return match.group(1), bool(match.group(2))


@lru_cache(maxsize=128)
def _compile(expression: str) -> CompositeExpression:
    """Compile (and cache) a transform expression."""
    return CompositeExpression(expression)


def validate_axis_expression(expression: str) -> tuple[bool, str | None]:
    """Validate a custom axis expression.

    Returns ``(True, None)`` when the expression parses, references only the
    axis variable :data:`AXIS_VARIABLE`, and actually depends on it; otherwise
    ``(False, message)``.
    """
    text = str(expression).strip()
    if not text:
        return False, "Expression cannot be empty"
    try:
        expr = CompositeExpression(text)
    except CompositeExpressionError as exc:
        return False, str(exc)
    unknown = expr.unknown_symbols({AXIS_VARIABLE})
    if unknown:
        joined = ", ".join(sorted(unknown))
        return False, f"Use only the variable '{AXIS_VARIABLE}' (found: {joined})"
    if AXIS_VARIABLE not in expr.referenced_symbols:
        return False, f"Expression must reference the axis variable '{AXIS_VARIABLE}'"
    return True, None


def _atomic(label: str) -> bool:
    return bool(_ATOMIC_LABEL.match(label.strip()))


@dataclass(frozen=True)
class AxisTransform:
    """A scalar transform applied to one plotted axis.

    ``kind`` is one of the preset identifiers or :data:`CUSTOM`.  ``expression``
    is only meaningful (and only stored) when ``kind == CUSTOM``.
    """

    kind: str = IDENTITY
    expression: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _ALL_KINDS:
            raise ValueError(f"Unknown axis-transform kind: {self.kind!r}")

    # -- constructors ----------------------------------------------------
    @classmethod
    def identity(cls) -> AxisTransform:
        return cls(IDENTITY)

    @classmethod
    def preset(cls, kind: str) -> AxisTransform:
        if kind == CUSTOM:
            raise ValueError("Use AxisTransform.custom() for custom expressions")
        return cls(kind)

    @classmethod
    def custom(cls, expression: str) -> AxisTransform:
        return cls(CUSTOM, str(expression).strip())

    # -- introspection ---------------------------------------------------
    @property
    def is_identity(self) -> bool:
        return self.kind == IDENTITY

    @property
    def expression_text(self) -> str:
        """The expression backing this transform, in :data:`AXIS_VARIABLE`."""
        if self.kind == CUSTOM:
            return self.expression
        return _PRESET_EXPRESSIONS.get(self.kind, AXIS_VARIABLE)

    def validate(self) -> tuple[bool, str | None]:
        if self.kind != CUSTOM:
            return True, None
        return validate_axis_expression(self.expression)

    # -- evaluation ------------------------------------------------------
    def apply(
        self,
        values: np.ndarray | list[float],
        errors: np.ndarray | list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform ``values`` (and ``errors``) element-wise.

        Returns ``(transformed_values, transformed_errors)`` as float arrays of
        the same shape.  A non-finite input value, or one on which the transform
        is undefined (``1/0``, ``log`` of a non-positive number, ...), maps to
        ``NaN`` in both outputs -- matching the panel convention that a NaN point
        is dropped and counted rather than plotted at a bogus coordinate.  An
        input with no usable uncertainty (``NaN`` / non-positive) yields a
        ``NaN`` output uncertainty.

        Uncertainties are propagated to **first order**
        (``σ_f ≈ |f'(x)|·σ_x``) and are therefore symmetric; this is accurate
        when the relative error is small but understates the asymmetry of the
        true interval where it is large (e.g. ``log`` / ``1/x`` of a low-signal
        point).
        """
        vals = np.asarray(values, dtype=float)
        if errors is None:
            errs = np.full(vals.shape, np.nan)
        else:
            errs = np.asarray(errors, dtype=float)
            if errs.shape != vals.shape:
                raise ValueError("values and errors must have the same shape")

        if self.is_identity:
            return vals.copy(), errs.copy()

        expr = _compile(self.expression_text)
        out_v = np.full(vals.shape, np.nan)
        out_e = np.full(vals.shape, np.nan)
        flat_v = vals.ravel()
        flat_e = errs.ravel()
        ov = out_v.ravel()
        oe = out_e.ravel()
        for i, raw in enumerate(flat_v):
            if not np.isfinite(raw):
                continue
            sigma = flat_e[i]
            has_err = np.isfinite(sigma) and sigma > 0.0
            try:
                evaluation = expr.evaluate_with_uncertainty(
                    {AXIS_VARIABLE: float(raw)},
                    {AXIS_VARIABLE: float(sigma) if has_err else 0.0},
                )
            except CompositeExpressionError:
                continue
            ov[i] = evaluation.value
            if has_err:
                oe[i] = evaluation.uncertainty
        return out_v, out_e

    # -- labelling -------------------------------------------------------
    def describe(self, base: str) -> str:
        """Return a plain-text axis label for this transform of ``base``.

        ``base`` is the untransformed axis symbol/label (e.g. ``"λ"``,
        ``"B (G)"``).  Produces, for example, ``"1/λ"``, ``"B²"``,
        ``"ln λ"``, or -- for a custom transform -- the expression with the
        axis variable replaced by ``base`` (``"1000/T"``).
        """
        text = base.strip()
        if self.is_identity or not text:
            return text
        wrapped = text if _atomic(text) else f"({text})"
        if self.kind == RECIPROCAL:
            return f"1/{wrapped}"
        if self.kind == SQUARE:
            return f"{wrapped}²"
        if self.kind == LOG:
            return f"ln {text}" if _atomic(text) else f"ln{wrapped}"
        if self.kind == LOG10:
            return f"log₁₀ {text}" if _atomic(text) else f"log₁₀{wrapped}"
        if self.kind == SQRT:
            return f"√{wrapped}"
        # Custom: splice the base label in for the axis variable.
        return _VARIABLE_TOKEN.sub(wrapped, self.expression_text)

    def describe_with_unit(self, symbol: str, unit: str | None = None) -> str:
        """Return a *unit-aware* axis label for this transform.

        ``symbol`` is the bare quantity symbol (``"λ"``, ``"B"``, ``"T"``) and
        ``unit`` its unit (``"µs⁻¹"``, ``"G"``, ``"K"``).  The reciprocal and
        square of a *simple* unit are computed exactly (``1/λ (µs)`` from
        ``µs⁻¹``; ``B² (G²)`` from ``G``); ``ln``/``log₁₀``/``√`` and any custom
        or non-simple case bracket the whole dimensioned quantity
        (``ln[λ (µs⁻¹)]``, ``1000/[T (K)]``) rather than guess at the algebra of
        a dimensioned logarithm. With no unit it matches :meth:`describe`.
        """
        symbol = symbol.strip()
        unit = (unit or "").strip()
        if self.is_identity:
            return f"{symbol} ({unit})" if unit else symbol
        if not unit:
            return self.describe(symbol)

        sym = symbol if _atomic(symbol) else f"({symbol})"
        dimensioned = f"{symbol} ({unit})"
        simple = _simple_unit(unit)

        if self.kind == RECIPROCAL:
            if simple is not None:
                token, inverted = simple
                new_unit = token if inverted else f"{token}⁻¹"
                return f"1/{sym} ({new_unit})"
            return f"1/[{dimensioned}]"
        if self.kind == SQUARE:
            if simple is not None:
                token, inverted = simple
                new_unit = f"{token}⁻²" if inverted else f"{token}²"
                return f"{sym}² ({new_unit})"
            return f"[{dimensioned}]²"
        if self.kind in (LOG, LOG10, SQRT):
            prefix = {LOG: "ln", LOG10: "log₁₀", SQRT: "√"}[self.kind]
            return f"{prefix}[{dimensioned}]"
        # Custom: splice the whole (bracketed) dimensioned quantity in for x.
        return _VARIABLE_TOKEN.sub(f"[{dimensioned}]", self.expression_text)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, str]:
        """Serialise, omitting the identity transform to an empty dict."""
        if self.is_identity:
            return {}
        payload = {"kind": self.kind}
        if self.kind == CUSTOM:
            payload["expression"] = self.expression
        return payload

    @classmethod
    def from_dict(cls, data: dict | None) -> AxisTransform:
        """Reconstruct from :meth:`to_dict`; unknown/empty input -> identity."""
        if not isinstance(data, dict):
            return cls.identity()
        kind = data.get("kind", IDENTITY)
        if kind not in _ALL_KINDS:
            return cls.identity()
        if kind == CUSTOM:
            return cls(CUSTOM, str(data.get("expression", "")).strip())
        return cls(kind)

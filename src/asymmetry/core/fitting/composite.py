"""Composite fit-function builder primitives.

This module exposes baseline-free muSR components that can be combined with
``+``, ``-``, ``*``, and ``/`` to produce a single model callable compatible
with :class:`asymmetry.core.fitting.engine.FitEngine`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from collections import Counter

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.models import (
    ModelDefinition,
    exponential_relaxation,
    gaussian_relaxation,
    oscillatory,
    static_gkt_zf,
    stretched_exponential,
)
from asymmetry.core.fitting.parameters import ParamInfo, get_param_info


@dataclass(frozen=True)
class ComponentDefinition:
    """Descriptor for a baseline-free component function."""

    name: str
    description: str
    function: Callable[..., NDArray[np.float64]]
    param_names: list[str]
    param_defaults: dict[str, float]
    param_info: dict[str, ParamInfo]
    formula_template: str


def _exp_component(t: NDArray, A: float, Lambda: float) -> NDArray[np.float64]:
    return exponential_relaxation(t, A0=A, Lambda=Lambda, baseline=0.0)


def _gaussian_component(t: NDArray, A: float, sigma: float) -> NDArray[np.float64]:
    return gaussian_relaxation(t, A0=A, sigma=sigma, baseline=0.0)


def _oscillatory_component(
    t: NDArray,
    A: float,
    frequency: float,
    phase: float,
) -> NDArray[np.float64]:
    return A * np.cos(2.0 * np.pi * frequency * t + phase)


def _stretched_component(
    t: NDArray,
    A: float,
    Lambda: float,
    beta: float,
) -> NDArray[np.float64]:
    return stretched_exponential(t, A0=A, Lambda=Lambda, beta=beta, baseline=0.0)


def _gkt_component(t: NDArray, A: float, Delta: float) -> NDArray[np.float64]:
    return static_gkt_zf(t, A0=A, Delta=Delta, baseline=0.0)


def _constant_component(t: NDArray, A_bg: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(t, dtype=float), fill_value=A_bg, dtype=float)


COMPONENTS: dict[str, ComponentDefinition] = {
    "Exponential": ComponentDefinition(
        name="Exponential",
        description="A exp(-Lambda t)",
        function=_exp_component,
        param_names=["A", "Lambda"],
        param_defaults={"A": 25.0, "Lambda": 0.5},
        param_info={"A": get_param_info("A"), "Lambda": get_param_info("Lambda")},
        formula_template="{A}*exp(-{Lambda}*t)",
    ),
    "Gaussian": ComponentDefinition(
        name="Gaussian",
        description="A exp(-(sigma t)^2)",
        function=_gaussian_component,
        param_names=["A", "sigma"],
        param_defaults={"A": 25.0, "sigma": 0.5},
        param_info={"A": get_param_info("A"), "sigma": get_param_info("sigma")},
        formula_template="{A}*exp(-({sigma}*t)^2)",
    ),
    "Oscillatory": ComponentDefinition(
        name="Oscillatory",
        description="A cos(2 pi f t + phase)",
        function=_oscillatory_component,
        param_names=["A", "frequency", "phase"],
        param_defaults={"A": 25.0, "frequency": 1.0, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "frequency": get_param_info("frequency"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*cos(2*pi*{frequency}*t + {phase})",
    ),
    "StretchedExponential": ComponentDefinition(
        name="StretchedExponential",
        description="A exp(-(|Lambda| t)^beta)",
        function=_stretched_component,
        param_names=["A", "Lambda", "beta"],
        param_defaults={"A": 25.0, "Lambda": 0.5, "beta": 1.0},
        param_info={
            "A": get_param_info("A"),
            "Lambda": get_param_info("Lambda"),
            "beta": get_param_info("beta"),
        },
        formula_template="{A}*exp(-(abs({Lambda})*t)^({beta}))",
    ),
    "StaticGKT_ZF": ComponentDefinition(
        name="StaticGKT_ZF",
        description="Static Gaussian Kubo-Toyabe (zero field)",
        function=_gkt_component,
        param_names=["A", "Delta"],
        param_defaults={"A": 25.0, "Delta": 0.5},
        param_info={"A": get_param_info("A"), "Delta": get_param_info("Delta")},
        formula_template=(
            "{A}*(1/3 + 2/3*(1-({Delta}*t)^2)*exp(-({Delta}*t)^2/2))"
        ),
    ),
    "Constant": ComponentDefinition(
        name="Constant",
        description="Constant background A_bg",
        function=_constant_component,
        param_names=["A_bg"],
        param_defaults={"A_bg": 0.0},
        param_info={"A_bg": get_param_info("A_bg")},
        formula_template="{A_bg}",
    ),
}


_ALLOWED_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})


class CompositeModel:
    """A flat composite model built from baseline-free components."""

    def __init__(self, component_names: list[str], operators: list[str] | None = None) -> None:
        if not component_names:
            raise ValueError("Composite model must contain at least one component")

        missing = [name for name in component_names if name not in COMPONENTS]
        if missing:
            raise ValueError(f"Unknown component(s): {missing}")

        if operators is None:
            operators = ["+"] * (len(component_names) - 1)

        if len(operators) != max(len(component_names) - 1, 0):
            raise ValueError("operators length must be len(component_names) - 1")
        if any(op not in _ALLOWED_OPERATORS for op in operators):
            raise ValueError("operators must be one of '+', '-', '*', '/'")

        self.component_names = list(component_names)
        self.operators = list(operators)
        self.components = [COMPONENTS[name] for name in component_names]
        self._param_mappings = self._build_param_mapping()

        param_names: list[str] = []
        defaults: dict[str, float] = {}
        param_info: dict[str, ParamInfo] = {}
        for mapping, component in zip(self._param_mappings, self.components, strict=True):
            for pname in component.param_names:
                unique_name = mapping[pname]
                param_names.append(unique_name)
                defaults[unique_name] = component.param_defaults[pname]
                param_info[unique_name] = get_param_info(unique_name)
        self.param_names = param_names
        self.param_defaults = defaults
        self.param_info = param_info

    def _build_param_mapping(self) -> list[dict[str, str]]:
        name_counts = Counter(
            pname for component in self.components for pname in component.param_names
        )
        mappings: list[dict[str, str]] = []
        for idx, component in enumerate(self.components, start=1):
            mapping: dict[str, str] = {}
            for pname in component.param_names:
                if pname == "A":
                    # Amplitudes are always indexed by component.
                    mapping[pname] = f"{pname}_{idx}"
                elif name_counts[pname] > 1:
                    mapping[pname] = f"{pname}_{idx}"
                else:
                    mapping[pname] = pname
            mappings.append(mapping)
        return mappings

    def function(self, t: NDArray, **kwargs: float) -> NDArray[np.float64]:
        """Evaluate the composite function with standard arithmetic precedence."""
        t_arr = np.asarray(t, dtype=float)

        values: list[NDArray[np.float64]] = []
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            component_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            values.append(np.asarray(component.function(t_arr, **component_kwargs), dtype=float))

        if not values:
            return np.zeros_like(t_arr)

        reduced_values: list[NDArray[np.float64]] = [values[0]]
        reduced_ops: list[str] = []
        for op, rhs in zip(self.operators, values[1:], strict=True):
            if op in {"*", "/"}:
                lhs = reduced_values[-1]
                if op == "*":
                    reduced_values[-1] = lhs * rhs
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        out = np.full_like(lhs, 1e30, dtype=float)
                        np.divide(lhs, rhs, out=out, where=np.abs(rhs) > 1e-30)
                    reduced_values[-1] = out
            else:
                reduced_ops.append(op)
                reduced_values.append(rhs)

        result = reduced_values[0]
        for op, rhs in zip(reduced_ops, reduced_values[1:], strict=True):
            if op == "+":
                result = result + rhs
            else:
                result = result - rhs
        return result

    def _extract_component_kwargs(
        self,
        component: ComponentDefinition,
        mapping: dict[str, str],
        kwargs: dict[str, float],
    ) -> dict[str, float]:
        component_kwargs: dict[str, float] = {}
        for pname in component.param_names:
            unique_name = mapping[pname]
            if unique_name not in kwargs:
                raise KeyError(f"Missing composite parameter '{unique_name}'")
            component_kwargs[pname] = float(kwargs[unique_name])
        return component_kwargs

    def additive_component_indices(self) -> list[int]:
        """Return component indices that contribute in additive (+) form.

        This includes the first component and any component joined with a
        ``+`` operator. Components joined with ``-``, ``*``, or ``/`` are
        excluded because their visual contribution is not an additive area.
        """
        if not self.components:
            return []

        indices = [0]
        for idx, op in enumerate(self.operators, start=1):
            if op == "+":
                indices.append(idx)
        return indices

    def evaluate_components(
        self,
        t: NDArray,
        *,
        additive_only: bool = False,
        **kwargs: float,
    ) -> list[tuple[str, NDArray[np.float64]]]:
        """Evaluate individual component curves.

        Parameters
        ----------
        t : array-like
            Time points where components are evaluated.
        additive_only : bool, optional
            If True, only return additive components (first component and
            components joined with ``+`` operators).
        **kwargs : float
            Composite-model parameters using unique parameter names.
        """
        t_arr = np.asarray(t, dtype=float)
        curves: list[tuple[str, NDArray[np.float64]]] = []

        if additive_only:
            include = set(self.additive_component_indices())
        else:
            include = set(range(len(self.components)))

        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True)
        ):
            if idx not in include:
                continue
            component_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            y_vals = np.asarray(component.function(t_arr, **component_kwargs), dtype=float)
            curves.append((self.component_names[idx], y_vals))
        return curves

    def formula_string(self) -> str:
        """Return a symbolic formula preview string."""
        parts: list[str] = []
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            fmt_values = {pname: mapping[pname] for pname in component.param_names}
            parts.append(component.formula_template.format(**fmt_values))

        if not parts:
            return ""
        expression = parts[0]
        for op, term in zip(self.operators, parts[1:], strict=True):
            expression = f"{expression} {op} {term}"
        return expression

    def to_model_definition(self, name: str = "Composite") -> ModelDefinition:
        """Create a ModelDefinition-compatible wrapper for the fit engine."""
        return ModelDefinition(
            name=name,
            description=self.formula_string(),
            function=self.function,
            param_names=list(self.param_names),
            param_defaults=dict(self.param_defaults),
            param_info=dict(self.param_info),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the model."""
        return {
            "component_names": list(self.component_names),
            "operators": list(self.operators),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompositeModel:
        """Construct a CompositeModel from serialized data."""
        component_names = data.get("component_names")
        operators = data.get("operators")
        if not isinstance(component_names, list) or not all(
            isinstance(v, str) for v in component_names
        ):
            raise ValueError("Invalid composite model data: component_names")
        if operators is not None:
            if not isinstance(operators, list) or not all(isinstance(v, str) for v in operators):
                raise ValueError("Invalid composite model data: operators")
        return cls(component_names=component_names, operators=operators)

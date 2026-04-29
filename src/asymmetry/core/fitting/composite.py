"""Composite fit-function builder primitives.

This module exposes baseline-free muSR components that can be combined with
``+``, ``-``, ``*``, and ``/`` to produce a single model callable compatible
with :class:`asymmetry.core.fitting.engine.FitEngine`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.models import (
    ModelDefinition,
    exponential_relaxation,
    gaussian_relaxation,
    static_gkt_zf,
    stretched_exponential,
)
from asymmetry.core.fitting.muon_fluorine.polarization import (
    general_fmuf_polarization,
    linear_fmuf_polarization,
    mu_f_polarization,
)
from asymmetry.core.fitting.parameters import ParamInfo, get_param_info
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


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
    latex_equation: str = ""
    category: str = "General"


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


def _oscillatory_field_component(
    t: NDArray,
    A: float,
    field: float,
    phase: float,
) -> NDArray[np.float64]:
    frequency = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * float(field)
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


def _muf_component(t: NDArray, A: float, r_muF: float) -> NDArray[np.float64]:
    return A * mu_f_polarization(t, r_muF)


def _linear_fmuf_component(t: NDArray, A: float, r_muF: float) -> NDArray[np.float64]:
    return A * linear_fmuf_polarization(t, r_muF)


def _general_fmuf_component(
    t: NDArray,
    A: float,
    r1: float,
    r2: float,
    theta: float,
) -> NDArray[np.float64]:
    try:
        return A * general_fmuf_polarization(t, r1, r2, theta)
    except ValueError:
        # Keep minimization alive for transient invalid trial points.
        return np.full_like(np.asarray(t, dtype=float), fill_value=1.0e3, dtype=float)


COMPONENTS: dict[str, ComponentDefinition] = {
    "Exponential": ComponentDefinition(
        name="Exponential",
        description="A exp(-Lambda t)",
        function=_exp_component,
        param_names=["A", "Lambda"],
        param_defaults={"A": 25.0, "Lambda": 0.5},
        param_info={"A": get_param_info("A"), "Lambda": get_param_info("Lambda")},
        formula_template="{A}*exp(-{Lambda}*t)",
        latex_equation=r"A(t) = A e^{-\Lambda t}",
    ),
    "Gaussian": ComponentDefinition(
        name="Gaussian",
        description="A exp(-(sigma t)^2)",
        function=_gaussian_component,
        param_names=["A", "sigma"],
        param_defaults={"A": 25.0, "sigma": 0.5},
        param_info={"A": get_param_info("A"), "sigma": get_param_info("sigma")},
        formula_template="{A}*exp(-({sigma}*t)^2)",
        latex_equation=r"A(t) = A e^{-(\sigma t)^2}",
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
        latex_equation=r"A(t) = A \cos(2\pi f t + \phi)",
    ),
    "OscillatoryField": ComponentDefinition(
        name="OscillatoryField",
        description="A cos(2 pi gamma_mu B t + phase)",
        function=_oscillatory_field_component,
        param_names=["A", "field", "phase"],
        param_defaults={"A": 25.0, "field": 100.0, "phase": 0.0},
        param_info={
            "A": get_param_info("A"),
            "field": get_param_info("field"),
            "phase": get_param_info("phase"),
        },
        formula_template="{A}*cos(2*pi*gamma_mu*{field}*t + {phase})",
        latex_equation=r"A(t) = A \cos(2\pi \gamma_\mu B t + \phi)",
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
        latex_equation=r"A(t) = A \exp\left(-(\lvert \Lambda \rvert t)^\beta\right)",
    ),
    "StaticGKT_ZF": ComponentDefinition(
        name="StaticGKT_ZF",
        description="Static Gaussian Kubo-Toyabe (zero field)",
        function=_gkt_component,
        param_names=["A", "Delta"],
        param_defaults={"A": 25.0, "Delta": 0.5},
        param_info={"A": get_param_info("A"), "Delta": get_param_info("Delta")},
        formula_template=("{A}*(1/3 + 2/3*(1-({Delta}*t)^2)*exp(-({Delta}*t)^2/2))"),
        latex_equation=(
            r"A(t) = A\left[\frac{1}{3} + \frac{2}{3}\left(1-(\Delta t)^2\right)e^{-(\Delta t)^2/2}\right]"
        ),
    ),
    "MuF": ComponentDefinition(
        name="MuF",
        description="Analytical mu-F polarization function D_z(t)",
        function=_muf_component,
        param_names=["A", "r_muF"],
        param_defaults={"A": 25.0, "r_muF": 1.17},
        param_info={"A": get_param_info("A"), "r_muF": get_param_info("r_muF")},
        formula_template="{A}*Dz_muF(t,{r_muF})",
        latex_equation=(
            r"A(t)=A\frac{1}{6}\left[1+2\cos\left(\frac{\omega_d t}{2}\right)+\cos(\omega_d t)+2\cos\left(\frac{3\omega_d t}{2}\right)\right]"
        ),
        category="Muon-Fluorine",
    ),
    "FmuF_Linear": ComponentDefinition(
        name="FmuF_Linear",
        description="Analytical collinear F-mu-F polarization function",
        function=_linear_fmuf_component,
        param_names=["A", "r_muF"],
        param_defaults={"A": 25.0, "r_muF": 1.17},
        param_info={"A": get_param_info("A"), "r_muF": get_param_info("r_muF")},
        formula_template="{A}*G_FmuF_linear(t,{r_muF})",
        latex_equation=(r"A(t)=A\,G_{F\mu F}(t)"),
        category="Muon-Fluorine",
    ),
    "FmuF_General": ComponentDefinition(
        name="FmuF_General",
        description="Numerical powder-averaged F-mu-F polarization (r1, r2, theta)",
        function=_general_fmuf_component,
        param_names=["A", "r1", "r2", "theta"],
        param_defaults={"A": 25.0, "r1": 1.17, "r2": 1.17, "theta": 180.0},
        param_info={
            "A": get_param_info("A"),
            "r1": get_param_info("r1"),
            "r2": get_param_info("r2"),
            "theta": get_param_info("theta"),
        },
        formula_template="{A}*Dz_FmuF_general(t,{r1},{r2},{theta})",
        latex_equation=(r"A(t)=A\,D_z^{\mathrm{powder}}\!(t;r_1,r_2,\theta)"),
        category="Muon-Fluorine",
    ),
    "Constant": ComponentDefinition(
        name="Constant",
        description="Constant background A_bg",
        function=_constant_component,
        param_names=["A_bg"],
        param_defaults={"A_bg": 0.0},
        param_info={"A_bg": get_param_info("A_bg")},
        formula_template="{A_bg}",
        latex_equation=r"A(t) = A_{bg}",
    ),
}


_ALLOWED_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})
_UNIT_AMPLITUDE_SENTINEL = "__UNIT_AMPLITUDE__"


class CompositeModel:
    """A flat composite model built from baseline-free components."""

    def __init__(
        self,
        component_names: list[str],
        operators: list[str] | None = None,
        open_parentheses: list[int] | None = None,
        close_parentheses: list[int] | None = None,
    ) -> None:
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

        if open_parentheses is None:
            open_parentheses = [0] * len(component_names)
        if close_parentheses is None:
            close_parentheses = [0] * len(component_names)
        if len(open_parentheses) != len(component_names):
            raise ValueError("open_parentheses length must be len(component_names)")
        if len(close_parentheses) != len(component_names):
            raise ValueError("close_parentheses length must be len(component_names)")
        if any((not isinstance(v, int)) or v < 0 for v in open_parentheses):
            raise ValueError("open_parentheses values must be non-negative integers")
        if any((not isinstance(v, int)) or v < 0 for v in close_parentheses):
            raise ValueError("close_parentheses values must be non-negative integers")

        balance = 0
        for open_count, close_count in zip(open_parentheses, close_parentheses, strict=True):
            balance += open_count
            balance -= close_count
            if balance < 0:
                raise ValueError("Invalid parentheses: closing before opening")
        if balance != 0:
            raise ValueError("Invalid parentheses: unbalanced expression")

        self.component_names = list(component_names)
        self.operators = list(operators)
        self.open_parentheses = list(open_parentheses)
        self.close_parentheses = list(close_parentheses)
        self.components = [COMPONENTS[name] for name in component_names]
        self._uses_parentheses = any(self.open_parentheses) or any(self.close_parentheses)
        # Keep legacy amplitude-sharing behavior for flat expressions.
        self._share_chain_amplitude = not self._uses_parentheses
        self._suppress_component_amplitude = self._identify_suppressed_amplitudes()
        self._param_mappings = self._build_param_mapping()

        param_names: list[str] = []
        defaults: dict[str, float] = {}
        param_info: dict[str, ParamInfo] = {}
        for mapping, component in zip(self._param_mappings, self.components, strict=True):
            for pname in component.param_names:
                unique_name = mapping[pname]
                if unique_name == _UNIT_AMPLITUDE_SENTINEL:
                    continue
                if unique_name not in defaults:
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
        amplitude_group_starts: list[int] = []
        current_start = 1
        for idx in range(1, len(self.components) + 1):
            if idx == 1:
                current_start = 1
            else:
                # Start a new amplitude group after additive operators.
                if self.operators[idx - 2] in {"+", "-"}:
                    current_start = idx
            amplitude_group_starts.append(current_start)

        for idx, component in enumerate(self.components, start=1):
            mapping: dict[str, str] = {}
            for pname in component.param_names:
                if (
                    self._is_scaling_parameter(pname)
                    and self._suppress_component_amplitude[idx - 1]
                ):
                    mapping[pname] = _UNIT_AMPLITUDE_SENTINEL
                    continue
                if pname == "A" and self._share_chain_amplitude:
                    # Share one amplitude within each multiplicative/divisive chain.
                    mapping[pname] = f"{pname}_{amplitude_group_starts[idx - 1]}"
                elif name_counts[pname] > 1:
                    mapping[pname] = f"{pname}_{idx}"
                else:
                    mapping[pname] = pname
            mappings.append(mapping)
        return mappings

    def _identify_suppressed_amplitudes(self) -> list[bool]:
        """Return flags for components whose amplitude should be fixed to unity.

        For parenthesized expressions, suppress amplitude on components that are
        multiplied/divided by an additive grouped expression, e.g. ``a*(b+c)``.
        """
        suppress = [False] * len(self.components)
        if not self._uses_parentheses:
            return suppress

        for op_index, op in enumerate(self.operators):
            if op not in {"*", "/"}:
                continue

            lhs_index = op_index
            rhs_index = op_index + 1
            if rhs_index >= len(self.components):
                continue

            rhs_is_additive_group = self.open_parentheses[
                rhs_index
            ] > 0 and self._rhs_group_contains_additive_operator(rhs_index)
            lhs_is_additive_group = self.close_parentheses[
                lhs_index
            ] > 0 and self._lhs_group_contains_additive_operator(lhs_index)

            if rhs_is_additive_group and not lhs_is_additive_group:
                if self._component_has_scaling_parameter(lhs_index):
                    suppress[lhs_index] = True
            if lhs_is_additive_group and not rhs_is_additive_group:
                if self._component_has_scaling_parameter(rhs_index):
                    suppress[rhs_index] = True

        return suppress

    def _rhs_group_contains_additive_operator(self, rhs_index: int) -> bool:
        """Return True if a grouped RHS expression includes top-level + or -."""
        # Track the first newly-opened parenthesis at rhs_index.
        balance = 1
        for k in range(rhs_index, len(self.components)):
            if k == rhs_index:
                balance += max(self.open_parentheses[k] - 1, 0)
            else:
                balance += self.open_parentheses[k]

            if k > rhs_index and balance > 0 and self.operators[k - 1] in {"+", "-"}:
                return True

            balance -= self.close_parentheses[k]
            if balance <= 0:
                break
        return False

    def _is_scaling_parameter(self, pname: str) -> bool:
        """Return True for parameters that act as component scale factors."""
        return pname in {"A", "A_bg"}

    def _component_has_scaling_parameter(self, idx: int) -> bool:
        return any(self._is_scaling_parameter(pname) for pname in self.components[idx].param_names)

    def _lhs_group_contains_additive_operator(self, lhs_index: int) -> bool:
        """Return True if a grouped LHS expression includes top-level + or -."""
        # Track the first newly-closed parenthesis at lhs_index.
        balance = 1
        for k in range(lhs_index, -1, -1):
            if k == lhs_index:
                balance += max(self.close_parentheses[k] - 1, 0)
            else:
                balance += self.close_parentheses[k]

            if k < lhs_index and balance > 0 and self.operators[k] in {"+", "-"}:
                return True

            balance -= self.open_parentheses[k]
            if balance <= 0:
                break
        return False

    def function(self, t: NDArray, **kwargs: float) -> NDArray[np.float64]:
        """Evaluate the composite function with standard arithmetic precedence."""
        t_arr = np.asarray(t, dtype=float)

        if self._uses_parentheses:
            return self._evaluate_parenthesized(t_arr, kwargs)

        values: list[NDArray[np.float64]] = []
        amplitudes: list[str | None] = []
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            component_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            amp_name = None
            if "A" in component.param_names:
                amp_name = mapping["A"]
                # Apply composite amplitude once per multiplicative group.
                component_kwargs["A"] = 1.0
            values.append(np.asarray(component.function(t_arr, **component_kwargs), dtype=float))
            amplitudes.append(amp_name)

        if not values:
            return np.zeros_like(t_arr)

        reduced_values: list[NDArray[np.float64]] = [values[0]]
        reduced_amplitudes: list[str | None] = [amplitudes[0]]
        reduced_ops: list[str] = []
        for op, rhs, rhs_amp in zip(self.operators, values[1:], amplitudes[1:], strict=True):
            if op in {"*", "/"}:
                lhs = reduced_values[-1]
                if op == "*":
                    reduced_values[-1] = lhs * rhs
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        out = np.full_like(lhs, 1e30, dtype=float)
                        np.divide(lhs, rhs, out=out, where=np.abs(rhs) > 1e-30)
                    reduced_values[-1] = out

                lhs_amp = reduced_amplitudes[-1]
                if lhs_amp is None:
                    reduced_amplitudes[-1] = rhs_amp
                elif rhs_amp is not None and rhs_amp != lhs_amp:
                    raise ValueError("Inconsistent amplitude mapping in multiplicative chain")
            else:
                reduced_ops.append(op)
                reduced_values.append(rhs)
                reduced_amplitudes.append(rhs_amp)

        weighted_values: list[NDArray[np.float64]] = []
        for amp_name, value in zip(reduced_amplitudes, reduced_values, strict=True):
            if amp_name is None:
                weighted_values.append(value)
            else:
                weighted_values.append(float(kwargs[amp_name]) * value)

        result = weighted_values[0]
        for op, rhs in zip(reduced_ops, weighted_values[1:], strict=True):
            if op == "+":
                result = result + rhs
            else:
                result = result - rhs
        return result

    def _evaluate_parenthesized(
        self,
        t_arr: NDArray[np.float64],
        kwargs: dict[str, float],
    ) -> NDArray[np.float64]:
        values: list[NDArray[np.float64]] = []
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            component_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            values.append(np.asarray(component.function(t_arr, **component_kwargs), dtype=float))

        if not values:
            return np.zeros_like(t_arr)

        value_stack: list[NDArray[np.float64]] = []
        op_stack: list[str] = []

        def precedence(op: str) -> int:
            return 2 if op in {"*", "/"} else 1

        def apply_top_operator() -> None:
            if len(value_stack) < 2 or not op_stack:
                raise ValueError("Invalid expression")
            op = op_stack.pop()
            rhs = value_stack.pop()
            lhs = value_stack.pop()
            if op == "+":
                value_stack.append(lhs + rhs)
            elif op == "-":
                value_stack.append(lhs - rhs)
            elif op == "*":
                value_stack.append(lhs * rhs)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out = np.full_like(lhs, 1e30, dtype=float)
                    np.divide(lhs, rhs, out=out, where=np.abs(rhs) > 1e-30)
                value_stack.append(out)

        for idx, value in enumerate(values):
            for _ in range(self.open_parentheses[idx]):
                op_stack.append("(")

            value_stack.append(value)

            for _ in range(self.close_parentheses[idx]):
                while op_stack and op_stack[-1] != "(":
                    apply_top_operator()
                if not op_stack or op_stack[-1] != "(":
                    raise ValueError("Invalid parentheses in expression")
                op_stack.pop()

            if idx < len(self.operators):
                op = self.operators[idx]
                while (
                    op_stack and op_stack[-1] != "(" and precedence(op_stack[-1]) >= precedence(op)
                ):
                    apply_top_operator()
                op_stack.append(op)

        while op_stack:
            if op_stack[-1] == "(":
                raise ValueError("Invalid parentheses in expression")
            apply_top_operator()

        if len(value_stack) != 1:
            raise ValueError("Invalid expression")
        return value_stack[0]

    def _extract_component_kwargs(
        self,
        component: ComponentDefinition,
        mapping: dict[str, str],
        kwargs: dict[str, float],
    ) -> dict[str, float]:
        component_kwargs: dict[str, float] = {}
        for pname in component.param_names:
            unique_name = mapping[pname]
            if unique_name == _UNIT_AMPLITUDE_SENTINEL:
                component_kwargs[pname] = 1.0
                continue
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
        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True), start=1
        ):
            fmt_values = {
                pname: ("1" if mapping[pname] == _UNIT_AMPLITUDE_SENTINEL else mapping[pname])
                for pname in component.param_names
            }
            if (
                self._share_chain_amplitude
                and "A" in fmt_values
                and idx > 1
                and self.operators[idx - 2] in {"*", "/"}
            ):
                fmt_values["A"] = "1"
            term = component.formula_template.format(**fmt_values)
            if fmt_values.get("A") == "1" and term.startswith("1*"):
                term = term[2:]

            if self.open_parentheses[idx - 1] > 0:
                term = "(" * self.open_parentheses[idx - 1] + term
            if self.close_parentheses[idx - 1] > 0:
                term = term + ")" * self.close_parentheses[idx - 1]
            parts.append(term)

        if not parts:
            return ""
        expression = parts[0]
        for op, term in zip(self.operators, parts[1:], strict=True):
            if op == "*" and term == "1":
                continue
            if op == "/" and term == "1":
                continue
            if op == "*" and expression == "1":
                expression = term
                continue
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
            "open_parentheses": list(self.open_parentheses),
            "close_parentheses": list(self.close_parentheses),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompositeModel:
        """Construct a CompositeModel from serialized data."""
        component_names = data.get("component_names")
        operators = data.get("operators")
        open_parentheses = data.get("open_parentheses")
        close_parentheses = data.get("close_parentheses")
        if not isinstance(component_names, list) or not all(
            isinstance(v, str) for v in component_names
        ):
            raise ValueError("Invalid composite model data: component_names")
        if operators is not None:
            if not isinstance(operators, list) or not all(isinstance(v, str) for v in operators):
                raise ValueError("Invalid composite model data: operators")
        if open_parentheses is not None:
            if not isinstance(open_parentheses, list) or not all(
                isinstance(v, int) for v in open_parentheses
            ):
                raise ValueError("Invalid composite model data: open_parentheses")
        if close_parentheses is not None:
            if not isinstance(close_parentheses, list) or not all(
                isinstance(v, int) for v in close_parentheses
            ):
                raise ValueError("Invalid composite model data: close_parentheses")
        return cls(
            component_names=component_names,
            operators=operators,
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
        )

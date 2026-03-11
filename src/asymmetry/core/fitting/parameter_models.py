"""Model fitting for fitted parameters as a function of field or temperature."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.parameters import Parameter, ParameterSet


@dataclass(frozen=True)
class ParameterModelComponentDefinition:
    """Descriptor for a parameter-vs-x basis function."""

    name: str
    description: str
    function: Callable[..., NDArray[np.float64]]
    param_names: list[str]
    param_defaults: dict[str, float]
    formula_template: str
    scopes: tuple[str, ...] = ("common",)


def _constant(x: NDArray, c: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(x, dtype=float), float(c), dtype=float)


def _linear(x: NDArray, m: float, b: float) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    return m * xx + b


def _power_law(x: NDArray, a: float, n: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    safe_x = np.maximum(np.abs(xx), 1e-12)
    return a * np.power(safe_x, n) + c


def _exp_decay(x: NDArray, a: float, tau: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    tau_safe = np.sign(tau) * max(abs(tau), 1e-12)
    exponent = np.clip(-xx / tau_safe, -700, 700)
    return a * np.exp(exponent) + c


def _arrhenius(x: NDArray, a: float, Ea: float) -> NDArray[np.float64]:
    # x is T (K), k_B in meV/K to keep Ea in meV-scale units.
    kb_mev = 8.617333262e-2
    tt = np.asarray(x, dtype=float)
    safe_t = np.maximum(np.abs(tt), 1e-9)
    exponent = np.clip(-Ea / (kb_mev * safe_t), -700, 700)
    return a * np.exp(exponent)


def _critical_divergence(x: NDArray, a: float, Tc: float, nu: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    dist = np.maximum(np.abs(xx - Tc), 1e-9)
    return a * np.power(dist, -nu) + c


def _redfield(x: NDArray, a: float, B0: float, c: float = 0.0) -> NDArray[np.float64]:
    xx = np.asarray(x, dtype=float)
    B0_safe = np.sign(B0) * max(abs(B0), 1e-12)
    return a / (1.0 + (xx / B0_safe) ** 2) + c


def _lorentzian(x: NDArray, a: float, B0: float, c: float = 0.0) -> NDArray[np.float64]:
    return _redfield(x, a, B0, c)


PARAMETER_MODEL_COMPONENTS: dict[str, ParameterModelComponentDefinition] = {
    "Constant": ParameterModelComponentDefinition(
        name="Constant",
        description="c",
        function=_constant,
        param_names=["c"],
        param_defaults={"c": 0.0},
        formula_template="{c}",
        scopes=("common",),
    ),
    "Linear": ParameterModelComponentDefinition(
        name="Linear",
        description="m*x + b",
        function=_linear,
        param_names=["m", "b"],
        param_defaults={"m": 1.0, "b": 0.0},
        formula_template="{m}*x + {b}",
        scopes=("common", "field", "temperature"),
    ),
    "PowerLaw": ParameterModelComponentDefinition(
        name="PowerLaw",
        description="a*|x|^n + c",
        function=_power_law,
        param_names=["a", "n", "c"],
        param_defaults={"a": 1.0, "n": 1.0, "c": 0.0},
        formula_template="{a}*|x|^{n} + {c}",
        scopes=("common",),
    ),
    "ExponentialDecay": ParameterModelComponentDefinition(
        name="ExponentialDecay",
        description="a*exp(-x/tau) + c",
        function=_exp_decay,
        param_names=["a", "tau", "c"],
        param_defaults={"a": 1.0, "tau": 10.0, "c": 0.0},
        formula_template="{a}*exp(-x/{tau}) + {c}",
        scopes=("common",),
    ),
    "Arrhenius": ParameterModelComponentDefinition(
        name="Arrhenius",
        description="a*exp(-Ea/(k_B T))",
        function=_arrhenius,
        param_names=["a", "Ea"],
        param_defaults={"a": 1.0, "Ea": 1.0},
        formula_template="{a}*exp(-{Ea}/(k_B*T))",
        scopes=("temperature",),
    ),
    "CriticalDivergence": ParameterModelComponentDefinition(
        name="CriticalDivergence",
        description="a*|T-Tc|^{-nu} + c",
        function=_critical_divergence,
        param_names=["a", "Tc", "nu", "c"],
        param_defaults={"a": 1.0, "Tc": 10.0, "nu": 1.0, "c": 0.0},
        formula_template="{a}*|x-{Tc}|^(-{nu}) + {c}",
        scopes=("temperature",),
    ),
    "Redfield": ParameterModelComponentDefinition(
        name="Redfield",
        description="a/(1 + (B/B0)^2) + c",
        function=_redfield,
        param_names=["a", "B0", "c"],
        param_defaults={"a": 1.0, "B0": 100.0, "c": 0.0},
        formula_template="{a}/(1 + (x/{B0})^2) + {c}",
        scopes=("field",),
    ),
    "Lorentzian": ParameterModelComponentDefinition(
        name="Lorentzian",
        description="a/(1 + (B/B0)^2) + c",
        function=_lorentzian,
        param_names=["a", "B0", "c"],
        param_defaults={"a": 1.0, "B0": 100.0, "c": 0.0},
        formula_template="{a}/(1 + (x/{B0})^2) + {c}",
        scopes=("field",),
    ),
}

_ALLOWED_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})


def component_names_for_x(x_key: str) -> list[str]:
    """Return available basis functions for a given x-variable."""
    if x_key == "field":
        scopes = {"common", "field"}
    elif x_key == "temperature":
        scopes = {"common", "temperature"}
    else:
        scopes = {"common"}

    names = [
        name
        for name, comp in PARAMETER_MODEL_COMPONENTS.items()
        if scopes.intersection(comp.scopes)
    ]
    return sorted(names)


class ParameterCompositeModel:
    """Flat composite model for parameter-vs-x fitting."""

    def __init__(self, component_names: list[str], operators: list[str] | None = None) -> None:
        if not component_names:
            raise ValueError("Model must contain at least one component")

        missing = [name for name in component_names if name not in PARAMETER_MODEL_COMPONENTS]
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
        self.components = [PARAMETER_MODEL_COMPONENTS[name] for name in self.component_names]
        self._param_mappings = self._build_param_mapping()

        param_names: list[str] = []
        param_defaults: dict[str, float] = {}
        for mapping, component in zip(self._param_mappings, self.components, strict=True):
            for pname in component.param_names:
                unique_name = mapping[pname]
                param_names.append(unique_name)
                param_defaults[unique_name] = component.param_defaults[pname]

        self.param_names = param_names
        self.param_defaults = param_defaults

    def _build_param_mapping(self) -> list[dict[str, str]]:
        counts = Counter(
            pname
            for component in self.components
            for pname in component.param_names
        )
        mappings: list[dict[str, str]] = []
        for idx, component in enumerate(self.components, start=1):
            mapping: dict[str, str] = {}
            for pname in component.param_names:
                if counts[pname] > 1:
                    mapping[pname] = f"{pname}_{idx}"
                else:
                    mapping[pname] = pname
            mappings.append(mapping)
        return mappings

    def formula_string(self) -> str:
        """Return a readable formula string for display."""
        terms: list[str] = []
        for mapping, component in zip(self._param_mappings, self.components, strict=True):
            mapping_text = {k: mapping[k] for k in component.param_names}
            terms.append(component.formula_template.format(**mapping_text))

        if not terms:
            return "0"

        expression = terms[0]
        for op, term in zip(self.operators, terms[1:], strict=True):
            expression = f"({expression}) {op} ({term})"
        return expression

    def function(self, x: NDArray, **kwargs: float) -> NDArray[np.float64]:
        """Evaluate the composite model with arithmetic precedence."""
        xx = np.asarray(x, dtype=float)
        values: list[NDArray[np.float64]] = []
        for component, mapping in zip(self.components, self._param_mappings, strict=True):
            local_kwargs: dict[str, float] = {}
            for pname in component.param_names:
                unique_name = mapping[pname]
                if unique_name not in kwargs:
                    raise KeyError(f"Missing parameter '{unique_name}'")
                local_kwargs[pname] = float(kwargs[unique_name])
            values.append(np.asarray(component.function(xx, **local_kwargs), dtype=float))

        if not values:
            return np.zeros_like(xx)

        reduced_values: list[NDArray[np.float64]] = [values[0]]
        reduced_ops: list[str] = []
        i = 0
        while i < len(self.operators):
            op = self.operators[i]
            if op in {"*", "/"}:
                left = reduced_values.pop()
                right = values[i + 1]
                if op == "*":
                    reduced_values.append(left * right)
                else:
                    safe_right = np.where(np.abs(right) < 1e-12, np.nan, right)
                    reduced_values.append(left / safe_right)
            else:
                reduced_ops.append(op)
                reduced_values.append(values[i + 1])
            i += 1

        result = reduced_values[0]
        for op, value in zip(reduced_ops, reduced_values[1:], strict=True):
            if op == "+":
                result = result + value
            else:
                result = result - value

        return result


@dataclass
class ParameterModelFitResult:
    """Result of fitting a parameter-vs-x model."""

    success: bool
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    parameters: ParameterSet = field(default_factory=ParameterSet)
    uncertainties: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class ModelFitRange:
    """A model and fit results over a specific x-range."""

    x_min: float | None
    x_max: float | None
    model: ParameterCompositeModel
    parameters: ParameterSet
    result: ParameterModelFitResult | None = None


@dataclass
class ParameterModelFit:
    """Model fits attached to a single parameter trace."""

    parameter_name: str
    x_key: str
    ranges: list[ModelFitRange] = field(default_factory=list)
    active: bool = True


@dataclass
class ParameterModelFitExecution:
    """Fitted model with sampled curve ready for plotting/export."""

    range_index: int
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    result: ParameterModelFitResult


def fit_parameter_model(
    x: NDArray,
    y: NDArray,
    yerr: NDArray | None,
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    x_min: float | None = None,
    x_max: float | None = None,
    method: str = "migrad",
) -> ParameterModelFitResult:
    """Fit a parameter-vs-x model using iminuit."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yerr is None:
        ee = np.ones_like(xx)
    else:
        ee = np.asarray(yerr, dtype=float)

    mask = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee) & (ee > 0)
    if x_min is not None:
        mask &= xx >= float(x_min)
    if x_max is not None:
        mask &= xx <= float(x_max)

    if not np.any(mask):
        return ParameterModelFitResult(success=False, message="No valid points in selected range")

    x_fit = xx[mask]
    y_fit = yy[mask]
    e_fit = ee[mask]

    try:
        from iminuit import Minuit
        from iminuit.cost import LeastSquares
    except ImportError as exc:
        return ParameterModelFitResult(success=False, message=f"iminuit import error: {exc}")

    free = parameters.free_parameters
    fixed_kw = {p.name: p.value for p in parameters if p.fixed}
    param_names = [p.name for p in free]

    def model_wrapper(x_local: NDArray, *args: float) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(param_names, args, strict=False))}
        return model.function(x_local, **kw)

    cost = LeastSquares(x_fit, y_fit, e_fit, model_wrapper)
    init = [p.value for p in free]
    m = Minuit(cost, *init, name=param_names)

    for i, p in enumerate(free):
        if p.min != -float("inf"):
            m.limits[i] = (p.min, m.limits[i][1])
        if p.max != float("inf"):
            m.limits[i] = (m.limits[i][0], p.max)

    if method == "simplex":
        m.simplex()
    else:
        m.migrad()

    result_params = ParameterSet()
    uncertainties: dict[str, float] = {}

    for p in parameters:
        if p.fixed:
            result_params.add(Parameter(name=p.name, value=p.value, min=p.min, max=p.max, fixed=True))
        else:
            idx = param_names.index(p.name)
            value = float(m.values[idx])
            result_params.add(Parameter(name=p.name, value=value, min=p.min, max=p.max, fixed=False))
            err = m.errors[idx]
            if err is not None and np.isfinite(err):
                uncertainties[p.name] = float(err)

    ndof = max(len(x_fit) - len(free), 1)
    return ParameterModelFitResult(
        success=bool(m.valid),
        chi_squared=float(m.fval),
        reduced_chi_squared=float(m.fval) / ndof,
        parameters=result_params,
        uncertainties=uncertainties,
        message="Fit successful" if m.valid else "Fit failed",
    )


def evaluate_parameter_model_fit(
    fit: ParameterModelFit,
    num_points: int = 200,
) -> list[ParameterModelFitExecution]:
    """Return sampled curves for each successful active range."""
    curves: list[ParameterModelFitExecution] = []

    for idx, fit_range in enumerate(fit.ranges):
        if fit_range.result is None or not fit_range.result.success:
            continue

        x_min = fit_range.x_min
        x_max = fit_range.x_max
        if x_min is None or x_max is None:
            continue
        if x_max <= x_min:
            continue

        xs = np.linspace(float(x_min), float(x_max), num=max(2, int(num_points)), dtype=float)
        kwargs = {p.name: p.value for p in fit_range.result.parameters}
        ys = fit_range.model.function(xs, **kwargs)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            continue

        curves.append(
            ParameterModelFitExecution(
                range_index=idx,
                x=xs[mask],
                y=ys[mask],
                result=fit_range.result,
            )
        )

    return curves

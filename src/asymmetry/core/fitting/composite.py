"""Composite fit-function builder primitives.

This module exposes baseline-free muSR components that can be combined with
``+``, ``-``, ``*``, and ``/`` to produce a single model callable compatible
with :class:`asymmetry.core.fitting.engine.FitEngine`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.models import (
    ModelDefinition,
    exponential_relaxation,
    gaussian_relaxation,
    longitudinal_field_kubo_toyabe,
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


def _lf_kt_component(t: NDArray, A: float, Delta: float, B_L: float) -> NDArray[np.float64]:
    """Longitudinal-field Kubo-Toyabe depolarization function.

    Wrapper adapting longitudinal_field_kubo_toyabe for use as a composite component.
    """
    return longitudinal_field_kubo_toyabe(t, A0=A, Delta=Delta, B_L=B_L, baseline=0.0)


def _constant_component(t: NDArray, A_bg: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(t, dtype=float), fill_value=A_bg, dtype=float)


def _gaussian_peak_component(
    t: NDArray,
    height: float,
    nu0: float,
    fwhm: float,
) -> NDArray[np.float64]:
    x = np.asarray(t, dtype=float)
    width = max(abs(float(fwhm)), 1e-12)
    exponent = -4.0 * np.log(2.0) * ((x - float(nu0)) / width) ** 2
    return float(height) * np.exp(exponent)


def _lorentzian_peak_component(
    t: NDArray,
    height: float,
    nu0: float,
    fwhm: float,
) -> NDArray[np.float64]:
    x = np.asarray(t, dtype=float)
    width = max(abs(float(fwhm)), 1e-12)
    return float(height) / (1.0 + 4.0 * ((x - float(nu0)) / width) ** 2)


def _constant_background_component(t: NDArray, bg: float) -> NDArray[np.float64]:
    return np.full_like(np.asarray(t, dtype=float), fill_value=float(bg), dtype=float)


def _linear_background_component(t: NDArray, bg: float, slope: float) -> NDArray[np.float64]:
    x = np.asarray(t, dtype=float)
    return float(bg) + float(slope) * x


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
    "LongitudinalFieldKT": ComponentDefinition(
        name="LongitudinalFieldKT",
        description="Static Gaussian Kubo-Toyabe with longitudinal field (Hayano et al. 1979)",
        function=_lf_kt_component,
        param_names=["A", "Delta", "B_L"],
        param_defaults={"A": 25.0, "Delta": 0.5, "B_L": 0.0},
        param_info={
            "A": get_param_info("A"),
            "Delta": get_param_info("Delta"),
            "B_L": get_param_info("B_L"),
        },
        formula_template="{A}*Gz(t; Delta={Delta}, B_L={B_L})",
        latex_equation=(
            r"A(t) = A\left[1 - \frac{2\Delta^2}{\omega_0^2}\left(1 - e^{-\Delta^2 t^2/2}\cos(\omega_0 t)\right) "
            r"+ \frac{2\Delta^4}{\omega_0^3}\int_0^t e^{-\Delta^2\tau^2/2}\sin(\omega_0\tau)\,d\tau\right] "
            r"\quad\text{where}\quad \omega_0 = \gamma_\mu B_L"
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
    "GaussianPeak": ComponentDefinition(
        name="GaussianPeak",
        description="Frequency-domain Gaussian peak",
        function=_gaussian_peak_component,
        param_names=["height", "nu0", "fwhm"],
        param_defaults={"height": 1.0, "nu0": 1.0, "fwhm": 0.1},
        param_info={
            "height": get_param_info("height"),
            "nu0": get_param_info("nu0"),
            "fwhm": get_param_info("fwhm"),
        },
        formula_template="{height}*exp(-4*ln(2)*((nu-{nu0})/{fwhm})^2)",
        latex_equation=r"S(\nu)=h\exp\left[-4\ln2\left((\nu-\nu_0)/w\right)^2\right]",
        category="Frequency Domain",
    ),
    "LorentzianPeak": ComponentDefinition(
        name="LorentzianPeak",
        description="Frequency-domain Lorentzian peak",
        function=_lorentzian_peak_component,
        param_names=["height", "nu0", "fwhm"],
        param_defaults={"height": 1.0, "nu0": 1.0, "fwhm": 0.1},
        param_info={
            "height": get_param_info("height"),
            "nu0": get_param_info("nu0"),
            "fwhm": get_param_info("fwhm"),
        },
        formula_template="{height}/(1+4*((nu-{nu0})/{fwhm})^2)",
        latex_equation=r"S(\nu)=h/[1+4((\nu-\nu_0)/w)^2]",
        category="Frequency Domain",
    ),
    "ConstantBackground": ComponentDefinition(
        name="ConstantBackground",
        description="Frequency-domain constant background",
        function=_constant_background_component,
        param_names=["bg"],
        param_defaults={"bg": 0.0},
        param_info={"bg": get_param_info("bg")},
        formula_template="{bg}",
        latex_equation=r"S(\nu)=b_g",
        category="Frequency Domain",
    ),
    "LinearBackground": ComponentDefinition(
        name="LinearBackground",
        description="Frequency-domain linear background",
        function=_linear_background_component,
        param_names=["bg", "slope"],
        param_defaults={"bg": 0.0, "slope": 0.0},
        param_info={"bg": get_param_info("bg"), "slope": get_param_info("slope")},
        formula_template="{bg}+{slope}*nu",
        latex_equation=r"S(\nu)=b_g+m\nu",
        category="Frequency Domain",
    ),
}


_ALLOWED_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})
_UNIT_AMPLITUDE_SENTINEL = "__UNIT_AMPLITUDE__"
_FRACTION_GROUP_DECORATOR = "frac"


def _tokenize_component_expression(expression: str) -> list[str]:
    """Return infix expression tokens for component-name expressions."""
    stripped = expression.strip()
    if not stripped:
        raise ValueError("Expression is required")

    token_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[(){}+\-*/]")
    tokens: list[str] = []
    position = 0
    for match in token_pattern.finditer(stripped):
        gap = stripped[position : match.start()]
        if gap.strip():
            raise ValueError(f"Unexpected token near '{gap.strip()}'")
        tokens.append(match.group(0))
        position = match.end()

    trailing = stripped[position:]
    if trailing.strip():
        raise ValueError(f"Unexpected token near '{trailing.strip()}'")
    return tokens


def _parse_group_decorator(tokens: list[str], idx: int) -> tuple[str | None, int]:
    """Return an optional group decorator starting at ``idx``."""
    if idx >= len(tokens) or tokens[idx] != "{":
        return None, idx
    if idx + 2 >= len(tokens) or tokens[idx + 2] != "}":
        raise ValueError("Invalid group decorator")
    decorator = tokens[idx + 1]
    if decorator != _FRACTION_GROUP_DECORATOR:
        raise ValueError(f"Unknown group decorator '{decorator}'")
    return decorator, idx + 3


def parse_component_expression(
    expression: str,
    *,
    allowed_components: set[str] | frozenset[str],
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Parse a component expression into constructor-ready parts."""
    tokens = _tokenize_component_expression(expression)

    component_names: list[str] = []
    operators: list[str] = []
    open_parentheses: list[int] = []
    close_parentheses: list[int] = []
    pending_open = 0
    expecting_operand = True

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if expecting_operand:
            if token == "(":
                pending_open += 1
                idx += 1
                continue
            if token in _ALLOWED_OPERATORS or token == ")":
                raise ValueError(f"Expected component before '{token}'")
            if token not in allowed_components:
                raise ValueError(f"Unknown component '{token}'")

            component_names.append(token)
            open_parentheses.append(pending_open)
            close_parentheses.append(0)
            pending_open = 0
            expecting_operand = False
            idx += 1
            continue

        if token in _ALLOWED_OPERATORS:
            operators.append(token)
            expecting_operand = True
            idx += 1
            continue
        if token == ")":
            if not component_names:
                raise ValueError("Closing parenthesis has no matching component")
            close_parentheses[-1] += 1
            idx += 1
            continue
        if token == "(":
            raise ValueError("Expected operator before '('")
        raise ValueError(f"Expected operator before '{token}'")

    if pending_open:
        raise ValueError("Invalid parentheses: unbalanced expression")
    if expecting_operand:
        raise ValueError("Expression cannot end with an operator")

    balance = 0
    for open_count, close_count in zip(open_parentheses, close_parentheses, strict=True):
        balance += open_count
        balance -= close_count
        if balance < 0:
            raise ValueError("Invalid parentheses: closing before opening")
    if balance != 0:
        raise ValueError("Invalid parentheses: unbalanced expression")

    return component_names, operators, open_parentheses, close_parentheses


def parse_composite_expression(
    expression: str,
) -> tuple[list[str], list[str], list[int], list[int], list[tuple[int, int]]]:
    """Parse a composite expression including optional group decorators."""
    tokens = _tokenize_component_expression(expression)

    component_names: list[str] = []
    operators: list[str] = []
    open_parentheses: list[int] = []
    close_parentheses: list[int] = []
    fraction_groups: list[tuple[int, int]] = []
    pending_open = 0
    expecting_operand = True
    paren_component_stack: list[int] = []
    last_closed_group: tuple[int, int] | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if expecting_operand:
            if token == "(":
                pending_open += 1
                idx += 1
                continue
            if token in _ALLOWED_OPERATORS or token in {")", "{", "}"}:
                raise ValueError(f"Expected component before '{token}'")
            if token not in COMPONENTS:
                raise ValueError(f"Unknown component '{token}'")

            component_index = len(component_names)
            component_names.append(token)
            open_parentheses.append(pending_open)
            close_parentheses.append(0)
            for _ in range(pending_open):
                paren_component_stack.append(component_index)
            pending_open = 0
            expecting_operand = False
            last_closed_group = None
            idx += 1
            continue

        if token in _ALLOWED_OPERATORS:
            operators.append(token)
            expecting_operand = True
            last_closed_group = None
            idx += 1
            continue
        if token == ")":
            if not component_names:
                raise ValueError("Closing parenthesis has no matching component")
            if not paren_component_stack:
                raise ValueError("Invalid parentheses: closing before opening")
            close_parentheses[-1] += 1
            start_index = paren_component_stack.pop()
            last_closed_group = (start_index, len(component_names) - 1)
            idx += 1
            decorator, idx = _parse_group_decorator(tokens, idx)
            if decorator == _FRACTION_GROUP_DECORATOR:
                fraction_groups.append(last_closed_group)
            continue
        if token == "(":
            raise ValueError("Expected operator before '('")
        if token == "{":
            raise ValueError("Group decorator must follow a closing parenthesis")
        raise ValueError(f"Expected operator before '{token}'")

    if pending_open:
        raise ValueError("Invalid parentheses: unbalanced expression")
    if expecting_operand:
        raise ValueError("Expression cannot end with an operator")

    balance = 0
    for open_count, close_count in zip(open_parentheses, close_parentheses, strict=True):
        balance += open_count
        balance -= close_count
        if balance < 0:
            raise ValueError("Invalid parentheses: closing before opening")
    if balance != 0:
        raise ValueError("Invalid parentheses: unbalanced expression")

    return component_names, operators, open_parentheses, close_parentheses, fraction_groups


def build_component_expression(
    component_names: list[str],
    operators: list[str],
    open_parentheses: list[int] | None = None,
    close_parentheses: list[int] | None = None,
    fraction_groups: list[tuple[int, int]] | None = None,
) -> str:
    """Return a human-editable expression string using component names."""
    if not component_names:
        return ""

    opens = list(open_parentheses or [0] * len(component_names))
    closes = list(close_parentheses or [0] * len(component_names))
    fraction_group_set = set(fraction_groups or [])
    parts: list[str] = []
    paren_component_stack: list[int] = []
    for idx, name in enumerate(component_names):
        prefix = "(" * opens[idx]
        for _ in range(opens[idx]):
            paren_component_stack.append(idx)

        suffix_parts: list[str] = []
        for _ in range(closes[idx]):
            if not paren_component_stack:
                raise ValueError("Invalid parentheses while building expression")
            start_index = paren_component_stack.pop()
            suffix_parts.append(
                ")" + ("{frac}" if (start_index, idx) in fraction_group_set else "")
            )
        token = prefix + name + "".join(suffix_parts)
        if idx == 0:
            parts.append(token)
        else:
            parts.append(f"{operators[idx - 1]} {token}")
    return " ".join(parts)


class CompositeModel:
    """A flat composite model built from baseline-free components."""

    def __init__(
        self,
        component_names: list[str],
        operators: list[str] | None = None,
        open_parentheses: list[int] | None = None,
        close_parentheses: list[int] | None = None,
        fraction_groups: list[tuple[int, int]] | None = None,
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
        self.fraction_groups = self._validate_fraction_groups(fraction_groups or [])
        self._fraction_param_number_by_component = self._build_fraction_param_number_map()
        self._fraction_term_number_by_component = self._build_fraction_term_number_map()
        self._fraction_group_by_component = self._build_fraction_group_component_map()
        self.components = [COMPONENTS[name] for name in component_names]
        self._uses_parentheses = any(self.open_parentheses) or any(self.close_parentheses)
        # Keep legacy amplitude-sharing behavior for flat expressions.
        self._share_chain_amplitude = not self._uses_parentheses
        self._suppress_component_amplitude = self._identify_suppressed_amplitudes()
        self._param_mappings = self._build_param_mapping()

        param_names: list[str] = []
        defaults: dict[str, float] = {}
        param_info: dict[str, ParamInfo] = {}
        for idx, (mapping, component) in enumerate(
            zip(self._param_mappings, self.components, strict=True)
        ):
            group = self._fraction_group_by_component.get(idx)
            if group is not None and group[0] == idx:
                amplitude_name = self._fraction_group_amplitude_name(group)
                param_names.append(amplitude_name)
                defaults[amplitude_name] = self._fraction_group_default_amplitude(group)
                param_info[amplitude_name] = get_param_info(amplitude_name)

            for pname in component.param_names:
                unique_name = mapping[pname]
                if unique_name == _UNIT_AMPLITUDE_SENTINEL:
                    continue
                if unique_name not in defaults:
                    param_names.append(unique_name)
                    defaults[unique_name] = component.param_defaults[pname]
                    param_info[unique_name] = get_param_info(unique_name)

            if group is not None and idx in self._fraction_group_term_starts(group):
                fraction_name = self._fraction_param_name(idx)
                param_names.append(fraction_name)
                defaults[fraction_name] = 1.0 / float(len(self._fraction_group_term_starts(group)))
                param_info[fraction_name] = get_param_info(fraction_name)
        self.param_names = param_names
        self.param_defaults = defaults
        self.param_info = param_info

    @classmethod
    def from_expression(cls, expression: str) -> CompositeModel:
        """Construct a CompositeModel from a component-name expression."""
        component_names, operators, open_parentheses, close_parentheses, fraction_groups = (
            parse_composite_expression(expression)
        )
        return cls(
            component_names=component_names,
            operators=operators,
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
            fraction_groups=fraction_groups,
        )

    def component_expression_string(self) -> str:
        """Return the builder-facing expression using component names."""
        return build_component_expression(
            self.component_names,
            self.operators,
            self.open_parentheses,
            self.close_parentheses,
            self.fraction_groups,
        )

    def _validate_fraction_groups(
        self,
        fraction_groups: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        actual_groups = set(self._parenthesized_group_ranges())
        validated: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        occupied_components: set[int] = set()
        for group in fraction_groups:
            if not isinstance(group, tuple) or len(group) != 2:
                raise ValueError("fraction_groups must contain (start, end) pairs")
            start, end = group
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("fraction_groups indices must be integers")
            if start < 0 or end >= len(self.component_names) or start >= end:
                raise ValueError("Invalid fraction group range")
            if group in seen:
                raise ValueError("Duplicate fraction group")
            seen.add(group)
            if group not in actual_groups:
                raise ValueError("Fraction groups must map to one parenthesized expression")
            term_ranges = self._fraction_group_term_ranges(group)
            if len(term_ranges) < 2:
                raise ValueError("Fraction groups require at least two additive terms")
            for idx in range(start, end + 1):
                if idx in occupied_components:
                    raise ValueError("Fraction groups cannot overlap")
                occupied_components.add(idx)
            validated.append(group)
        validated.sort()
        return validated

    def _parenthesized_group_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        stack: list[int] = []
        for idx in range(len(self.component_names)):
            for _ in range(self.open_parentheses[idx]):
                stack.append(idx)
            for _ in range(self.close_parentheses[idx]):
                if not stack:
                    raise ValueError("Invalid parentheses: closing before opening")
                ranges.append((stack.pop(), idx))
        if stack:
            raise ValueError("Invalid parentheses: unbalanced expression")
        return ranges

    def _build_fraction_group_component_map(self) -> dict[int, tuple[int, int]]:
        mapping: dict[int, tuple[int, int]] = {}
        for group in self.fraction_groups:
            start, end = group
            for idx in range(start, end + 1):
                mapping[idx] = group
        return mapping

    def _build_fraction_param_number_map(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        next_number = 1
        for group in self.fraction_groups:
            for idx in self._fraction_group_term_starts(group):
                mapping[idx] = next_number
                next_number += 1
        return mapping

    def _build_fraction_term_number_map(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        next_number = 1
        for group in self.fraction_groups:
            for term_start, term_end in self._fraction_group_term_ranges(group):
                for idx in range(term_start, term_end + 1):
                    mapping[idx] = next_number
                next_number += 1
        return mapping

    def _term_ranges(self, start: int, end: int, *, inside_group: bool) -> list[tuple[int, int]]:
        """Return top-level additive term ranges within one expression span."""
        if start < 0 or end >= len(self.component_names) or start > end:
            raise ValueError("Invalid term range")

        depth = int(self.open_parentheses[start]) - (1 if inside_group else 0)
        if depth < 0:
            raise ValueError("Invalid parentheses while parsing term ranges")

        term_ranges: list[tuple[int, int]] = []
        term_start = start
        for idx in range(start, end):
            depth_after = depth - int(self.close_parentheses[idx])
            if depth_after < 0:
                raise ValueError("Invalid parentheses while parsing term ranges")
            operator = self.operators[idx]
            if depth_after == 0 and operator in {"+", "-"}:
                if operator != "+":
                    raise ValueError("Fraction groups only support additive '+' terms")
                term_ranges.append((term_start, idx))
                term_start = idx + 1
            depth = depth_after + int(self.open_parentheses[idx + 1])

        depth -= int(self.close_parentheses[end]) - (1 if inside_group else 0)
        if depth != 0:
            raise ValueError("Invalid parentheses while parsing term ranges")
        term_ranges.append((term_start, end))
        return term_ranges

    def _fraction_group_term_ranges(self, group: tuple[int, int]) -> list[tuple[int, int]]:
        """Return the additive term ranges represented by one fraction group."""
        return self._term_ranges(group[0], group[1], inside_group=True)

    def _fraction_group_term_starts(self, group: tuple[int, int]) -> list[int]:
        """Return component indices where each weighted fraction term starts."""
        return [start for start, _end in self._fraction_group_term_ranges(group)]

    def _fraction_group_amplitude_name(self, group: tuple[int, int]) -> str:
        return f"A_{group[0] + 1}"

    def _fraction_param_name(self, component_index: int) -> str:
        number = self._fraction_param_number_by_component.get(component_index, component_index + 1)
        return f"fraction_{number}"

    def _fraction_group_default_amplitude(self, group: tuple[int, int]) -> float:
        start, end = group
        for idx in range(start, end + 1):
            component = self.components[idx]
            for pname in component.param_names:
                if self._is_scaling_parameter(pname):
                    return float(component.param_defaults[pname])
        return 1.0

    def _fraction_group_weights(
        self,
        group: tuple[int, int],
        kwargs: dict[str, float],
    ) -> dict[int, float]:
        component_indices = self._fraction_group_term_starts(group)
        raw_weights: list[float] = []
        for idx in component_indices:
            name = self._fraction_param_name(idx)
            if name not in kwargs:
                raise KeyError(f"Missing composite parameter '{name}'")
            raw_weights.append(max(float(kwargs[name]), 0.0))

        total = sum(raw_weights)
        if total <= 1e-30:
            normalized = [1.0 / float(len(raw_weights))] * len(raw_weights)
        else:
            normalized = [value / total for value in raw_weights]
        return dict(zip(component_indices, normalized, strict=True))

    def normalized_parameter_values(self, values: dict[str, float]) -> dict[str, float]:
        """Return a copy with fraction-group parameters normalized for display."""
        normalized = dict(values)
        for group in self.fraction_groups:
            try:
                group_weights = self._fraction_group_weights(group, normalized)
            except KeyError:
                continue
            for idx, weight in group_weights.items():
                normalized[self._fraction_param_name(idx)] = weight
        return normalized

    def fraction_parameter_groups(self) -> list[list[str]]:
        """Return fraction-parameter names grouped by additive fraction group."""
        groups: list[list[str]] = []
        for group in self.fraction_groups:
            groups.append(
                [self._fraction_param_name(idx) for idx in self._fraction_group_term_starts(group)]
            )
        return groups

    def with_default_fraction_groups(self) -> CompositeModel:
        """Return a copy with a top-level additive fraction group when suitable."""
        if self.fraction_groups or len(self.component_names) < 2:
            return self

        try:
            term_ranges = self._term_ranges(0, len(self.component_names) - 1, inside_group=False)
        except ValueError:
            return self
        if len(term_ranges) < 2:
            return self

        open_parentheses = list(self.open_parentheses)
        close_parentheses = list(self.close_parentheses)
        if (0, len(self.component_names) - 1) not in self._parenthesized_group_ranges():
            open_parentheses[0] += 1
            close_parentheses[-1] += 1

        return CompositeModel(
            component_names=list(self.component_names),
            operators=list(self.operators),
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
            fraction_groups=[*self.fraction_groups, (0, len(self.component_names) - 1)],
        )

    def _build_param_mapping(self) -> list[dict[str, str]]:
        name_counts = Counter(
            pname for component in self.components for pname in component.param_names
        )
        mappings: list[dict[str, str]] = []
        used_names: set[str] = set()
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
                if self._fraction_group_by_component.get(
                    idx - 1
                ) is not None and self._is_scaling_parameter(pname):
                    mapping[pname] = _UNIT_AMPLITUDE_SENTINEL
                    continue
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
                    term_number = self._fraction_term_number_by_component.get(idx - 1)
                    if term_number is not None:
                        candidate = f"{pname}_{term_number}"
                        mapping[pname] = (
                            candidate if candidate not in used_names else f"{pname}_{idx}"
                        )
                    else:
                        mapping[pname] = f"{pname}_{idx}"
                else:
                    mapping[pname] = pname
                if mapping[pname] != _UNIT_AMPLITUDE_SENTINEL:
                    used_names.add(mapping[pname])
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
        fraction_group_set = set(self.fraction_groups)
        fraction_weights: dict[int, float] = {}
        for group in self.fraction_groups:
            fraction_weights.update(self._fraction_group_weights(group, kwargs))

        values: list[NDArray[np.float64]] = []
        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True)
        ):
            component_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            value = np.asarray(component.function(t_arr, **component_kwargs), dtype=float)
            if idx in fraction_weights:
                value = fraction_weights[idx] * value
            values.append(value)

        if not values:
            return np.zeros_like(t_arr)

        value_stack: list[NDArray[np.float64]] = []
        op_stack: list[str] = []
        paren_start_stack: list[int] = []

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
                paren_start_stack.append(idx)

            value_stack.append(value)

            for _ in range(self.close_parentheses[idx]):
                while op_stack and op_stack[-1] != "(":
                    apply_top_operator()
                if not op_stack or op_stack[-1] != "(":
                    raise ValueError("Invalid parentheses in expression")
                op_stack.pop()
                if not paren_start_stack:
                    raise ValueError("Invalid parentheses in expression")
                group = (paren_start_stack.pop(), idx)
                if group in fraction_group_set:
                    amplitude_name = self._fraction_group_amplitude_name(group)
                    if amplitude_name not in kwargs:
                        raise KeyError(f"Missing composite parameter '{amplitude_name}'")
                    value_stack[-1] = float(kwargs[amplitude_name]) * value_stack[-1]

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

        fraction_weights: dict[int, float] = {}
        for group in self.fraction_groups:
            fraction_weights.update(self._fraction_group_weights(group, kwargs))

        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True)
        ):
            if idx not in include:
                continue
            component_kwargs = self._extract_component_kwargs(component, mapping, kwargs)
            y_vals = np.asarray(component.function(t_arr, **component_kwargs), dtype=float)
            group = self._fraction_group_by_component.get(idx)
            if group is not None:
                weight = fraction_weights[idx] if idx in fraction_weights else 1.0
                y_vals = float(kwargs[self._fraction_group_amplitude_name(group)]) * weight * y_vals
            curves.append((self.component_names[idx], y_vals))
        return curves

    def formula_string(self) -> str:
        """Return a symbolic formula preview string."""
        if self.fraction_groups:
            return self._formula_string_with_fraction_groups()

        parts: list[str] = []
        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True), start=1
        ):
            term = self._component_formula_term(idx - 1, component, mapping)

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

    def _component_formula_term(
        self,
        component_index: int,
        component: ComponentDefinition,
        mapping: dict[str, str],
    ) -> str:
        fmt_values = {
            pname: ("1" if mapping[pname] == _UNIT_AMPLITUDE_SENTINEL else mapping[pname])
            for pname in component.param_names
        }
        if (
            self._share_chain_amplitude
            and "A" in fmt_values
            and component_index > 0
            and self.operators[component_index - 1] in {"*", "/"}
        ):
            fmt_values["A"] = "1"
        term = component.formula_template.format(**fmt_values)
        if fmt_values.get("A") == "1" and term.startswith("1*"):
            term = term[2:]
        return term

    def _formula_string_with_fraction_groups(self) -> str:
        terms: list[str] = []
        fraction_term_starts = {
            idx for group in self.fraction_groups for idx in self._fraction_group_term_starts(group)
        }
        for idx, (component, mapping) in enumerate(
            zip(self.components, self._param_mappings, strict=True)
        ):
            term = self._component_formula_term(idx, component, mapping)
            if idx in fraction_term_starts:
                fraction_name = self._fraction_param_name(idx)
                term = fraction_name if term == "1" else f"{fraction_name}*{term}"
            terms.append(term)

        value_stack: list[str] = []
        op_stack: list[str] = []
        paren_start_stack: list[int] = []
        fraction_group_set = set(self.fraction_groups)

        def precedence(op: str) -> int:
            return 2 if op in {"*", "/"} else 1

        def apply_top_operator() -> None:
            if len(value_stack) < 2 or not op_stack:
                raise ValueError("Invalid expression")
            op = op_stack.pop()
            rhs = value_stack.pop()
            lhs = value_stack.pop()
            value_stack.append(f"({lhs} {op} {rhs})")

        for idx, term in enumerate(terms):
            for _ in range(self.open_parentheses[idx]):
                op_stack.append("(")
                paren_start_stack.append(idx)

            value_stack.append(term)

            for _ in range(self.close_parentheses[idx]):
                while op_stack and op_stack[-1] != "(":
                    apply_top_operator()
                if not op_stack or op_stack[-1] != "(":
                    raise ValueError("Invalid parentheses in expression")
                op_stack.pop()
                if not paren_start_stack:
                    raise ValueError("Invalid parentheses in expression")
                group = (paren_start_stack.pop(), idx)
                if group in fraction_group_set:
                    amplitude_name = self._fraction_group_amplitude_name(group)
                    grouped_term = value_stack.pop()
                    value_stack.append(f"{amplitude_name}*({grouped_term})")

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

        expression = value_stack[0]
        if expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1]
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
            "fraction_groups": [[start, end] for start, end in self.fraction_groups],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompositeModel:
        """Construct a CompositeModel from serialized data."""
        component_names = data.get("component_names")
        operators = data.get("operators")
        open_parentheses = data.get("open_parentheses")
        close_parentheses = data.get("close_parentheses")
        fraction_groups = data.get("fraction_groups")
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
        if fraction_groups is not None:
            if not isinstance(fraction_groups, list) or not all(
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(idx, int) for idx in value)
                for value in fraction_groups
            ):
                raise ValueError("Invalid composite model data: fraction_groups")
        return cls(
            component_names=component_names,
            operators=operators,
            open_parentheses=open_parentheses,
            close_parentheses=close_parentheses,
            fraction_groups=[(start, end) for start, end in (fraction_groups or [])],
        )

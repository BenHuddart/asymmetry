"""Parameter objects with bounds, constraints, and linking."""

from __future__ import annotations

from dataclasses import dataclass
import re


_INDEXED_PARAM_RE = re.compile(r"^(.+)_([0-9]+)$")
_GLE_CONTROL_WORD_RE = re.compile(r"^\\[A-Za-z]+$")


@dataclass(frozen=True)
class ParamInfo:
    """Display and unit metadata for a fit parameter."""

    name: str
    plain: str
    unicode: str
    latex: str
    gle: str
    unit: str | None = None
    default_min: float | None = None  # None means no lower bound (-inf)

    def with_index(self, index: str) -> ParamInfo:
        """Return indexed metadata (e.g. ``A`` -> ``A_2``)."""
        return ParamInfo(
            name=f"{self.name}_{index}",
            plain=f"{self.plain}_{index}",
            unicode=f"{self.unicode}_{index}",
            latex=_append_latex_index(self.latex, index),
            gle=f"{self.gle}_{{{index}}}",
            unit=self.unit,
            default_min=self.default_min,
        )

    def plain_label(self, *, include_unit: bool = True) -> str:
        if include_unit and self.unit:
            return f"{self.plain} ({self.unit})"
        return self.plain

    def unicode_label(self, *, include_unit: bool = True) -> str:
        if include_unit and self.unit:
            return f"{self.unicode} ({self.unit})"
        return self.unicode

    def latex_label(self, *, include_unit: bool = True) -> str:
        if include_unit and self.unit:
            return f"{self.latex} ({self.unit})"
        return self.latex

    def gle_label(self, *, include_unit: bool = True) -> str:
        gle_symbol = _gle_symbol_with_spacing_boundary(self.gle)
        if include_unit and self.unit:
            return f"{gle_symbol} ({_unit_to_gle(self.unit)})"
        return gle_symbol


def split_parameter_name(name: str) -> tuple[str, str | None]:
    """Split a parameter name into base and optional numeric suffix."""
    match = _INDEXED_PARAM_RE.match(name)
    if not match:
        return name, None
    return match.group(1), match.group(2)


def _append_latex_index(symbol: str, index: str) -> str:
    if symbol.startswith("$") and symbol.endswith("$"):
        return f"${symbol[1:-1]}_{{{index}}}$"
    return f"{symbol}_{index}"


def _unit_to_gle(unit: str) -> str:
    # Keep GLE labels in native markup (no $...$ math mode).
    unit_gle = unit.replace("μ", r"{\rm \mu}{}")
    # Normalize common unicode superscripts to ASCII exponent markup.
    unit_gle = unit_gle.replace("⁻¹", "^{-1}")
    unit_gle = unit_gle.replace("⁻²", "^{-2}")
    unit_gle = unit_gle.replace("⁻³", "^{-3}")
    return unit_gle


def _gle_symbol_with_spacing_boundary(symbol: str) -> str:
    # For bare control words (e.g. \lambda), add {} so following spaces are not swallowed.
    if _GLE_CONTROL_WORD_RE.match(symbol):
        return f"{symbol}{{}}"
    return symbol


PARAM_INFO_REGISTRY: dict[str, ParamInfo] = {
    "A": ParamInfo("A", "A", "A", r"$A$", r"{\it A}", "%", default_min=0.0),
    "A0": ParamInfo("A0", "A0", "A₀", r"$A_0$", r"{\it A}_{0}", "%", default_min=0.0),
    "A_bg": ParamInfo("A_bg", "A_bg", "A_bg", r"$A_{bg}$", r"{\it A}_{bg}", "%", default_min=0.0),
    "Lambda": ParamInfo("Lambda", "Lambda", "λ", r"$\lambda$", r"\lambda", "μs⁻¹", default_min=0.0),
    "sigma": ParamInfo("sigma", "sigma", "σ", r"$\sigma$", r"\sigma", "μs⁻¹", default_min=0.0),
    "Delta": ParamInfo("Delta", "Delta", "Δ", r"$\Delta$", r"\Delta", "μs⁻¹", default_min=0.0),
    "beta": ParamInfo("beta", "beta", "β", r"$\beta$", r"\beta", default_min=0.0),
    "phase": ParamInfo("phase", "phase", "φ", r"$\phi$", r"\phi", "rad"),
    "frequency": ParamInfo("frequency", "frequency", "f", r"$f$", r"{\it f}", "MHz", default_min=0.0),
    "baseline": ParamInfo("baseline", "baseline", "baseline", "baseline", "baseline", "%"),
    "a": ParamInfo("a", "a", "a", r"$a$", r"{\it a}"),
    "b": ParamInfo("b", "b", "b", r"$b$", r"{\it b}"),
    "c": ParamInfo("c", "c", "c", r"$c$", r"{\it c}"),
    "n": ParamInfo("n", "n", "n", r"$n$", r"{\it n}"),
    "tau": ParamInfo("tau", "tau", "τ", r"$\tau$", r"\tau", default_min=0.0),
    "B0": ParamInfo("B0", "B0", "B₀", r"$B_0$", r"{\it B}_{0}", "G"),
    "Bwid": ParamInfo("Bwid", "Bwid", "B_wid", r"$B_{wid}$", r"{\it B}_{wid}", "G", default_min=0.0),
    "Tc": ParamInfo("Tc", "Tc", "T_c", r"$T_c$", r"{\it T}_{c}", "K", default_min=0.0),
    "Ea": ParamInfo("Ea", "Ea", "E_a", r"$E_a$", r"{\it E}_{a}", "meV", default_min=0.0),
    "D": ParamInfo("D", "D", "D", r"$D$", r"{\it D}", "MHz", default_min=0.0),
    "nu": ParamInfo("nu", "nu", "ν", r"$\nu$", r"\nu", "MHz", default_min=0.0),
    "m": ParamInfo("m", "m", "m", r"$m$", r"{\it m}"),
    "f": ParamInfo("f", "f", "f", r"$f$", r"{\it f}", "μs⁻¹", default_min=0.0),
    "D_2D": ParamInfo("D_2D", "D_2D", "D_2D", r"$D_{2D}$", r"{\it D}_{2D}", "μs⁻¹", default_min=0.0),
    "D_nD": ParamInfo("D_nD", "D_nD", "D_nD", r"$D_{nD}$", r"{\it D}_{nD}", "μs⁻¹", default_min=0.0),
    "D_perp": ParamInfo("D_perp", "D_perp", "D_⊥", r"$D_{\perp}$", r"{\it D}_{\perp}", "μs⁻¹", default_min=0.0),
    "lambda_BG": ParamInfo(
        "lambda_BG", "lambda_BG", "λ_BG", r"$\lambda_{BG}$", r"\lambda_{BG}", "μs⁻¹", default_min=0.0
    ),
    "lambda_0D": ParamInfo(
        "lambda_0D", "lambda_0D", "λ_0D", r"$\lambda_{0D}$", r"\lambda_{0D}", "μs⁻¹", default_min=0.0
    ),
    "C": ParamInfo("C", "C", "C", r"$C$", r"{\it C}", "MHz", default_min=0.0),
}


def get_param_info(name: str) -> ParamInfo:
    """Return metadata for a parameter name, including indexed variants."""
    base_name, index = split_parameter_name(name)
    info = PARAM_INFO_REGISTRY.get(base_name)
    if info is None:
        info = ParamInfo(base_name, base_name, base_name, f"${base_name}$", base_name)
    if index is None:
        return ParamInfo(name, info.plain, info.unicode, info.latex, info.gle, info.unit, info.default_min)
    return info.with_index(index)


def param_info_map(param_names: list[str]) -> dict[str, ParamInfo]:
    """Build a parameter metadata mapping for a parameter-name sequence."""
    return {name: get_param_info(name) for name in param_names}


@dataclass
class Parameter:
    """A single fit parameter."""

    name: str
    value: float = 0.0
    min: float = -float("inf")
    max: float = float("inf")
    fixed: bool = False
    expr: str | None = None  # Expression constraint (e.g. tie to another param)

    @property
    def is_constrained(self) -> bool:
        return self.fixed or self.expr is not None


class ParameterSet:
    """Ordered collection of :class:`Parameter` objects."""

    def __init__(self, params: list[Parameter] | None = None) -> None:
        self._params: dict[str, Parameter] = {}
        for p in params or []:
            self.add(p)

    def add(self, param: Parameter) -> None:
        self._params[param.name] = param

    def __getitem__(self, name: str) -> Parameter:
        return self._params[name]

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    @property
    def free_parameters(self) -> list[Parameter]:
        return [p for p in self if not p.is_constrained]

    @property
    def names(self) -> list[str]:
        return list(self._params)

    def values_array(self) -> list[float]:
        return [p.value for p in self]

    def update_values(self, values: dict[str, float]) -> None:
        for name, val in values.items():
            if name in self._params:
                self._params[name].value = val

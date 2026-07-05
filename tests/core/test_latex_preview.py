"""Typeset (mathtext) preview for composite and parameter-vs-x models.

The real safety net for the typeset live-preview feature: every fragment emitted
by ``latex_terms()`` / ``latex_string()`` must parse through matplotlib's
mathtext parser without raising. Core code cannot import matplotlib, but *tests*
can — so this module imports it to validate mathtext-safety end to end.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib.mathtext import MathTextParser  # noqa: E402

from asymmetry.core.fitting.composite import (  # noqa: E402
    COMPONENTS,
    CompositeModel,
    LatexTerm,
)
from asymmetry.core.fitting.latex_preview import (  # noqa: E402
    fallback_function_latex,
    param_symbol_latex,
    transform_template,
)
from asymmetry.core.fitting.parameter_models import (  # noqa: E402
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
)

_PARSER = MathTextParser("agg")


def _assert_mathtext_safe(fragment: str) -> None:
    """Fail if ``fragment`` does not parse as matplotlib mathtext."""
    # Wrap exactly as the GUI renderer will: single ``$...$`` math span.
    _PARSER.parse(f"${fragment}$")


# --- contract shape ----------------------------------------------------------


def test_latex_terms_never_empty_for_valid_model() -> None:
    assert CompositeModel(["Exponential"]).latex_terms()
    assert ParameterCompositeModel(["Constant"]).latex_terms()


def test_latex_string_is_join_of_terms() -> None:
    model = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    terms = model.latex_terms()
    assert model.latex_string() == "".join(t.separator + t.latex for t in terms)


def test_first_term_has_empty_separator_rest_signed() -> None:
    model = CompositeModel.from_expression("Exponential + Gaussian")
    terms = model.latex_terms()
    assert terms[0].separator == ""
    assert all(t.separator in (" + ", " - ") for t in terms[1:])


def test_single_multiplicative_chain_is_one_term() -> None:
    model = CompositeModel.from_expression("Exponential * Gaussian * Oscillatory")
    assert len(model.latex_terms()) == 1


def test_latex_terms_returns_latexterm_instances() -> None:
    for term in CompositeModel(["Exponential"]).latex_terms():
        assert isinstance(term, LatexTerm)


# --- every registered component is mathtext-safe -----------------------------


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_every_time_domain_component_parses(name: str) -> None:
    model = CompositeModel([name])
    _assert_mathtext_safe(model.latex_string())
    for term in model.latex_terms():
        _assert_mathtext_safe(term.latex)


@pytest.mark.parametrize("name", sorted(PARAMETER_MODEL_COMPONENTS))
def test_every_parameter_model_component_parses(name: str) -> None:
    model = ParameterCompositeModel([name])
    _assert_mathtext_safe(model.latex_string())
    for term in model.latex_terms():
        _assert_mathtext_safe(term.latex)


# --- representative composite / fraction / nested models ---------------------

_COMPOSITE_EXPRESSIONS = [
    "Exponential",
    "Exponential + Constant",
    "Exponential + Gaussian + Constant",
    "Exponential * Gaussian",
    "Exponential / Gaussian",
    "Exponential * (Gaussian + Constant)",
    "Oscillatory * Exponential + Constant",
    "(Exponential + Gaussian){frac} + Constant",
    "(Exponential + Gaussian + Oscillatory){frac}",
    "(Exponential + Gaussian){frac} + (Oscillatory + Constant){frac}",
    "MuoniumTF + Constant",
    "StaticGKT_ZF * Exponential",
    "GaussianPeak + LorentzianPeak + ConstantBackground",
]


@pytest.mark.parametrize("expression", _COMPOSITE_EXPRESSIONS)
def test_composite_expressions_parse(expression: str) -> None:
    model = CompositeModel.from_expression(expression)
    _assert_mathtext_safe(model.latex_string())
    for term in model.latex_terms():
        _assert_mathtext_safe(term.latex)


def test_fraction_remainder_prefix_parses() -> None:
    # (1 - f_X - f_Y) remainder over two free fractions must be mathtext-safe.
    model = CompositeModel.from_expression("(Exponential + Gaussian + Oscillatory){frac}")
    fragment = model.latex_string()
    assert r"\sqrt" not in fragment  # no accidental quadrature here
    assert "f_{\\mathrm{Exponential}}" in fragment
    _assert_mathtext_safe(fragment)


# --- fraction group tagging --------------------------------------------------


def test_fraction_group_tags_the_intersecting_term() -> None:
    model = CompositeModel.from_expression("(Exponential + Gaussian){frac} + Constant")
    terms = model.latex_terms()
    # Term 0 spans the fraction group (0, 1); term 1 (Constant) touches none.
    assert terms[0].group == (0, 1)
    assert terms[1].group is None


def test_two_fraction_groups_tag_their_own_terms() -> None:
    model = CompositeModel.from_expression(
        "(Exponential + Gaussian){frac} + (Oscillatory + Constant){frac}"
    )
    groups = [t.group for t in model.latex_terms()]
    assert (0, 1) in groups
    assert (2, 3) in groups


def test_non_fraction_model_tags_no_groups() -> None:
    model = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    assert all(t.group is None for t in model.latex_terms())


# --- parameter-model quadrature + chains -------------------------------------


def test_quadrature_chain_renders_as_sqrt_of_squares() -> None:
    model = ParameterCompositeModel.from_expression(f"Constant {chr(0x2295)} Linear")
    latex = model.latex_string()
    assert latex.startswith(r"\sqrt{")
    assert "^{2}" in latex
    _assert_mathtext_safe(latex)


def test_builtin_quadrature_component_is_safe() -> None:
    # SC_SWave_Q's own template is a sqrt(...) form; ensure it parses.
    _assert_mathtext_safe(ParameterCompositeModel(["SC_SWave_Q"]).latex_string())


def test_parameter_model_group_always_none() -> None:
    model = ParameterCompositeModel.from_expression("Linear + Constant")
    assert all(t.group is None for t in model.latex_terms())


def test_parameter_model_additive_split() -> None:
    model = ParameterCompositeModel.from_expression("Linear + Constant")
    terms = model.latex_terms()
    assert len(terms) == 2
    assert terms[0].separator == ""
    assert terms[1].separator == " + "


def test_parenthesized_parameter_model_falls_back_safely() -> None:
    model = ParameterCompositeModel.from_expression("(Linear + Constant)")
    _assert_mathtext_safe(model.latex_string())


# --- transformer unit behaviour ----------------------------------------------


def test_transform_returns_none_for_opaque_kernel() -> None:
    # A ``;``-separated opaque kernel call is outside the transformable subset.
    assert transform_template("Gz(t; Delta=\x000\x00)", {"\x000\x00": r"\Delta"}) is None


def test_transform_handles_exp_sqrt_and_powers() -> None:
    symbols = {"\x000\x00": r"\lambda"}
    out = transform_template("exp(-\x000\x00*t)", symbols)
    assert out == r"e^{-\lambda\,t}"
    _assert_mathtext_safe(out)


def test_param_symbol_strips_dollars() -> None:
    assert param_symbol_latex(r"$\lambda$", "Lambda") == r"\lambda"


def test_param_symbol_falls_back_for_freetext() -> None:
    # A non-mathtext (free-text) latex string must yield a safe \mathrm symbol.
    symbol = param_symbol_latex("Integral asymmetry (%)", "integral_asym")
    _assert_mathtext_safe(symbol)


def test_param_symbol_rejects_double_backslash() -> None:
    # A mis-escaped registry latex (``$\\nu$``) must not leak an invalid symbol.
    symbol = param_symbol_latex(r"$\\nu$", "nu")
    _assert_mathtext_safe(symbol)


def test_fallback_function_form_is_safe() -> None:
    fragment = fallback_function_latex("MuoniumTF", [r"A_\mu", r"\phi"])
    assert fragment.startswith(r"\mathrm{MuoniumTF}(t")
    _assert_mathtext_safe(fragment)


# --- formula_string regression (must stay byte-identical) --------------------


@pytest.mark.parametrize("expression", _COMPOSITE_EXPRESSIONS)
def test_formula_string_unchanged(expression: str) -> None:
    # latex_* additions must not perturb the ASCII formula preview.
    model = CompositeModel.from_expression(expression)
    # Re-deriving from the round-tripped structure gives the same formula.
    again = CompositeModel.from_dict(model.to_dict())
    assert model.formula_string() == again.formula_string()

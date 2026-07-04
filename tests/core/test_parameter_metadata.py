"""Tests for centralized parameter metadata and registry coverage."""

from __future__ import annotations

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
)
from asymmetry.core.fitting.parameters import get_param_info


def test_get_param_info_known_parameter_has_rich_formats() -> None:
    info = get_param_info("Lambda")
    assert info.plain == "Lambda"
    assert info.unicode == "λ"
    assert info.latex == r"$\lambda$"
    assert info.gle == r"\lambda"
    assert info.unit == "µs⁻¹"


def test_get_param_info_shape_factor_a_has_expected_defaults() -> None:
    info = get_param_info("shape_factor_a")
    assert info.plain == "shape_factor_a"
    assert info.latex == r"$a_{\mathrm{shape}}$"
    assert info.default_min == 0.0
    assert info.description is not None


def test_get_param_info_fraction_weight_name() -> None:
    # f_<Component> fraction weights are synthesized (not registered): a
    # component-labelled symbol, a [0, 1] floor, and a description.
    info = get_param_info("f_Oscillatory")
    assert info.plain == "f_Oscillatory"
    assert info.latex == r"$f_{\mathrm{Oscillatory}}$"
    assert info.default_min == 0.0
    assert info.description == "Fractional weight of the Oscillatory term."


def test_get_param_info_fraction_weight_disambiguated_name() -> None:
    # A _<n> suffix merges into the subscript rather than double-subscripting.
    info = get_param_info("f_Gaussian_2")
    assert info.plain == "f_Gaussian_2"
    assert info.latex == r"$f_{\mathrm{Gaussian},2}$"
    assert info.default_min == 0.0


def test_signed_baseline_params_have_no_lower_bound() -> None:
    # A_bg is a signed DC baseline of the asymmetry: a 2-group F–B transverse-
    # field asymmetry sits on a large negative offset, so a 0 lower bound would
    # clamp the fit. It (and its indexed variants) must default to −inf.
    assert get_param_info("A_bg").default_min is None
    assert get_param_info("A_bg_2").default_min is None
    # Other signed baseline / offset / polynomial-constant params likewise.
    for name in ("baseline", "c0", "c1", "bg", "slope"):
        assert get_param_info(name).default_min is None, name


def test_positive_definite_params_keep_zero_lower_bound() -> None:
    # Amplitudes, relaxation rates and widths are physically non-negative and
    # must keep their 0 floor (only genuinely-signed baselines were relaxed).
    for name in ("A", "A0", "Lambda", "sigma", "Delta"):
        assert get_param_info(name).default_min == 0.0, name


def test_get_param_info_indexed_parameter_preserves_formats() -> None:
    info = get_param_info("A0_2")
    assert info.plain == "A0_2"
    assert info.unicode == "A₀_2"
    # An existing subscript merges with the index — a naive suffix would give
    # the invalid double subscript $A_0_{2}$, which mathtext rejects.
    assert info.latex == r"$A_{0,2}$"
    assert info.gle == r"{\it A}_{0,2}"
    assert info.unit == "%"


def test_gle_labels_use_native_gle_markup_without_latex_math_mode() -> None:
    for name in ["A0", "A_bg", "Lambda", "sigma", "Delta", "phase", "frequency"]:
        info = get_param_info(name)
        assert "$" not in info.gle


def test_gle_labels_convert_microsecond_inverse_unit_safely() -> None:
    info = get_param_info("Lambda")
    label = info.gle_label()
    assert "\\mus" not in label
    assert "{\\rm \\mu}{}s^{-1}" in label


def test_gle_labels_keep_space_between_greek_symbol_and_unit() -> None:
    info = get_param_info("Lambda")
    assert info.gle_label().startswith(r"\lambda{} (")


def test_models_registry_has_param_info_for_all_param_names() -> None:
    for model in MODELS.values():
        assert set(model.param_names) == set(model.param_info)


def test_composite_components_have_param_info_for_all_param_names() -> None:
    for component in COMPONENTS.values():
        assert set(component.param_names) == set(component.param_info)


def test_parameter_model_components_have_param_info_for_all_param_names() -> None:
    for component in PARAMETER_MODEL_COMPONENTS.values():
        assert set(component.param_names) == set(component.param_info)


def test_composite_model_propagates_indexed_param_info() -> None:
    model = CompositeModel(["Exponential", "Exponential"], operators=["+"])
    assert model.param_info["A_1"].latex == r"$A_{1}$"
    assert model.param_info["Lambda_2"].latex == r"$\lambda_{2}$"
    assert model.param_info["A_1"].unit == "%"


def test_parameter_composite_model_keeps_component_specific_units() -> None:
    model = ParameterCompositeModel(["DiffusionLF_2D"], operators=[])
    assert model.param_info["A"].unit == "MHz"

    indexed = ParameterCompositeModel(["DiffusionLF_2D", "DiffusionLF_3D"], operators=["+"])
    assert indexed.param_info["A_1"].unit == "MHz"
    assert indexed.param_info["A_2"].unit == "MHz"


def test_get_param_info_free_text_name_stays_out_of_mathtext() -> None:
    """Regression: unknown free-text quantity names (the integral scan's
    'Integral asymmetry (%)') were wrapped in $...$, and matplotlib's
    mathtext parser raised at draw time on the spaces and '%'."""
    info = get_param_info("Integral asymmetry (%)")
    assert "$" not in info.latex
    assert info.latex == "Integral asymmetry (%)"

    # The mathtext parser itself must accept the resulting axis label.
    from matplotlib.mathtext import MathTextParser

    label = info.latex_label()
    if "$" in label:  # pragma: no cover — guarded above
        MathTextParser("agg").parse(label)

    # Clean unknown symbols keep the mathtext wrapping.
    assert get_param_info("Zeta").latex == "$Zeta$"

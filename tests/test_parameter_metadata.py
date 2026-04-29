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
    assert info.unit == "μs⁻¹"


def test_get_param_info_shape_factor_a_has_expected_defaults() -> None:
    info = get_param_info("shape_factor_a")
    assert info.plain == "shape_factor_a"
    assert info.latex == r"$a_{\mathrm{shape}}$"
    assert info.default_min == 0.0
    assert info.description is not None


def test_get_param_info_indexed_parameter_preserves_formats() -> None:
    info = get_param_info("A0_2")
    assert info.plain == "A0_2"
    assert info.unicode == "A₀_2"
    assert info.latex == r"$A_0_{2}$"
    assert info.gle == r"{\it A}_{0}_{2}"
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

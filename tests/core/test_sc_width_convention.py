"""Independent Gaussian-characteristic-function and Brandt width checks."""

import numpy as np
import pytest
from scipy.integrate import quad

from asymmetry.core.fitting.models import gaussian_relaxation
from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
from asymmetry.core.fitting.sc import GaussianWidth
from asymmetry.core.fitting.sc.constants import lambda_nm_to_sigma_us, sigma_to_lambda_nm
from asymmetry.core.utils.constants import MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


@pytest.mark.parametrize("function", [sigma_to_lambda_nm, lambda_nm_to_sigma_us])
def test_width_conversion_requires_an_explicit_convention(function):
    with pytest.raises(TypeError, match="width"):
        function(1.0)


def test_gaussian_field_distribution_fixes_sqrt2():
    field_rms_tesla = 0.001
    angular_width = 2 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * field_rms_tesla
    for time in (0.1, 0.4, 1.2):
        characteristic, _ = quad(
            lambda z: np.exp(-z * z / 2) * np.cos(angular_width * time * z) / np.sqrt(2 * np.pi),
            -np.inf,
            np.inf,
            epsabs=1e-13,
            epsrel=1e-13,
        )
        fitted_sigma = np.sqrt(-np.log(characteristic)) / time
        assert angular_width / fitted_sigma == pytest.approx(np.sqrt(2), rel=1e-10)
        assert GaussianWidth.GAUSSIAN_COMPONENT.value == pytest.approx(
            angular_width / fitted_sigma, rel=1e-10
        )
        assert gaussian_relaxation(time, 1.0, fitted_sigma) == pytest.approx(
            characteristic, abs=1e-12
        )


def test_workflow_relation_matches_library_after_conversion():
    sigma_g = 1.164
    gaussian_lambda = sigma_to_lambda_nm(sigma_g, width=GaussianWidth.GAUSSIAN_COMPONENT)
    second_moment_lambda = sigma_to_lambda_nm(sigma_g, width=GaussianWidth.SECOND_MOMENT)
    # Rounded teaching-guide coefficient, for exp(-(sigma_G*t)^2).
    assert gaussian_lambda == pytest.approx(np.sqrt(75780 / sigma_g), rel=1e-3)
    assert second_moment_lambda / gaussian_lambda == pytest.approx(2**0.25, rel=1e-12)


@pytest.mark.parametrize("width", list(GaussianWidth))
def test_width_conversion_round_trip(width):
    sigma = np.array([0.2, 1.0, 2.0])
    penetration = sigma_to_lambda_nm(sigma, width=width)
    np.testing.assert_allclose(lambda_nm_to_sigma_us(penetration, width=width), sigma, rtol=1e-14)


@pytest.mark.parametrize("width", list(GaussianWidth))
@pytest.mark.parametrize(
    "convert,positive,limit,expected_limit,invalid",
    [
        (sigma_to_lambda_nm, 1.0, 0.0, np.inf, [-1.0, -np.inf, np.inf]),
        (lambda_nm_to_sigma_us, 200.0, np.inf, 0.0, [0.0, -1.0, -np.inf]),
    ],
)
def test_width_conversion_domain(width, convert, positive, limit, expected_limit, invalid):
    positive_value = convert(positive, width=width).item()
    assert 0.0 < positive_value < np.inf
    values = np.array([[np.nan, positive, limit], invalid])
    expected = np.array([[np.nan, positive_value, expected_limit], [np.nan] * 3])
    np.testing.assert_array_equal(convert(values, width=width), expected)

    for value, expected_value in zip(values.flat, expected.flat, strict=True):
        converted = convert(float(value), width=width)
        assert converted.shape == ()
        np.testing.assert_equal(converted, expected_value)


@pytest.mark.parametrize("width", list(GaussianWidth))
def test_brandt_width_scales_as_inverse_depth_squared(width):
    sigma, sigma_at_double_depth = lambda_nm_to_sigma_us([200.0, 400.0], width=width)
    assert sigma > 0.0
    assert sigma_at_double_depth == pytest.approx(sigma / 4.0, rel=1e-14)


@pytest.mark.parametrize(
    "width,width_factor",
    [(GaussianWidth.SECOND_MOMENT, 1.0), (GaussianWidth.GAUSSIAN_COMPONENT, np.sqrt(2.0))],
)
def test_extreme_finite_widths_give_representable_depths(width, width_factor):
    widths = np.array([1e308, np.finfo(float).smallest_subnormal])
    # Brandt Eq. (11) with the rounded 0.0609 amplitude gives this scale in nm²/µs.
    expected_depths = np.sqrt(107244.85533737145 / width_factor) / np.sqrt(widths)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        depths = sigma_to_lambda_nm(widths, width=width)
        recovered_widths = lambda_nm_to_sigma_us(expected_depths, width=width)
    np.testing.assert_allclose(depths, expected_depths, rtol=1e-14, atol=0.0)
    assert recovered_widths[0] == pytest.approx(widths[0], rel=1e-14)
    assert recovered_widths[1] == widths[1]


@pytest.mark.parametrize("width", list(GaussianWidth))
def test_very_small_positive_depth_can_overflow_the_width(width):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert np.isposinf(lambda_nm_to_sigma_us(1e-200, width=width))


@pytest.mark.parametrize("width", list(GaussianWidth))
def test_custom_brandt_coefficient_rescales_both_conversions(width):
    depth = sigma_to_lambda_nm(1.0, width=width, brandt_coefficient=0.0609)
    sigma = lambda_nm_to_sigma_us(200.0, width=width, brandt_coefficient=0.0609)
    assert sigma_to_lambda_nm(1.0, width=width, brandt_coefficient=0.2436) == 2 * depth
    assert lambda_nm_to_sigma_us(200.0, width=width, brandt_coefficient=0.2436) == 4 * sigma


@pytest.mark.parametrize("function", [sigma_to_lambda_nm, lambda_nm_to_sigma_us])
def test_unknown_width_convention_is_rejected(function):
    with pytest.raises(ValueError, match="GaussianWidth"):
        function(1.0, width="unrecognised")


@pytest.mark.parametrize("name", ["SC_Brandt_VortexLattice", "SC_Brandt_VortexLattice_Powder"])
def test_brandt_help_identifies_second_moment_input(name):
    from asymmetry.core.fitting.component_docs import (
        get_component_applicability,
        get_component_references,
    )

    help_text = get_component_applicability(name)
    assert "second-moment" in help_text
    assert "√2" in help_text
    assert get_component_references(name)


@pytest.mark.parametrize(
    "expression", ["Gaussian", "Gaussian + Gaussian", "Oscillatory * Gaussian + Gaussian"]
)
def test_width_sources_follow_component_identity(expression):
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.parameter_carry import ComponentParameter
    from asymmetry.core.fitting.sc.trend_width import model_width_sources

    model = CompositeModel.from_expression(expression)
    sources = model_width_sources(model)
    for name, identity in model.parameter_identities().items():
        if isinstance(identity, ComponentParameter):
            source = sources[name]
            assert source["component"] == model.components[identity.component].name
            assert source["parameter"] == identity.local_name
            assert source["index"] == identity.component
            assert bool(source["widths"]) == (
                source["component"] == "Gaussian" and identity.local_name == "sigma"
            )


@pytest.mark.parametrize(
    "consumer", ["SC_Brandt_VortexLattice", "SC_Brandt_VortexLattice_Powder + Constant"]
)
@pytest.mark.parametrize(
    "expression,allowed",
    [
        (None, False),
        ("sqrt(2)*x", True),
        ("x * sqrt(2)", True),
        ("1.4142135623730951*x", False),
        ("sqrt(2)*x + x**2", False),
        ("x*x", False),
    ],
)
def test_only_explicit_width_conversion_admits_known_gaussian(consumer, expression, allowed):
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.trend_width import TrendWidthContext, WidthSource

    transform = AxisTransform.identity() if expression is None else AxisTransform.custom(expression)
    context = TrendWidthContext(
        "renamed",
        (None, WidthSource("Gaussian", 2, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,))),
        transform,
    )
    if allowed:
        context.validate(ParameterCompositeModel.from_expression(consumer))
        assert context.effective_widths() == (GaussianWidth.SECOND_MOMENT,)
    else:
        with pytest.raises(ValueError, match="renamed.*Gaussian.*second-moment.*sqrt"):
            context.validate(ParameterCompositeModel.from_expression(consumer))
    context.validate(ParameterCompositeModel(["SC_SWave"]))


def test_mixed_widths_need_separate_conversion_and_selected_rows_keep_identity():
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.trend_width import TrendWidthContext, WidthSource

    gaussian = WidthSource("Gaussian", 0, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,))
    second = WidthSource("SC_SWave", 0, "sigma_0", (GaussianWidth.SECOND_MOMENT,))
    mixed = TrendWidthContext("rate", (None, second, gaussian), AxisTransform.custom("sqrt(2)*x"))
    with pytest.raises(ValueError, match="mixed conventions"):
        mixed.validate(ParameterCompositeModel(["SC_Brandt_VortexLattice"]))
    selected = mixed.select([False, False, True])
    selected.validate(ParameterCompositeModel(["SC_Brandt_VortexLattice"]))
    assert TrendWidthContext.from_dict(selected.to_dict()) == selected
    with pytest.raises(ValueError, match="already has second-moment"):
        mixed.select([False, True, False]).validate(
            ParameterCompositeModel(["SC_Brandt_VortexLattice"])
        )
    TrendWidthContext("rate", (second,), AxisTransform.custom("x")).validate(
        ParameterCompositeModel(["SC_Brandt_VortexLattice"])
    )


def test_registered_alias_keeps_canonical_gaussian_identity(monkeypatch):
    from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
    from asymmetry.core.fitting.sc.trend_width import model_width_sources

    monkeypatch.setitem(COMPONENTS, "GaussianAlias", COMPONENTS["Gaussian"])
    sources = model_width_sources(CompositeModel.from_expression("GaussianAlias + Gaussian"))
    assert sources["sigma_1"]["component"] == "Gaussian"
    assert sources["sigma_2"]["widths"] == ["GAUSSIAN_COMPONENT"]


def test_brandt_consumer_alias_uses_canonical_component(monkeypatch):
    from asymmetry.core.fitting.parameter_models import (
        PARAMETER_MODEL_COMPONENTS,
        ParameterCompositeModel,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.trend_width import TrendWidthContext, WidthSource

    monkeypatch.setitem(
        PARAMETER_MODEL_COMPONENTS,
        "BrandtAlias",
        PARAMETER_MODEL_COMPONENTS["SC_Brandt_VortexLattice"],
    )
    model = ParameterCompositeModel.from_expression("BrandtAlias")
    source = WidthSource("Gaussian", 0, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,))
    with pytest.raises(ValueError, match="Gaussian.*second-moment"):
        fit_parameter_model(
            np.arange(1.0, 5.0),
            np.ones(4),
            np.ones(4),
            model,
            ParameterSet([Parameter(name, value) for name, value in model.param_defaults.items()]),
            width_context=TrendWidthContext("width", (source,) * 4),
        )


def test_nonadditive_parameter_models_do_not_claim_an_absolute_width_contract():
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.trend_width import (
        TrendWidthContext,
        WidthSource,
        model_width_sources,
    )

    source = WidthSource("Gaussian", 0, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,))
    context = TrendWidthContext("normalised", (source,), AxisTransform.custom("x*x"))
    model = ParameterCompositeModel.from_expression(
        "SC_Brandt_VortexLattice / SC_Brandt_VortexLattice"
    )
    context.validate(model)
    for expression in ["SC_SWave * Constant", "Linear / Constant"]:
        sources = model_width_sources(
            ParameterCompositeModel.from_expression(expression), (GaussianWidth.GAUSSIAN_COMPONENT,)
        )
        assert all(not source["widths"] for source in sources.values())


@pytest.mark.parametrize(
    "expression",
    [
        "SC_Brandt_VortexLattice + Constant*Constant",
        "SC_Brandt_VortexLattice ⊕ (Constant*Constant)",
    ],
)
def test_bare_brandt_branch_still_requires_second_moment_beside_a_product(expression):
    from asymmetry.core.fitting.parameter_models import fit_parameter_model
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.trend_width import TrendWidthContext, WidthSource

    model = ParameterCompositeModel.from_expression(expression)
    parameters = ParameterSet(
        [
            Parameter(name, value, fixed=name != "lambda_ab")
            for name, value in model.param_defaults.items()
        ]
    )
    source = WidthSource("Gaussian", 0, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,))
    with pytest.raises(ValueError, match="Gaussian.*second-moment"):
        fit_parameter_model(
            np.linspace(0.01, 1.0, 5),
            np.ones(5),
            np.ones(5),
            model,
            parameters,
            width_context=TrendWidthContext("width", (source,) * 5),
        )
    nested = ParameterCompositeModel.from_expression(
        "(SC_Brandt_VortexLattice + Constant)*Constant"
    )
    TrendWidthContext("normalised", (source,)).validate(nested)

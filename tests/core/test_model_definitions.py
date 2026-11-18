"""Independent landmarks and integral definitions for elementary fit shapes."""

import numpy as np
import pytest
from scipy.integrate import quad

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


def test_relaxation_half_height_landmarks():
    amplitude, rate = 17.0, 0.7
    for name, half_time in (
        ("Exponential", np.log(2) / rate),
        ("Gaussian", np.sqrt(np.log(2)) / rate),
    ):
        parameter = "Lambda" if name == "Exponential" else "sigma"
        values = COMPONENTS[name].function(
            np.array([0, half_time]), A=amplitude, **{parameter: rate}
        )
        np.testing.assert_allclose(values, [amplitude, amplitude / 2], atol=1e-13)


def test_oscillation_quarter_period_and_phase():
    frequency, amplitude = 1.7, 13.0
    time = np.arange(5) / (4 * frequency)
    expected = amplitude * np.array([1, 0, -1, 0, 1])
    for name, coordinate in (
        ("Oscillatory", {"frequency": frequency}),
        (
            "OscillatoryField",
            {"field": frequency / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)},
        ),
    ):
        function = COMPONENTS[name].function
        np.testing.assert_allclose(
            function(time, A=amplitude, phase=0, **coordinate), expected, atol=1e-12
        )
        shifted = function(time, A=amplitude, phase=np.pi / 2, **coordinate)
        np.testing.assert_allclose(shifted, amplitude * np.array([0, -1, 0, 1, 0]), atol=1e-12)


@pytest.mark.parametrize("time", [0.01, 0.3, 2, 20])
def test_stretched_half_is_a_positive_rate_mixture(time):
    # Levy rate density: r=Lambda/(4u²) turns its Laplace transform into this integral.
    rate = 0.7
    reference = quad(
        lambda u: 2 / np.sqrt(np.pi) * np.exp(-u * u - rate * time / (4 * u * u)),
        0,
        np.inf,
        epsabs=1e-12,
    )[0]
    actual = COMPONENTS["StretchedExponential"].function(
        np.array([time]), A=1, Lambda=rate, beta=0.5
    )
    np.testing.assert_allclose(actual, reference, atol=1e-11)


@pytest.mark.parametrize("scaled_time", [0, 1e-3, 0.2, 1, 20, 1e4])
def test_risch_kehr_is_the_continuous_rate_integral(scaled_time):
    # r=x² gives the normalized rate density 1/(pi sqrt(r)(1+r)).
    reference = quad(
        lambda x: 2 / np.pi * np.exp(-scaled_time * x * x) / (1 + x * x),
        0,
        np.inf,
        epsabs=1e-12,
    )[0]
    actual = COMPONENTS["RischKehr"].function(np.array([scaled_time / 0.8]), A=1, Gamma=0.8)
    np.testing.assert_allclose(actual, reference, atol=1e-11)


@pytest.mark.parametrize("name", ["GaussianPeak", "LorentzianPeak"])
def test_spectral_peak_height_width_and_integrated_area(name):
    function = COMPONENTS[name].function
    height, center, width = 7.0, -0.3, 1.4
    values = function(
        np.array([center - width / 2, center, center + width / 2]), height, center, width
    )
    np.testing.assert_allclose(values, [height / 2, height, height / 2], atol=1e-13)
    area = quad(lambda x: function(np.array([x]), height, center, width)[0], -np.inf, np.inf)[0]
    expected = (
        height * width * (np.sqrt(np.pi / np.log(2)) / 2 if name == "GaussianPeak" else np.pi / 2)
    )
    assert area == pytest.approx(expected, rel=1e-10)


def test_background_intercepts_and_slope():
    coordinate = np.array([-2.0, 0.0, 3.0])
    for name, parameters in (("Constant", {"A_bg": -0.7}), ("ConstantBackground", {"bg": -0.7})):
        np.testing.assert_array_equal(
            COMPONENTS[name].function(coordinate, **parameters), [-0.7] * 3
        )
    linear = COMPONENTS["LinearBackground"].function(coordinate, bg=-0.7, slope=0.4)
    np.testing.assert_allclose(linear, [-1.5, -0.7, 0.5], atol=1e-14)


def test_offset_arcsine_components_match_independent_average():
    time = np.linspace(0, 4, 51)
    center, halfwidth, phase = 1.2, 0.7, 0.4
    # Uniform modulation angle induces the offset arcsine field distribution.
    reference = np.array(
        [
            quad(
                lambda angle: np.cos(2 * np.pi * (center + halfwidth * np.cos(angle)) * t + phase),
                0,
                np.pi,
                epsabs=1e-12,
            )[0]
            / np.pi
            for t in time
        ]
    )
    for name, parameters in (
        ("OverhauserPowderCentre", {"frequency": center, "delta_frequency": halfwidth}),
        (
            "OverhauserPowderCutoff",
            {"frequency": center + halfwidth, "ratio": (center - halfwidth) / (center + halfwidth)},
        ),
    ):
        actual = COMPONENTS[name].function(
            time, A=1, lambda_T=0, lambda_L=0, phase=phase, **parameters
        )
        np.testing.assert_allclose(actual, 1 / 3 + 2 / 3 * reference, atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    "model_name,expression",
    [
        ("ExponentialRelaxation", "Exponential + Constant"),
        ("GaussianRelaxation", "Gaussian + Constant"),
        ("Oscillatory", "Oscillatory * Exponential + Constant"),
        ("StretchedExponential", "StretchedExponential + Constant"),
        ("StaticGKT_ZF", "StaticGKT_ZF + Constant"),
        ("LFKuboToyabe", "LongitudinalFieldKT + Constant"),
        ("DynamicGaussianKT", "DynamicGaussianKT + Constant"),
        ("DynamicLorentzianKT", "DynamicLorentzianKT + Constant"),
        ("Keren", "Keren + Constant"),
        ("Abragam", "Abragam + Constant"),
    ],
)
def test_standalone_amplitude_baseline_and_composition(model_name, expression):
    """Registry correspondence only; independent shape evidence lives in other cases."""
    from asymmetry.core.fitting.composite import CompositeModel

    definition = MODELS[model_name]
    parameters = {**definition.param_defaults, "A0": 17.0, "baseline": -0.6}
    composite = CompositeModel.from_expression(expression)
    translated = parameters.copy()
    translated["A_1"] = translated.pop("A0")
    translated["A_bg"] = translated.pop("baseline")
    time = np.linspace(0, 3, 19)
    np.testing.assert_allclose(
        definition.function(time, **parameters), composite.function(time, **translated), atol=1e-12
    )

"""Independent quadrature, Bessel-series and asymptotic checks of SC numerics."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, k1e, zeta

from asymmetry.core.fitting import Parameter, ParameterCompositeModel, ParameterSet
from asymmetry.core.fitting.parameter_models import fit_parameter_model
from asymmetry.core.fitting.sc import kernel, models


def _reference_integral(delta, temperature):
    # q = sqrt(E²-Delta²) removes the density-of-states endpoint singularity.
    def integrand(q):
        occupation = expit(-np.hypot(q, delta) / temperature)
        return -occupation * (1 - occupation) / temperature

    return quad(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]


def _reference_s_wave(temperatures, tc, gap_ratio):
    values = []
    for temperature in temperatures:
        reduced = temperature / tc
        if reduced <= 0:
            values.append(1.0)
        elif reduced >= 1:
            values.append(0.0)
        else:
            # Carrington-Manzano interpolation, independent of the model helper.
            gap = gap_ratio * np.tanh(1.82 * (1.018 * (1 / reduced - 1)) ** 0.51)
            values.append(1 + 2 * _reference_integral(gap, reduced))
    return np.array(values)


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("ratio", [0.01, 0.1, 1.0, 3.0, 10.0])
def test_thermal_integral_matches_transformed_quad(temperature, ratio):
    expected = _reference_integral(ratio * temperature, temperature)
    assert kernel.thermal_integral(ratio) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0])
def test_thermal_integral_matches_bessel_series(ratio):
    orders = np.arange(1, 201)
    # Expanding f(1-f) into exponentials and integrating gives d*n*K1(n*d).
    expected = -ratio * np.sum(
        (-1.0) ** (orders + 1) * orders * k1e(orders * ratio) * np.exp(-orders * ratio)
    )
    assert kernel.thermal_integral(ratio) == pytest.approx(expected, abs=1e-14)


def test_zero_gap_and_ginzburg_landau_limits():
    assert kernel.thermal_integral(0.0) == pytest.approx(-0.5, abs=1e-15)
    d = 0.01
    density = 1 + 2 * kernel.thermal_integral(d)
    assert density / d**2 == pytest.approx(7 * zeta(3) / (4 * np.pi**2), rel=1e-4)


@pytest.mark.parametrize("ratio", [10.0, 20.0])
def test_s_wave_low_temperature_asymptote(ratio):
    density = 1 + 2 * kernel.thermal_integral(ratio)
    leading = np.sqrt(2 * np.pi * ratio) * np.exp(-ratio)
    assert (1 - density) / leading == pytest.approx(
        1 + 3 / (8 * ratio) - 15 / (128 * ratio**2), abs=2 / ratio**3
    )


def test_default_s_wave_curve_matches_independent_integration():
    temperature = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    np.testing.assert_allclose(
        models.rho_s_wave(temperature, Tc=1.0, gap_ratio=1.764),
        _reference_s_wave(temperature, 1.0, 1.764),
        rtol=0,
        atol=1e-10,
    )


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_thermal_integral_and_density_are_scale_free(scale):
    deltas, temperatures = np.array([0.01, 0.1, 1.0, 3.0]), np.array([0.1, 0.5, 1.0, 0.5])
    np.testing.assert_allclose(
        kernel.thermal_integral(scale * deltas / (scale * temperatures)),
        kernel.thermal_integral(deltas / temperatures),
        rtol=0,
        atol=5e-16,
    )
    np.testing.assert_allclose(
        models.rho_s_wave(temperatures * scale, Tc=2 * scale),
        models.rho_s_wave(temperatures, Tc=2.0),
        rtol=0,
        atol=1e-15,
    )


def test_thermal_integral_shapes_symmetry_and_bounds():
    arguments = np.linspace(-80, 80, 801).reshape(3, 267)
    values = kernel.thermal_integral(arguments)
    assert values.shape == arguments.shape
    assert np.all(values >= -0.5 - 1e-15)
    assert np.all(values <= 0)
    np.testing.assert_array_equal(values, kernel.thermal_integral(-arguments))
    assert kernel.thermal_integral(np.empty((0, 2))).shape == (0, 2)


def test_s_wave_gap_law_recovers_parameters():
    temperature = np.linspace(0.5, 12.0, 24)
    values = 1.3 * _reference_s_wave(temperature, 10.0, 1.764) + 0.1
    model = ParameterCompositeModel(["SC_SWave"])
    parameters = ParameterSet(
        [
            Parameter("sigma_0", 1.2, min=0),
            Parameter("Tc", 9.8, min=0.1),
            Parameter("gap_ratio", 1.8, min=0.1),
            Parameter("sigma_bg", 0.12, min=0),
        ]
    )
    result = fit_parameter_model(temperature, values, np.full_like(values, 0.01), model, parameters)
    assert result.success
    assert result.parameters["Tc"].value == pytest.approx(10.0, abs=1e-3)
    assert result.parameters["gap_ratio"].value == pytest.approx(1.764, abs=5e-4)

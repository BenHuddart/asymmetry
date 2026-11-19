"""Independent observable definitions; these do not identify a sample mechanism."""

from math import comb

import numpy as np
import pytest
from scipy.integrate import quad

from asymmetry.core.fitting.ballistic import autocorrelation_nD as ballistic_return
from asymmetry.core.fitting.diffusion import autocorrelation_nD as diffusion_return
from asymmetry.core.fitting.muon_proton import mu_proton_levels
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS, _redfield
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)


@pytest.mark.parametrize(
    "name", ["Quadratic", "Polynomial", "Cubic", "Quartic", "Quintic", "Sextic"]
)
def test_polynomial_bases_obey_binomial_landmark(name):
    definition = PARAMETER_MODEL_COMPONENTS[name]
    degree = len(definition.param_names) - 1
    x = np.array([-2.0, -1.0, 0.0, 0.25, 1.0, 2.0])
    coefficients = [comb(degree, order) for order in range(degree + 1)]
    np.testing.assert_allclose(
        definition.function(x, *coefficients), (1.0 + x) ** degree, atol=1e-12
    )


@pytest.mark.parametrize("ratio", [0.0, 0.3, 1.0, 3.0])
def test_redfield_matches_exponential_correlation_with_explicit_units(ratio):
    # Keren PRB 50, 10039 (1994), Eqs. (2),(5): r is inverse time,
    # omega and Delta are angular frequencies. Stored nu is r/(2*pi).
    delta, rate = 0.3, 4.0
    omega = ratio * rate
    field = omega / (2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)
    expected = quad(
        lambda t: 2.0 * delta**2 * np.exp(-rate * t) * np.cos(omega * t), 0, np.inf, epsabs=1e-12
    )[0]
    actual = _redfield(
        np.array([field]), D=np.sqrt(2.0 / np.pi) * delta, nu=rate / (2.0 * np.pi), m=2
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_transport_returns_are_independent_brillouin_zone_averages(dimension):
    # A nearest-neighbour random walk has exp[-2Dt(1-cos k)] as its
    # characteristic function; coherent hopping has exp[i*2Dt*cos k].
    rate = 1.7
    times = np.array([0.0, 0.1, 0.8, 2.0])
    classical = [
        quad(lambda k: np.exp(-2 * rate * t * (1 - np.cos(k))), -np.pi, np.pi, epsabs=1e-12)[0]
        / (2 * np.pi)
        for t in times
    ]
    coherent = [
        quad(lambda k: np.cos(2 * rate * t * np.cos(k)), -np.pi, np.pi, epsabs=1e-12)[0]
        / (2 * np.pi)
        for t in times
    ]
    np.testing.assert_allclose(
        diffusion_return(times, rate, n=dimension), np.array(classical) ** dimension, atol=2e-13
    )
    np.testing.assert_allclose(
        ballistic_return(times, rate, n=dimension),
        np.array(coherent) ** (2 * dimension),
        atol=2e-13,
    )


def test_angular_fourier_coefficients_are_orthogonal_projections():
    # Full-turn Fourier projections separate the declared harmonics without
    # assigning a physical tilt or magnetic structure to either coefficient.
    theta = np.arange(1024) * 2 * np.pi / 1024
    fn = PARAMETER_MODEL_COMPONENTS["AngularFourier2"].function
    values = fn(np.degrees(theta), K_avg=7.0, K_1=2.0, theta1=30.0, K_amp=3.0, theta2=60.0)
    assert np.mean(values) == pytest.approx(7.0)
    for order, amplitude, phase in [(1, 2.0, 30.0), (2, 3.0, 60.0)]:
        coefficient = 2 * np.mean(values * np.exp(-1j * order * theta))
        assert coefficient == pytest.approx(
            amplitude * np.exp(-1j * order * np.radians(phase)), abs=1e-13
        )


@pytest.mark.parametrize("coupling", [0.5, 20.0, 500.0])
def test_equal_coupling_zero_field_radical_has_angular_momentum_multiplets(coupling):
    # H/h=A*S_e.(S_mu+S_p): nuclear singlet gives 0 (twice);
    # nuclear triplet coupled to S_e gives J=1/2 at -A and J=3/2 at A/2.
    expected = coupling * np.array([-1.0, -1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(mu_proton_levels(0.0, coupling, coupling), expected, atol=1e-12)


def test_lcr_shapes_have_declared_width_and_pair_sign():
    center, width, amplitude = 400.0, 20.0, 3.0
    gaussian = PARAMETER_MODEL_COMPONENTS["GaussianLCR"].function
    lorentzian = PARAMETER_MODEL_COMPONENTS["LorentzianLCR"].function
    pair = PARAMETER_MODEL_COMPONENTS["LorentzianLCRPair"].function
    np.testing.assert_allclose(
        gaussian(np.array([center - width, center, center + width]), amplitude, center, width),
        amplitude * np.array([np.exp(-0.5), 1, np.exp(-0.5)]),
    )
    np.testing.assert_allclose(
        lorentzian(np.array([center - width, center, center + width]), amplitude, center, width),
        amplitude * np.array([0.5, 1, 0.5]),
    )
    # At a separation of one HWHM, the positive/negative lobes at the
    # two declared centers have equal magnitude and opposite sign.
    np.testing.assert_allclose(
        pair(np.array([center, center + width]), amplitude, center, width, width),
        [amplitude / 2, -amplitude / 2],
    )


def test_lambda_background_is_a_constant():
    fn = PARAMETER_MODEL_COMPONENTS["Lambda_bg"].function
    np.testing.assert_array_equal(fn(np.array([-1000.0, 0.0, 1000.0]), -0.3), [-0.3] * 3)

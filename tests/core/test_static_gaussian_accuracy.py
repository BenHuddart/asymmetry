"""Independent spin-rotation and adaptive-integral checks of static Gaussian KT."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import roots_hermitenorm

from asymmetry.core.fitting.models import (
    LARMOR_RAD_PER_US_PER_GAUSS,
    MODELS,
    gaussian_broadened_kt,
    longitudinal_field_kubo_toyabe,
)


def gaussian_field_average(order, ratio, times):
    """Average classical spin rotation over three independent normal fields."""
    x, weights = roots_hermitenorm(order)
    weights /= np.sqrt(2 * np.pi)
    xx, yy, zz = np.meshgrid(x, x, x + ratio, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2 + zz**2).ravel()
    longitudinal = (zz.ravel() / radius) ** 2
    weights = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).ravel()
    return np.array(
        [np.sum(weights * (longitudinal + (1 - longitudinal) * np.cos(radius * t))) for t in times]
    )


def hayano_integral(times, ratio):
    """Independently integrate the published time integral, without interpolation."""
    return np.array(
        [
            1
            - 2 / ratio**2 * (1 - np.exp(-(u**2) / 2) * np.cos(ratio * u))
            + 2
            / ratio**3
            * quad(
                lambda x: np.exp(-(x**2) / 2) * np.sin(ratio * x),
                0,
                u,
                epsabs=1e-13,
                epsrel=1e-13,
                limit=500,
            )[0]
            for u in times
        ]
    )


@pytest.mark.parametrize("ratio", [0, 0.001, 0.01, 0.049, 0.05, 0.051, 0.06, 0.1, 1])
def test_gaussian_spin_rotation_average(ratio):
    u = np.linspace(0, 5, 31)
    expected = gaussian_field_average(64, ratio, u)
    np.testing.assert_allclose(expected, gaussian_field_average(48, ratio, u), atol=1e-12, rtol=0)
    for delta in (0.05, 0.4, 1, 5):
        for sign in (-1, 1):
            actual = longitudinal_field_kubo_toyabe(
                u / delta, 1, delta, sign * ratio * delta / LARMOR_RAD_PER_US_PER_GAUSS
            )
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("ratio", [0.05, 0.06, 0.999999, 1, 1.000001, 10, 100])
def test_extended_time_adaptive_integral(ratio):
    u = np.array([0, 0.001, 0.13, 1.7, 4, 8, 11.999, 12.001, 20])
    expected = hayano_integral(u, ratio)
    actual = longitudinal_field_kubo_toyabe(u, 1, 1, ratio / LARMOR_RAD_PER_US_PER_GAUSS)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("ratio", [0.05, 1])
def test_field_derivative_across_numerical_transitions(ratio):
    u = np.array([0.3, 1.1, 2.3, 5, 20])
    step = 1e-5
    expected = (hayano_integral(u, ratio + step) - hayano_integral(u, ratio - step)) / (2 * step)
    actual = (
        longitudinal_field_kubo_toyabe(u, 1, 1, (ratio + step) / LARMOR_RAD_PER_US_PER_GAUSS)
        - longitudinal_field_kubo_toyabe(u, 1, 1, (ratio - step) / LARMOR_RAD_PER_US_PER_GAUSS)
    ) / (2 * step)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


def test_shapes_zero_width_and_registry_paths():
    from asymmetry.core.fitting.composite import COMPONENTS

    times = np.array([[[0, 0.2], [3, -1]], [[2, 0.1], [8, 4]]])
    field = 0.06 / LARMOR_RAD_PER_US_PER_GAUSS
    expected = longitudinal_field_kubo_toyabe(times.ravel(), 23, 1, field, 2).reshape(times.shape)
    actual = MODELS["LFKuboToyabe"].function(times, 23, 1, field, 2)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        COMPONENTS["LongitudinalFieldKT"].function(times, A=23, Delta=1, B_L=field), expected - 2
    )
    np.testing.assert_allclose(longitudinal_field_kubo_toyabe(times, 23, 0, field, 2), 25)
    assert longitudinal_field_kubo_toyabe(0.0, 23, 0, 0, 2) == 25
    for width in (0, 0.1, 0.4):
        shaped = gaussian_broadened_kt(times, 1, field, width)
        np.testing.assert_array_equal(
            shaped, gaussian_broadened_kt(times.ravel(), 1, field, width).reshape(times.shape)
        )
    np.testing.assert_array_equal(
        gaussian_broadened_kt(times, 1, field, 0),
        longitudinal_field_kubo_toyabe(times, 1, 1, field),
    )


def test_fit_independent_spin_rotation_curve():
    u = np.linspace(0, 5, 61)
    truth = np.array([1.0, 0.3])
    expected = gaussian_field_average(64, truth[1], u)
    error = 1e-3

    def residual(params):
        return (
            longitudinal_field_kubo_toyabe(u, 1, params[0], params[1] / LARMOR_RAD_PER_US_PER_GAUSS)
            - expected
        ) / error

    fit = least_squares(residual, [0.9, 0.4], bounds=([0.2, 0.01], [2, 2]))
    covariance = np.linalg.inv(fit.jac.T @ fit.jac)
    assert fit.success
    # Absolute supplied errors define the local Fisher uncertainty; no residual rescaling.
    assert np.all(np.abs(fit.x - truth) < 0.1 * np.sqrt(np.diag(covariance)))
    assert np.max(np.abs(fit.fun * error)) < 1e-5


def test_zero_width_limit_at_extreme_field_to_width_ratio():
    times = np.array([0.0, 0.1, 1.0, 5.0])
    np.testing.assert_array_equal(
        longitudinal_field_kubo_toyabe(times, 1, 1e-200, 1.0), np.ones_like(times)
    )


@pytest.mark.parametrize("ratio", [0.0, 0.001, 0.3, 1.0])
@pytest.mark.parametrize("parameter", ["width", "field"])
def test_first_and_second_parameter_differences_against_spin_rotation(ratio, parameter):
    times = np.array([0.0, 0.13, 0.7, 1.7, 3.1, 4.99])
    step = 1e-3
    actual, expected = [], []
    for offset in (-step, 0.0, step):
        width = 1.0 + offset if parameter == "width" else 1.0
        omega = ratio + offset if parameter == "field" else ratio
        actual.append(
            longitudinal_field_kubo_toyabe(times, 1.0, width, omega / LARMOR_RAD_PER_US_PER_GAUSS)
        )
        expected.append(gaussian_field_average(48, omega / width, times * width))
    # Matching second differences protects the local curvature used in covariance fits.
    for values, reference in (
        ((actual[2] - actual[0]) / (2 * step), (expected[2] - expected[0]) / (2 * step)),
        (
            (actual[2] - 2 * actual[1] + actual[0]) / step**2,
            (expected[2] - 2 * expected[1] + expected[0]) / step**2,
        ),
    ):
        np.testing.assert_allclose(values, reference, atol=1e-7, rtol=0)

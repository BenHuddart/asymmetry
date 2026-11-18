"""Independent Gaussian-width averages of the Hayano static LF expression."""

import numpy as np
import pytest
from scipy.integrate import quad, quad_vec
from scipy.optimize import least_squares

from asymmetry.core.fitting.models import (
    LARMOR_RAD_PER_US_PER_GAUSS,
    gaussian_broadened_kt,
    longitudinal_field_kubo_toyabe,
)
from asymmetry.core.fitting.strong_collision import EvaluationWorkLimitError


def exact_zero_field(time, width):
    denominator = 1 + (width * time) ** 2
    return 1 / 3 + 2 / 3 * denominator**-1.5 * (1 - time**2 / denominator) * np.exp(
        -(time**2) / (2 * denominator)
    )


def width_average(time, field, width, tolerance=2e-11):
    """Adaptive original-width integral, independent of analytic tilted moments."""

    def conditional(z):
        delta = abs(1 + width * z)
        return (
            np.exp(-z * z / 2)
            / np.sqrt(2 * np.pi)
            * longitudinal_field_kubo_toyabe(time, 1, delta, field / LARMOR_RAD_PER_US_PER_GAUSS)
        )

    bounds = sorted(set([-10.0, 0.0, 10.0, max(-10.0, -1 / width)]))
    return sum(
        quad_vec(conditional, a, b, epsabs=tolerance, epsrel=tolerance)[0]
        for a, b in zip(bounds[:-1], bounds[1:])
    )


def nested_hayano(time, field, width, tolerance):
    """Original Hayano LF sine integral nested in the Gaussian-width integral.

    Hayano et al., Phys. Rev. B 20, 850 (1979), DOI:10.1103/PhysRevB.20.850.
    This reference uses neither the production static kernel nor tilted moments;
    finite fields >=.06 avoid its inverse-field cancellation. Normal tails beyond
    |z|=10 have probability <1.53e-23 and |physical polarisation|<=1.
    """

    def conditional(z):
        delta = abs(1 + width * z)
        sine = quad(
            lambda s: np.exp(-(delta**2) * s**2 / 2),
            0,
            time,
            weight="sin",
            wvar=field,
            epsabs=tolerance / 100,
            epsrel=tolerance / 100,
        )[0]
        value = (
            1
            - 2
            * delta**2
            / field**2
            * (1 - np.exp(-(delta**2) * time**2 / 2) * np.cos(field * time))
            + 2 * delta**4 / field**3 * sine
        )
        return np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi) * value

    bounds = sorted(set([-10.0, 0.0, 10.0, -1 / width]))
    return sum(
        quad(conditional, a, b, epsabs=tolerance, epsrel=tolerance)[0]
        for a, b in zip(bounds[:-1], bounds[1:])
    )


@pytest.mark.parametrize("width", [0.1, 0.3, 0.5, 1.0])
def test_continuous_zero_field_average_through_long_tail(width):
    time = np.linspace(0, 20, 201)
    np.testing.assert_allclose(
        gaussian_broadened_kt(time, 1, 0, width), exact_zero_field(time, width), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("field,width", [(0.06, 1), (1, 1), (10, 0.5), (100, 1)])
def test_independent_long_time_finite_field_average(field, width):
    coarse = nested_hayano(20, field, width, 2e-8)
    fine = nested_hayano(20, field, width, 2e-9)
    assert abs(fine - coarse) < 2e-8
    for delta in (0.05, 1, 5):
        for sign in (-1, 1):
            actual = gaussian_broadened_kt(
                20 / delta, delta, sign * field * delta / LARMOR_RAD_PER_US_PER_GAUSS, width
            )
            assert actual == pytest.approx(fine, abs=1e-5)


@pytest.mark.parametrize("field", [0, 1e-6, 0.06, 1, 100])
def test_zero_broadening_and_nearby_width(field):
    time = np.array([0, 0.13, 1.7, 5, 20])
    b_l = field / LARMOR_RAD_PER_US_PER_GAUSS
    expected = longitudinal_field_kubo_toyabe(time, 1, 1, b_l)
    np.testing.assert_array_equal(gaussian_broadened_kt(time, 1, b_l, 0), expected)
    np.testing.assert_allclose(
        gaussian_broadened_kt(time, 1, b_l, 1e-8), expected, atol=1e-12, rtol=0
    )


def test_shape_sign_order_empty_and_normalisation_contract():
    time = np.array([[20, 0, -1], [5, 1, -0.1]])
    field = 8.4 / LARMOR_RAD_PER_US_PER_GAUSS
    actual = gaussian_broadened_kt(time, -1, -field, -0.3)
    assert actual.shape == time.shape
    expected = np.array([gaussian_broadened_kt(abs(t), 1, field, 0.3) for t in time.flat])
    np.testing.assert_allclose(actual, expected.reshape(time.shape), atol=1e-15, rtol=0)
    assert isinstance(gaussian_broadened_kt(1, 1, field, 0.3), float)
    assert gaussian_broadened_kt(0, 1, field, 1) == 1
    np.testing.assert_array_equal(gaussian_broadened_kt(time, 0, field, 1), np.ones_like(time))
    for shape in [(0,), (2, 0, 3)]:
        assert gaussian_broadened_kt(np.empty(shape), 1, field, 0.3).shape == shape


@pytest.mark.parametrize("field,width", [(8, 1), (8.4, 0.3), (100, 1)])
def test_panel_transition_derivatives(field, width):
    time = np.array([1.0, 20.0])
    h = 1e-4
    for parameter in ("field", "time", "width"):
        differences = []
        for k in (-1, 0, 1):
            t = time * (1 + k * h) if parameter == "time" else time
            r = field + k * h if parameter == "field" else field
            w = width + k * h if parameter == "width" else width
            reference = width_average(t, r, w)
            actual = gaussian_broadened_kt(t, 1, r / LARMOR_RAD_PER_US_PER_GAUSS, w)
            differences.append(actual - reference)
        assert max(abs((differences[2] - differences[0]) / (2 * h))) < 1e-7
        assert max(abs((differences[2] - 2 * differences[1] + differences[0]) / h**2)) < 1e-5


@pytest.mark.parametrize("field,width", [(0.06, 0.3), (1, 1), (10, 0.5)])
def test_fit_recovery_from_adaptive_width_average(field, width):
    time = np.linspace(0, 20, 61)
    truth = np.array([1, field, width])
    expected = width_average(time, field, width)
    np.testing.assert_allclose(
        expected, width_average(time, field, width, 2e-12), atol=1e-9, rtol=0
    )

    def residual(p):
        return (
            gaussian_broadened_kt(time, p[0], p[1] / LARMOR_RAD_PER_US_PER_GAUSS, p[2]) - expected
        ) / 0.001

    fitted = least_squares(
        residual, truth * [0.93, 1.1, 0.9], bounds=([0.5, 1e-8, 1e-6], [2, 20, 1.2])
    )
    assert fitted.success
    errors = np.sqrt(np.diag(np.linalg.inv(fitted.jac.T @ fitted.jac)))
    assert np.all(abs(fitted.x - truth) < 0.1 * errors)
    assert max(abs(fitted.fun * 0.001)) < 1e-5


def test_broadened_work_limit_precedes_quadrature_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("quadrature allocation occurred before the work check")

    # Patch the shared NumPy arange only during an excessive-work request.
    monkeypatch.setattr(np, "arange", forbidden)
    with pytest.raises(
        EvaluationWorkLimitError, match=r"Gaussian.broadened.*2,097,152.*quadrature nodes"
    ):
        gaussian_broadened_kt(np.array([0, 200000]), 1, 1, 0.5)
    with pytest.raises(EvaluationWorkLimitError, match="quadrature nodes"):
        gaussian_broadened_kt(np.zeros(131073), 1, 1, 0.5)


def test_tiny_width_does_not_form_overflowing_field_ratio():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        np.testing.assert_array_equal(
            gaussian_broadened_kt(np.array([0, 0.1, 1]), 1e-310, 1, 0.5), 1
        )


def test_dense_ordinary_time_array_preserves_scalar_values():
    time = np.linspace(0, 5, 65536)
    field = 1 / LARMOR_RAD_PER_US_PER_GAUSS
    actual = gaussian_broadened_kt(time, 1, field, 0.5)
    assert actual.shape == time.shape
    selected = np.array([gaussian_broadened_kt(t, 1, field, 0.5) for t in time[::8191]])
    np.testing.assert_allclose(actual[::8191], selected, atol=1e-15, rtol=0)


def test_work_budget_boundary_and_recovery(monkeypatch):
    from asymmetry.core.fitting import gaussian_width_kt

    monkeypatch.setattr(gaussian_width_kt, "GAUSSIAN_WIDTH_NODE_BUDGET", 64)
    field = 1 / LARMOR_RAD_PER_US_PER_GAUSS
    # Two full panels plus two remainders require exactly 64 GL16 nodes.
    expected = gaussian_broadened_kt(np.array([0, 2]), 1, field, 0.5)
    with pytest.raises(EvaluationWorkLimitError, match="64 quadrature nodes"):
        gaussian_broadened_kt(np.array([0, 3]), 1, field, 0.5)
    np.testing.assert_array_equal(gaussian_broadened_kt(np.array([0, 2]), 1, field, 0.5), expected)
    # Neither analytic limiting path uses the finite-field integration budget.
    assert gaussian_broadened_kt(200000, 1, 0, 0.5) == pytest.approx(1 / 3, abs=1e-12)
    assert np.isfinite(gaussian_broadened_kt(200000, 1, field, 0))


@pytest.mark.parametrize("field", [1e-20, 1e-150, 1e-220, np.nextafter(0.0, 1.0)])
def test_tiny_nonzero_field_retains_continuous_zero_field_limit(field):
    time = np.array([0, 0.1, 1, 2, 20.0])
    actual = gaussian_broadened_kt(time, 1, field / LARMOR_RAD_PER_US_PER_GAUSS, 1)
    np.testing.assert_allclose(actual, exact_zero_field(time, 1), atol=1e-12, rtol=0)


@pytest.mark.parametrize("field", [0.099999, 0.1, 0.100001])
def test_entire_factor_series_boundary_matches_width_oracle(field):
    time = np.array([0.1, 1, 20.0])
    expected = width_average(time, field, 1)
    actual = gaussian_broadened_kt(time, 1, field / LARMOR_RAD_PER_US_PER_GAUSS, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

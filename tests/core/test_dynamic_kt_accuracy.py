"""Independent renewal and similarity regressions for dynamic KT evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import cumulative_simpson, quad
from scipy.optimize import least_squares
from scipy.special import spherical_jn

from asymmetry.core.fitting.models import (
    LARMOR_RAD_PER_US_PER_GAUSS,
    dynamic_gaussian_kt,
    dynamic_lorentzian_kt,
    static_lorentzian_kt_lf,
)


@pytest.mark.parametrize("rate", [11.99, 12.01])
def test_gaussian_former_absolute_rate_switch(rate):
    coarse = scalar_renewal("gaussian", rate / 5, 0, 5, 4001)[::40]
    fine = scalar_renewal("gaussian", rate / 5, 0, 5, 8001)[::80]
    assert np.max(np.abs(coarse - fine)) < 1e-6
    actual = dynamic_gaussian_kt(np.linspace(0, 1, 101), 1, 5, rate, 0)
    np.testing.assert_allclose(actual, fine, atol=1e-5, rtol=0)


def test_lorentzian_dimensional_similarity_above_former_rate_switch():
    t = np.linspace(0, 1, 101)
    actual = dynamic_lorentzian_kt(t, 1, 5, 50, 100)
    expected = dynamic_lorentzian_kt(10 * t, 1, 0.5, 5, 10)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


def test_gaussian_curvature_at_former_grid_rounding_boundary():
    t = np.linspace(0, 12, 200)
    nu = 1.5459360981536812
    step = np.finfo(float).eps ** 0.25 * nu

    def prediction(rate):
        return dynamic_gaussian_kt(t, 18.002089667, 0.390035095681914, rate, 0, 4.00219947)

    base = prediction(nu)
    curvatures = [
        (prediction(nu + h) - 2 * base + prediction(nu - h)) / h**2 for h in (step, step / 2)
    ]
    assert np.linalg.norm(curvatures[0] - curvatures[1]) < 0.001 * np.linalg.norm(curvatures[1])


# These owned values use the published static functions transformed analytically,
# independently inverted at higher precision by two inverse-Laplace algorithms.


_REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/kt_renewal_reference.json").read_text()
)


@pytest.mark.parametrize(
    "case",
    _REFERENCE["cases"],
    ids=lambda c: f"{c['kind']}-nu{c['rate_over_width']}-field{c['omega_over_width']}",
)
def test_dynamic_dimensionless_reference_matrix(case):
    time = np.array(case["dimensionless_time"])
    expected = np.array(case["polarisation"], dtype=float)
    function = dynamic_gaussian_kt if case["kind"] == "gaussian" else dynamic_lorentzian_kt
    for width in (0.05, 0.4, 1.0, 5.0):
        for sign in (-1.0, 1.0):
            actual = function(
                time / width,
                1.0,
                width,
                case["rate_over_width"] * width,
                sign * case["omega_over_width"] * width / LARMOR_RAD_PER_US_PER_GAUSS,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


def published_static(kind, time, ratio):
    """Published Gaussian and spherical-Bessel Lorentzian time expressions."""
    if ratio == 0:
        if kind == "gaussian":
            return 1 / 3 + 2 / 3 * (1 - time**2) * np.exp(-(time**2) / 2)
        return 1 / 3 + 2 / 3 * (1 - time) * np.exp(-time)
    if kind == "gaussian":
        integral = cumulative_simpson(
            np.exp(-(time**2) / 2) * np.sin(ratio * time), x=time, initial=0.0
        )
        return (
            1
            - 2 / ratio**2 * (1 - np.exp(-(time**2) / 2) * np.cos(ratio * time))
            + 2 / ratio**3 * integral
        )
    j0 = np.sinc(ratio * time / np.pi)
    integral = cumulative_simpson(np.exp(-time) * j0, x=time, initial=0.0)
    return (
        1
        - np.exp(-time) * spherical_jn(1, ratio * time) / ratio
        - (np.exp(-time) * j0 - 1) / ratio**2
        - (1 + 1 / ratio**2) * integral
    )


def scalar_renewal(kind, rate, ratio, end, n):
    time = np.linspace(0, end, n)
    h = time[1]
    f = np.exp(-rate * time) * published_static(kind, time, ratio)
    result = np.ones(n)
    for i in range(1, n):
        result[i] = (f[i] + rate * h * (np.dot(f[1:i], result[i - 1 : 0 : -1]) + f[i] / 2)) / (
            1 - rate * h / 2
        )
    return result


@pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
@pytest.mark.parametrize(
    "rate,ratio", [(0.01, 0.06), (0.1, 1), (1, 10), (2.4, 0.06), (10, 1), (100, 10)]
)
def test_independent_scalar_renewal_refinement(kind, rate, ratio):
    end = 5.0 if rate < 100 else 0.2
    curves = [
        scalar_renewal(kind, rate, ratio, end, n)[:: (n - 1) // 100] for n in (4001, 8001, 16001)
    ]
    coarse = (4 * curves[1] - curves[0]) / 3
    fine = (4 * curves[2] - curves[1]) / 3
    np.testing.assert_allclose(coarse, fine, atol=1e-7, rtol=0)
    function = dynamic_gaussian_kt if kind == "gaussian" else dynamic_lorentzian_kt
    actual = function(np.linspace(0, end, 101), 1, 1, rate, ratio / LARMOR_RAD_PER_US_PER_GAUSS)
    np.testing.assert_allclose(actual, fine, atol=1e-5, rtol=0)


@pytest.mark.parametrize("ratio", [0.001, 0.01, 0.03, 0.049, 0.05, 0.051, 0.06, 0.1, 1, 10])
def test_lorentzian_weak_field_against_published_expression(ratio):
    time = np.array([0.0, 0.01, 0.1, 0.5, 1, 2, 5, 8])
    integral = np.array(
        [
            quad(
                lambda s: np.exp(-s) * np.sinc(ratio * s / np.pi), 0, t, epsabs=1e-13, epsrel=1e-13
            )[0]
            for t in time
        ]
    )
    expected = (
        1
        - np.exp(-time) * spherical_jn(1, ratio * time) / ratio
        - (np.exp(-time) * np.sinc(ratio * time / np.pi) - 1) / ratio**2
        - (1 + 1 / ratio**2) * integral
    )
    actual = static_lorentzian_kt_lf(time, 1, 1, ratio / LARMOR_RAD_PER_US_PER_GAUSS)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("kind,rate,ratio", [("gaussian", 2.4, 0), ("lorentzian", 10, 1)])
def test_dynamic_fit_recovers_independent_curve(kind, rate, ratio):
    case = next(
        c
        for c in _REFERENCE["cases"]
        if c["kind"] == kind and c["rate_over_width"] == rate and c["omega_over_width"] == ratio
    )
    function = dynamic_gaussian_kt if kind == "gaussian" else dynamic_lorentzian_kt
    truth = np.array([5.0, 5 * rate])
    time = np.array(case["dimensionless_time"]) / 5
    expected = np.array(case["polarisation"], float)
    sigma = 1e-3
    field = 5 * ratio / LARMOR_RAD_PER_US_PER_GAUSS
    fit = least_squares(
        lambda p: (function(time, 1, p[0], p[1], field) - expected) / sigma,
        truth * np.array([0.95, 1.03]),
        bounds=([0.1, 0.001], [20, 200]),
    )
    covariance = np.linalg.inv(fit.jac.T @ fit.jac)
    assert fit.success
    assert np.all(np.abs(fit.x - truth) < 0.1 * np.sqrt(np.diag(covariance)))
    assert np.max(np.abs(fit.fun * sigma)) < 1e-5


@pytest.mark.parametrize("function", [dynamic_gaussian_kt, dynamic_lorentzian_kt])
def test_dynamic_scalar_shaped_empty_and_zero_width(function):
    times = np.array([[[0.0, 0.01], [1.0, -2.0]], [[0.2, 5.0], [3.0, 0.5]]])
    actual = function(times, 23, 0.4, 15, 3, 2)
    np.testing.assert_array_equal(
        actual, function(times.ravel(), 23, 0.4, 15, 3, 2).reshape(times.shape)
    )
    np.testing.assert_allclose(function(times, 23, 0, 15, 3, 2), 25, atol=1e-12)
    assert np.isscalar(function(0.3, 23, 0.4, 15, 3, 2))
    assert function(np.array([]), 23, 0.4, 15, 3, 2).shape == (0,)


@pytest.mark.parametrize("function", [dynamic_gaussian_kt, dynamic_lorentzian_kt])
@pytest.mark.parametrize("rate", [-1, np.inf, np.nan])
def test_dynamic_kt_rejects_nonphysical_rates(function, rate):
    with pytest.raises(ValueError, match="fluctuation rate nu must be finite and >= 0"):
        function(np.arange(3.0), 1, 1, rate, 0)


@pytest.mark.parametrize("rate", [1e6, 1e12, 1e20])
def test_finite_field_lorentzian_fast_limit_retains_cusp(rate):
    time = np.linspace(0, 5, 101)
    actual = dynamic_lorentzian_kt(time, 1, 1, rate, 1 / LARMOR_RAD_PER_US_PER_GAUSS)
    # The Cauchy tail gives the finite 4a/3 limiting rate, independent of field.
    np.testing.assert_allclose(actual, np.exp(-4 * time / 3), atol=1e-5, rtol=0)


@pytest.mark.parametrize("rate", [0.01, 100, 1e20])
def test_weighted_renewal_preserves_constant_polarisation(rate):
    from asymmetry.core.fitting.strong_collision import strong_collision_polarisation

    actual = strong_collision_polarisation(np.ones(2049), rate, 0.01, initial_derivative=0.0)
    np.testing.assert_allclose(actual, 1.0, atol=1e-10, rtol=0)


def test_width_curvature_when_future_grid_sample_is_added():
    time = np.linspace(0, 12, 200)
    width = 300 / (64 * 12)
    step = np.finfo(float).eps ** 0.25 * width

    def prediction(value):
        return dynamic_gaussian_kt(time, 1, value, 1.55, 0)

    base = prediction(width)
    curvatures = [
        (prediction(width + h) - 2 * base + prediction(width - h)) / h**2 for h in (step, step / 2)
    ]
    assert np.linalg.norm(curvatures[0] - curvatures[1]) < 0.001 * np.linalg.norm(curvatures[1])


def lorentzian_published_field_expression(time, width, omega):
    # Removing the analytic Si term leaves a regular sine-weighted integrand at zero.
    from scipy.special import sici

    result = []
    for t in time:
        if t == 0:
            result.append(1.0)
            continue
        correction = quad(
            lambda s: np.expm1(-width * s) / s if s else -width,
            0,
            t,
            weight="sin",
            wvar=omega,
            epsabs=1e-12,
            limit=300,
        )[0]
        integral = (sici(omega * t)[0] + correction) / omega
        result.append(
            1
            - width / omega * spherical_jn(1, omega * t) * np.exp(-width * t)
            - (width / omega) ** 2 * (np.exp(-width * t) * np.sinc(omega * t / np.pi) - 1)
            - width * (1 + (width / omega) ** 2) * integral
        )
    return np.array(result)


@pytest.mark.parametrize("field", [3000.0, 5000.0])
def test_lorentzian_high_field_narrow_width_has_no_alias_growth(field):
    time = np.array([0.0, 0.01, 0.1, 1.0, 5.0, 20.0])
    width = 0.05
    expected = lorentzian_published_field_expression(
        time, width, field * LARMOR_RAD_PER_US_PER_GAUSS
    )
    actual = static_lorentzian_kt_lf(time, 1, width, field)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("width", [0.05, 1, 10, 50, 100])
def test_lorentzian_static_weak_field_resolves_width(width):
    time = np.array([0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 5]) / width
    field = 1e-6 * width / LARMOR_RAD_PER_US_PER_GAUSS
    expected = 1 / 3 + 2 / 3 * (1 - width * time) * np.exp(-width * time)
    actual = static_lorentzian_kt_lf(time, 1, width, field)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        dynamic_lorentzian_kt(time, 1, width, 1e-20 * width, field), actual, atol=1e-8, rtol=0
    )


@pytest.mark.parametrize("rate", [0.01, 2.4, 10, 100])
def test_lorentzian_field_correction_has_continuous_zero_limit(rate):
    time = np.linspace(0, 5, 101)
    zero = dynamic_lorentzian_kt(time, 1, 1, rate, 0)
    near = dynamic_lorentzian_kt(time, 1, 1, rate, 1e-6 / LARMOR_RAD_PER_US_PER_GAUSS)
    assert np.max(np.abs(near - zero)) < 3e-13
    # Resolve physical field curvature above the floating-point subtraction floor.
    curvatures = []
    for omega in (0.02, 0.01):
        plus = dynamic_lorentzian_kt(time, 1, 1, rate, omega / LARMOR_RAD_PER_US_PER_GAUSS)
        curvatures.append(2 * (plus - zero) / omega**2)
    np.testing.assert_allclose(curvatures[0], curvatures[1], atol=1e-6, rtol=0.002)


def test_dynamic_work_guard_precedes_grid_allocation(monkeypatch):
    from asymmetry.core.fitting import models
    from asymmetry.core.fitting.strong_collision import EvaluationWorkLimitError

    def forbidden_allocation(*args, **kwargs):
        pytest.fail("grid allocated before checking the evaluation-work budget")

    monkeypatch.setattr(models.np, "arange", forbidden_allocation)
    with pytest.raises(EvaluationWorkLimitError, match=r"requires .*samples.*524,288"):
        dynamic_gaussian_kt(np.array([0, 20]), 1, 1, 1, 20000)
    assert models._dynamic_kt_spline.cache_info().maxsize == 16
    assert models._static_lorentzian_lf_grid.cache_info().maxsize == 16

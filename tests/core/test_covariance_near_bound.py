"""Synthetic exponential fits with independently differentiated observed information."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.asymmetry_global import fit_global
from asymmetry.core.fitting.global_covariance import _member_information
from asymmetry.core.fitting.likelihood import GaussianObservations
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _exponential(t, background, amplitude, rate):
    return background + amplitude * np.exp(-rate * t)


def _member_derivatives(t, amplitude, rate):
    decaying = np.exp(-rate * t)
    jacobian = np.column_stack((np.ones_like(t), decaying, -amplitude * t * decaying))
    hessian = np.zeros((len(t), 3, 3))
    hessian[:, 1, 2] = hessian[:, 2, 1] = -t * decaying
    hessian[:, 2, 2] = amplitude * t**2 * decaying
    return jacobian, hessian


def _member_columns(index):
    return [0, 2 * index + 1, 2 * index + 2]


@pytest.fixture
def stationary_exponentials():
    t = np.linspace(0, 5, 51)
    values = np.array([0.7, 1.2, 1e-8, 0.9, 0.6])
    design = np.zeros((2 * len(t), len(values)))
    model = np.empty((2, len(t)))
    for index, line in enumerate(model):
        background, amplitude, rate = values[_member_columns(index)]
        line[:] = _exponential(t, background, amplitude, rate)
        rows = slice(index * len(t), (index + 1) * len(t))
        design[rows, _member_columns(index)] = _member_derivatives(t, amplitude, rate)[0]
    # Orthogonality to the joint Jacobian gives a stationary point with nonzero residuals.
    residual = np.random.default_rng(42).normal(0, 0.05, model.size)
    residual -= design @ np.linalg.lstsq(design, residual, rcond=None)[0]
    np.testing.assert_allclose(design.T @ (residual / 0.05**2), 0, atol=1e-8)
    datasets = {
        run: MuonDataset(t, y, np.full_like(t, 0.05), metadata={"run_number": run})
        for run, y in enumerate(model + residual.reshape(model.shape), start=1)
    }
    return datasets, values


def _observed_information(t, y, sigma, parameters):
    _, amplitude, rate = parameters
    jacobian, hessian = _member_derivatives(t, amplitude, rate)
    weighted_jacobian = jacobian / sigma[:, None]
    weighted_residual = (_exponential(t, *parameters) - y) / sigma**2
    return weighted_jacobian.T @ weighted_jacobian + np.einsum(
        "t,tij->ij", weighted_residual, hessian
    )


def _fit_exponentials(datasets, starts, rate_min, strategy="least_squares"):
    seeds = {
        run: ParameterSet(
            [
                Parameter("background", starts[0], min=0, max=10),
                Parameter("amplitude", starts[2 * run - 1], min=0, max=10),
                Parameter("rate", starts[2 * run], min=rate_min, max=10),
            ]
        )
        for run in datasets
    }
    return fit_global(
        datasets,
        _exponential,
        global_params=["background"],
        local_params=["amplitude", "rate"],
        initial_params=seeds,
        strategy=strategy,
    )


@pytest.mark.parametrize(
    "amplitude_bounds, rate_bounds, time_scale",
    [
        ((0, 10), (-0.1, 10), 1),
        ((0, 10), (0, 10), 1),
        ((0, 1.2 + 1e-8), (0, 10), 1),
        ((0, 10), (0, 1e-5), 1),
        ((0, 10), (2e-9, 2e-8), 1e4),
    ],
    ids=["centred", "lower", "opposite_directions", "narrow", "endpoint_roundoff"],
)
def test_member_information_matches_observed_curvature_inside_bounds(
    stationary_exponentials, amplitude_bounds, rate_bounds, time_scale
):
    datasets, values = stationary_exponentials
    dataset = datasets[1]
    bounds = np.array([(0, 10), amplitude_bounds, rate_bounds])
    parameters = values[:3]
    time = dataset.time * time_scale
    residual = dataset.asymmetry - _exponential(dataset.time, *parameters)
    # Away from stationarity the exponential has a nonzero residual-weighted mixed derivative.
    record = GaussianObservations(
        time, _exponential(time, *parameters) + residual + 0.025, dataset.error
    )

    def evaluate(point):
        assert np.all(bounds[:, 0] < point) and np.all(point < bounds[:, 1])
        return _exponential(record.time, *point)

    expected = _observed_information(record.time, record.values, record.sigma, parameters)
    actual = _member_information(evaluate, parameters, bounds, record)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    for estimate in (actual.extrapolated, actual.fine):
        assert np.max(abs(estimate - expected) / scale) < 1e-5


def test_narrow_centred_interval_preserves_resolvable_linear_curvature(stationary_exponentials):
    datasets, values = stationary_exponentials
    parameters = values[:3]
    dataset = datasets[1]
    time = dataset.time
    data = _exponential(time, *parameters)
    record = GaussianObservations(time, data, dataset.error)
    background = parameters[0]
    bounds = [(background - 3.6e-8, background + 3.6e-8), (0, 10), (-0.1, 10)]
    actual = _member_information(lambda p: _exponential(time, *p), parameters, bounds, record)
    expected = _observed_information(time, data, dataset.error, parameters)
    np.testing.assert_allclose(
        [actual.extrapolated[0, 0], actual.fine[0, 0]], expected[0, 0], rtol=1e-7
    )


def test_near_bound_global_covariance_matches_analytic_marginals(stationary_exponentials):
    datasets, values = stationary_exponentials
    information = np.zeros((len(values), len(values)))
    for index, dataset in enumerate(datasets.values()):
        columns = _member_columns(index)
        information[np.ix_(columns, columns)] += _observed_information(
            dataset.time, dataset.asymmetry, dataset.error, values[columns]
        )
    scaled = information / np.sqrt(np.outer(np.diag(information), np.diag(information)))
    eigenvalues = np.linalg.eigvalsh(scaled)
    assert eigenvalues[0] > 0.01
    assert eigenvalues[-1] / eigenvalues[0] < 250
    expected = np.linalg.inv(information)

    fits = [_fit_exponentials(datasets, values, rate_min) for rate_min in (-0.1, 0.0)]
    assert fits[0].chi_squared == pytest.approx(fits[1].chi_squared, abs=1e-10, rel=0)
    for fit in fits:
        assert fit.success
        for index, member in enumerate(fit.dataset_results.values()):
            columns = _member_columns(index)
            assert member.covariance_parameters == ["background", "amplitude", "rate"]
            fitted = [member.parameters[name].value for name in member.covariance_parameters]
            # The absolute tolerance controls the rate near zero. The relative
            # term allows optimizer roundoff for the order-one estimates.
            np.testing.assert_allclose(fitted, values[columns], atol=1e-10, rtol=2e-10)
            assert member.covariance_accurate
            block = expected[np.ix_(columns, columns)]
            scale = np.sqrt(np.outer(np.diag(block), np.diag(block)))
            assert np.max(abs(member.covariance - block) / scale) < 1e-5


def test_downhill_direction_towards_bound_can_end_at_interior_minimum(stationary_exponentials):
    datasets, values = stationary_exponentials
    values[2] = 1.1e-8
    fit = _fit_exponentials(datasets, values, rate_min=0)
    assert fit.success
    assert all(member.covariance_accurate for member in fit.dataset_results.values())


@pytest.mark.parametrize(
    "strategy, rate_min",
    [("least_squares", 0.0001), ("least_squares", 0.1), ("profiled", 0.01), ("profiled", 0.1)],
)
def test_active_rate_bound_withholds_symmetric_covariance(
    stationary_exponentials, strategy, rate_min
):
    datasets, values = stationary_exponentials
    values[2] = rate_min + 1e-8
    fit = _fit_exponentials(datasets, values, rate_min, strategy)
    assert fit.success
    rate = fit.dataset_results[1].parameters["rate"].value
    assert rate == pytest.approx(rate_min, abs=2e-8, rel=0)
    for member in fit.dataset_results.values():
        assert not member.covariance_accurate
        assert member.covariance is None
        assert "Marginal covariance unavailable: parameter at a bound for symmetric covariance" in (
            member.warnings
        )

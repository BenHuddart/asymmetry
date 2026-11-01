"""Likelihood boundaries and independent limiting values for fit objectives."""

import numpy as np
import pytest

from asymmetry.core.fitting.likelihood import (
    CountSource,
    FitInputError,
    GaussianObservations,
    PoissonCounts,
    poisson_deviance,
)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_gaussian_boundary_names_run_point_and_time(sigma):
    errors = np.ones(5)
    errors[3] = sigma
    with pytest.raises(FitInputError) as error:
        GaussianObservations(np.arange(5) / 10, np.ones(5), errors, run_number=7)
    assert str(error.value) == (
        f"run 7, point 3 (t = 0.300 µs): uncertainty {sigma:g} is not finite and positive; "
        "Gaussian fits need finite positive errors"
    )


@pytest.mark.parametrize("counts", [[1, -1], [1, 0.5], [1, float("nan")]])
def test_poisson_rejects_ineligible_counts(counts):
    with pytest.raises(FitInputError):
        PoissonCounts([0, 1], counts, CountSource.RAW, run_number=7)


@pytest.mark.parametrize("source", [CountSource.DEADTIME_CORRECTED, CountSource.LIFETIME_CORRECTED])
def test_poisson_rejects_corrected_counts_even_when_integral(source):
    with pytest.raises(FitInputError, match="Poisson likelihood needs raw counts"):
        PoissonCounts([0, 1], [1, 2], source, run_number=7)


def test_observations_own_read_only_buffers():
    values = np.ones(3)
    gaussian = GaussianObservations(np.arange(3), values, values)
    poisson = PoissonCounts(np.arange(3), values, CountSource.RAW)
    values[:] = 0
    assert np.array_equal(gaussian.values, np.ones(3))
    assert np.array_equal(poisson.counts, np.ones(3))
    for array in [gaussian.time, gaussian.values, gaussian.sigma, poisson.time, poisson.counts]:
        assert not array.flags.writeable


@pytest.mark.parametrize("n,mu,expected", [(0, 0, 0), (0, 2.5, 5), (3, 3, 0), (0, -1, 2)])
def test_poisson_deviance_exact_limits(n, mu, expected):
    assert poisson_deviance(np.array([n]), np.array([mu])) == expected


def test_poisson_constant_rate_deviance_matches_analytic_value():
    assert poisson_deviance(np.array([0, 0, 3, 1, 0, 2]), np.ones(6)) == pytest.approx(
        9.36426245424844, abs=1e-12
    )


@pytest.mark.parametrize(
    "entry", ["fit", "arrays", "joint", "profiled", "least_squares", "wrapper", "joint_blocks"]
)
@pytest.mark.parametrize("defect", [0.0, -1.0, float("nan"), float("inf")])
def test_gaussian_entries_share_the_boundary(entry, defect):
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    time = np.arange(5) / 10
    errors = np.ones(5)
    errors[3] = defect
    dataset = MuonDataset(time=time, asymmetry=np.ones(5), error=errors, metadata={"run_number": 7})
    parameters = ParameterSet([Parameter("c", 1.0)])

    def model(t, c):
        return np.full_like(t, c)

    engine = FitEngine()
    with pytest.raises(FitInputError) as error:
        if entry == "fit":
            engine.fit(dataset, model, parameters)
        elif entry == "arrays":
            engine.fit_arrays(time, dataset.asymmetry, errors, model, parameters, run_number=7)
        elif entry == "wrapper":
            from asymmetry.core.fitting.asymmetry_global import fit_global

            fit_global(
                [dataset], model, global_params=["c"], local_params=[], initial_params=parameters
            )
        elif entry == "joint_blocks":
            from asymmetry.core.fitting.engine import CoupledBlock

            block = CoupledBlock(
                key="test",
                datasets=[dataset],
                model_fn=model,
                global_params=["c"],
                local_params=[],
                initial_params={7: parameters},
                free_global_params=["c"],
                local_group_key=lambda name, run: run,
                t_min=None,
                t_max=None,
            )
            engine.joint_fit([block], [])
        else:
            engine.global_fit(
                [dataset],
                model,
                global_params=["c"],
                local_params=[],
                initial_params={7: parameters},
                strategy=entry,
            )
    assert str(error.value) == (
        f"run 7, point 3 (t = 0.300 µs): uncertainty {defect:g} is not finite and positive; "
        "Gaussian fits need finite positive errors"
    )


def test_poisson_constant_rate_mle_and_objective():
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.likelihood import Objective
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    data = PoissonCounts(np.arange(6), [0, 0, 3, 1, 0, 2], CountSource.RAW)

    def constant(t, c):
        return np.full_like(t, c, dtype=float)

    result = FitEngine().fit_observations(data, constant, ParameterSet([Parameter("c", 2.0)]))
    assert result.success
    assert result.objective is Objective.POISSON
    assert result.minimum_expected_count == pytest.approx(1.0, abs=1e-3)
    assert result.parameters["c"].value == pytest.approx(1.0, abs=1e-3)
    assert result.uncertainties["c"] == pytest.approx(np.sqrt(1 / 6), rel=1e-3)
    assert result.chi_squared == pytest.approx(9.36426245424844, abs=1e-5)


def test_poisson_invalid_optimum_fails_with_bin_context():
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    data = PoissonCounts([0.0, 0.1], [1, 2], CountSource.RAW, run_number=7)

    def constant(t, c):
        return np.full_like(t, c, dtype=float)

    result = FitEngine().fit_observations(
        data, constant, ParameterSet([Parameter("c", -1.0, fixed=True)])
    )
    assert not result.success
    assert "run 7, point 0" in result.message
    assert "bins [0, 1]" in result.message
    assert result.minimum_expected_count is None


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
def test_shared_constant_matches_analytic_weighted_mean(strategy):
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    time = np.arange(4.0)
    values = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 2.0, 2.0])]
    errors = [np.array([1.0, 1.0, 2.0, 2.0]), np.ones(4)]
    datasets = [
        MuonDataset(time, y, sigma, {"run_number": run})
        for run, (y, sigma) in enumerate(zip(values, errors, strict=True), start=1)
    ]
    initial = {run: ParameterSet([Parameter("c", 1.8)]) for run in [1, 2]}

    def constant(t, c):
        return np.full_like(t, c)

    results, shared = FitEngine().global_fit(
        datasets, constant, ["c"], [], initial, strategy=strategy
    )
    assert all(result.success for result in results.values())
    assert shared["c"].value == pytest.approx(1.9615384615384615, abs=1e-3 * 0.3922322702763681)
    assert results[1].uncertainties["c"] == pytest.approx(0.3922322702763681, rel=1e-6)
    assert sum(result.chi_squared for result in results.values()) == pytest.approx(
        2.2403846153846145, rel=1e-9
    )


def test_poisson_trial_continuation_is_smooth_and_restoring():
    from asymmetry.core.fitting.likelihood import _POISSON_CONTINUATION_FRACTION

    n = np.array([3.0])
    join = 3 * _POISSON_CONTINUATION_FRACTION
    step = join * 1e-5

    def cost(mu):
        return poisson_deviance(n, np.array([mu]))

    left = (cost(join) - cost(join - step)) / step
    right = (cost(join + step) - cost(join)) / step
    assert left == pytest.approx(right, rel=3e-5)
    assert left < 0
    costs = [cost(mu) for mu in [-1, -0.1, 0, join / 2, join]]
    assert all(a > b for a, b in zip(costs, costs[1:]))


@pytest.mark.parametrize("entry", ["single", "fb", "grouped"])
@pytest.mark.parametrize("defect", ["negative", "fractional", "deadtime"])
def test_count_entries_reject_ineligible_poisson_data(entry, defect):
    from asymmetry.core.fitting.count_domain import fit_fb_alpha, fit_single_histogram
    from asymmetry.core.fitting.grouped_time_domain import (
        build_grouped_time_domain_groups,
        fit_grouped_time_domain,
    )
    from tests.core.test_count_domain_fits import _fb_seed_params, _pulsed_tf_run, _tf
    from tests.core.test_grouped_time_domain_fitting import (
        _cosine_polarization,
        _initial_group_params,
    )

    dataset = _pulsed_tf_run()
    run = dataset.run
    if defect == "deadtime":
        run.grouping.update(
            deadtime_correction=True,
            deadtime_mode="manual",
            dead_time_us=[0.01] * len(run.histograms),
            good_frames=1e5,
        )
    else:
        for histogram in run.histograms:
            histogram.counts[:] = -1.0 if defect == "negative" else 0.1234567
    with pytest.raises(FitInputError) as error:
        if entry == "single":
            fit_single_histogram(dataset, 1, _tf, _fb_seed_params(n0=1e5), cost="poisson")
        elif entry == "fb":
            fit_fb_alpha(dataset, 1, 2, _tf, _fb_seed_params(n0=1e5), cost="poisson")
        else:
            groups = build_grouped_time_domain_groups(dataset, lifetime_corrected=False)
            fit_grouped_time_domain(
                groups,
                _cosine_polarization,
                global_params=["frequency"],
                local_params=["N0", "background", "amplitude", "relative_phase"],
                initial_params={group.group_id: _initial_group_params() for group in groups},
                cost="poisson",
            )
    if defect == "deadtime":
        assert "deadtime-corrected" in str(error.value)
        assert "DT₀ count-domain model or the Gaussian cost" in str(error.value)


def test_global_result_uses_checked_observations_when_caller_buffers_change():
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    time = np.arange(4.0)
    datasets = [
        MuonDataset(time.copy(), time + intercept, np.ones(4), {"run_number": run})
        for run, intercept in [(1, 2.0), (2, 4.0)]
    ]
    initial = {run: ParameterSet([Parameter("c", 1.0)]) for run in [1, 2]}
    calls = 0

    def model(t, c):
        nonlocal calls
        if calls == 0:
            for dataset in datasets:
                dataset.time[:] += 1000
                dataset.asymmetry[:] = 100
                dataset.error[:] = 0
        calls += 1
        return t + c

    results, shared = FitEngine().global_fit(datasets, model, ["c"], [], initial)
    assert all(result.success for result in results.values())
    assert shared["c"].value == pytest.approx(3.0, abs=1e-5)
    assert sum(result.chi_squared for result in results.values()) == pytest.approx(8.0, abs=1e-8)
    assert results[1].residuals == pytest.approx(np.full(4, -1.0), abs=1e-5)

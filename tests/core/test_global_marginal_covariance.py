"""Global error blocks agree with an independently assembled linear design."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
@pytest.mark.parametrize("centre", [0.0, 1.0])
@pytest.mark.parametrize("runs", [1, 2, 5])
def test_global_member_covariance_matches_design(strategy, centre, runs):
    x = np.linspace(-2, 2, 21) + centre
    rng = np.random.default_rng(20261031)
    datasets, seeds, designs, sigmas = [], {}, [], []
    for run in range(runs):
        sigma = np.linspace(0.1, 0.3, len(x)) * (1 + run / 3)
        datasets.append(
            MuonDataset(
                time=x,
                asymmetry=0.7 * x + 2 + run + rng.normal(0, sigma),
                error=sigma,
                metadata={"run_number": run + 1},
            )
        )
        seeds[run + 1] = ParameterSet([Parameter("m", 0.5), Parameter("b", 2)])
        design = np.zeros((len(x), 1 + runs))
        design[:, 0], design[:, run + 1] = x, 1
        designs.append(design)
        sigmas.append(sigma)
    weighted = np.concatenate(designs) / np.concatenate(sigmas)[:, None]
    oracle = np.linalg.inv(weighted.T @ weighted)

    def model(t, **p):
        return p["m"] * t + p["b"]

    results, _ = FitEngine().global_fit(
        datasets,
        model,
        ["m"],
        ["b"],
        seeds,
        strategy=strategy,
    )
    for run, result in results.items():
        assert result.success
        assert result.covariance_accurate
        assert result.covariance_parameters == ["m", "b"]
        expected = oracle[np.ix_([0, run], [0, run])]
        np.testing.assert_allclose(result.covariance, expected, rtol=2e-3, atol=1e-8)
        np.testing.assert_allclose(
            [result.uncertainties[name] for name in ("m", "b")],
            np.sqrt(np.diag(expected)),
            rtol=2e-3,
        )
        # The sum requires the within-member cross term, not just diagonal σ.
        np.testing.assert_allclose(
            np.ones(2) @ result.covariance @ np.ones(2), expected.sum(), rtol=2e-3
        )


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
@pytest.mark.parametrize("fixed_global", [False, True])
def test_fixed_member_and_grouped_local_columns(strategy, fixed_global):
    x = np.linspace(-1, 3, 31)
    datasets, seeds, designs = [], {}, []
    # g: free shared slope except the first member; q: fixed global;
    # b: local intercept, tied in runs 2/3; c: free local quadratic in run 1.
    for run in (1, 2, 3):
        seeds[run] = ParameterSet(
            [
                Parameter("g", 0.7, fixed=fixed_global or run == 1),
                Parameter("q", 0.2, fixed=True),
                Parameter("b", 2.0),
                Parameter("c", 0.1, fixed=run != 1),
            ]
        )
        datasets.append(
            MuonDataset(
                time=x,
                asymmetry=0.7 * x + 0.2 * x**3 + 2 + 0.1 * x**2,
                error=np.full(len(x), 0.2),
                metadata={"run_number": run},
            )
        )
        design = np.zeros((len(x), 3 if fixed_global else 4))
        shift = 0 if fixed_global else 1
        if not fixed_global and run != 1:
            design[:, 0] = x
        design[:, shift + (0 if run == 1 else 2)] = 1
        if run == 1:
            design[:, shift + 1] = x**2
        designs.append(design)
    design = np.concatenate(designs) / 0.2
    oracle = np.linalg.inv(design.T @ design)

    def model(t, **p):
        return p["g"] * t + p["q"] * t**3 + p["b"] + p["c"] * t**2

    results, _ = FitEngine().global_fit(
        datasets,
        model,
        ["g", "q"],
        ["b", "c"],
        seeds,
        strategy=strategy,
        local_param_groups={"b": {1: 1, 2: 2, 3: 2}},
    )
    for run, result in results.items():
        assert result.success
        expected_names = (["g"] if not fixed_global and run != 1 else []) + ["b"]
        expected_columns = ([0] if not fixed_global and run != 1 else []) + [
            (0 if fixed_global else 1) + (0 if run == 1 else 2)
        ]
        if run == 1:
            expected_names.append("c")
            expected_columns.append(1 if fixed_global else 2)
        assert result.covariance_parameters == expected_names
        assert set(result.uncertainties) == set(expected_names)
        np.testing.assert_allclose(
            result.covariance,
            oracle[np.ix_(expected_columns, expected_columns)],
            rtol=3e-3,
            atol=1e-8,
        )


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
def test_nonlinear_observed_curvature_retains_residual_second_derivative(strategy):
    x = np.linspace(-1, 1, 41)
    mean = np.exp(0.5 * x)
    derivative = x * mean
    basis = np.column_stack([np.ones(len(x)), derivative])
    shape = x * x * mean
    residual = shape - basis @ np.linalg.lstsq(basis, shape, rcond=None)[0]
    residual *= 0.3 / np.max(abs(residual))
    datasets = [
        MuonDataset(
            time=x,
            asymmetry=mean + run + residual,
            error=np.full(len(x), 0.2),
            metadata={"run_number": run},
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("a", 0.5), Parameter("b", run)]) for run in (1, 2)}

    def model(t, **p):
        return np.exp(p["a"] * t) + p["b"]

    results, _ = FitEngine().global_fit(datasets, model, ["a"], ["b"], seeds, strategy=strategy)
    information = np.zeros((3, 3))
    for run, result in results.items():
        assert result.success
        exponential = np.exp(result.parameters["a"].value * x)
        jacobian = np.column_stack([x * exponential, np.ones(len(x))])
        member = jacobian.T @ jacobian / 0.2**2
        member[0, 0] -= (
            np.sum(
                (
                    datasets[run - 1].asymmetry
                    - model(x, **{p.name: p.value for p in result.parameters})
                )
                * x
                * x
                * exponential
            )
            / 0.2**2
        )
        indices = [0, run]
        information[np.ix_(indices, indices)] += member
    oracle = np.linalg.inv(information)
    for run, result in results.items():
        np.testing.assert_allclose(
            result.covariance, oracle[np.ix_([0, run], [0, run])], rtol=3e-3, atol=1e-7
        )


@pytest.mark.parametrize("strategy", ["joint", "profiled"])
def test_poisson_observed_likelihood_curvature(strategy):
    from asymmetry.core.fitting.likelihood import CountSource, PoissonCounts

    x = np.linspace(-1, 1, 31)
    rng = np.random.default_rng(424)
    datasets, observations, seeds = [], {}, {}
    for run in (1, 2, 3):
        counts = rng.poisson((10 + run) * np.exp(0.4 * x)).astype(float)
        datasets.append(
            MuonDataset(
                time=x, asymmetry=counts, error=np.ones(len(x)), metadata={"run_number": run}
            )
        )
        observations[run] = PoissonCounts(x, counts, CountSource.RAW, run_number=run)
        seeds[run] = ParameterSet([Parameter("a", 0.4), Parameter("b", 10 + run, min=0.1)])

    def model(t, **p):
        return p["b"] * np.exp(p["a"] * t)

    results, _ = FitEngine().global_fit(
        datasets, model, ["a"], ["b"], seeds, strategy=strategy, observations=observations
    )
    information = np.zeros((4, 4))
    for run, result in results.items():
        assert result.success
        a, b = (result.parameters[name].value for name in ("a", "b"))
        mu = model(x, a=a, b=b)
        n = observations[run].counts
        # Analytic Hessian of Σ(μ−n log μ), including the b coordinate's curvature.
        member = np.array(
            [[np.sum(mu * x * x), np.sum(mu * x) / b], [np.sum(mu * x) / b, np.sum(n) / b**2]]
        )
        information[np.ix_([0, run], [0, run])] += member
    oracle = np.linalg.inv(information)
    for run, result in results.items():
        assert result.covariance_accurate
        np.testing.assert_allclose(
            result.covariance, oracle[np.ix_([0, run], [0, run])], rtol=4e-3, atol=1e-6
        )


@pytest.mark.parametrize("strategy", ["profiled", "least_squares"])
def test_unsupported_minos_is_explicit_even_when_globals_fixed(strategy):
    x = np.arange(5, dtype=float)
    dataset = MuonDataset(time=x, asymmetry=x, error=np.ones(5), metadata={"run_number": 1})
    with pytest.raises(NotImplementedError, match="MINOS is not available"):
        FitEngine().global_fit(
            [dataset],
            lambda t, **p: p["m"] * t + p["b"],
            ["m"],
            ["b"],
            {1: ParameterSet([Parameter("m", 1, fixed=True), Parameter("b", 0)])},
            strategy=strategy,
            minos=True,
        )


@pytest.mark.parametrize("strategy", ["profiled", "least_squares"])
def test_singular_curvature_never_falls_back_to_conditional_errors(strategy):
    x = np.arange(11, dtype=float)
    datasets = [
        MuonDataset(
            time=x, asymmetry=np.full(11, 3 + run), error=np.ones(11), metadata={"run_number": run}
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("m", 1), Parameter("b", 2 + run)]) for run in (1, 2)}
    results, _ = FitEngine().global_fit(
        datasets,
        lambda t, **p: np.full_like(t, p["m"] + p["b"]),
        ["m"],
        ["b"],
        seeds,
        strategy=strategy,
    )
    for result in results.values():
        assert all(np.isfinite(p.value) for p in result.parameters)
        assert not result.covariance_accurate
        assert result.covariance is None
        assert not result.uncertainties
        assert any("covariance unavailable" in message for message in result.warnings)


@pytest.mark.parametrize("strategy", ["profiled", "least_squares"])
def test_covariance_work_remains_cooperatively_cancellable(monkeypatch, strategy):
    from asymmetry.core.fitting import engine

    x = np.linspace(-1, 3, 21)
    datasets = [
        MuonDataset(
            time=x, asymmetry=0.7 * x + run, error=np.ones(len(x)), metadata={"run_number": run}
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("m", 0.5), Parameter("b", run)]) for run in (1, 2)}
    calculating = False
    calls = 0
    original = engine.observed_covariance

    def covariance(*args):
        nonlocal calculating
        calculating = True
        return original(*args)

    def cancelled():
        nonlocal calls
        if calculating:
            calls += 1
        return calls >= 3

    monkeypatch.setattr(engine, "observed_covariance", covariance)
    with pytest.raises(engine.FitCancelledError):
        FitEngine().global_fit(
            datasets,
            lambda t, **p: p["m"] * t + p["b"],
            ["m"],
            ["b"],
            seeds,
            strategy=strategy,
            cancel_callback=cancelled,
        )
    assert calls == 3


def test_historical_errors_keep_their_execution_identity():
    from asymmetry.core import computation
    from asymmetry.core.fitting.likelihood import Objective
    from asymmetry.core.fitting.result_payload import _deserialize_fit_result, _serialize_fit_result
    from asymmetry.core.fitting.result_summary import fit_result_summary
    from asymmetry.core.fitting.results import FitResult

    for revision in (None, "2", "3", "4", computation.COMPUTATION_REVISION):
        result = FitResult(
            success=True,
            objective=Objective.GAUSSIAN,
            computation_revision=revision,
            uncertainties={"b": 0.043643578},
        )
        restored = _deserialize_fit_result(_serialize_fit_result(result))
        assert restored.computation_revision == revision
        assert restored.uncertainties == result.uncertainties
        summary = fit_result_summary(restored)
        assert summary["computation_revision"] == revision
        assert summary["uncertainties"] == result.uncertainties
        assert computation.is_current(revision) == (revision == computation.COMPUTATION_REVISION)


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
def test_multiple_global_and_local_covariance_blocks(strategy):
    x = np.linspace(-1, 2, 51)
    rng = np.random.default_rng(42)
    designs, datasets, seeds = [], [], {}
    for run in range(3):
        design = np.zeros((len(x), 8))
        design[:, :2] = np.column_stack([x, x * x])
        design[:, 2 + 2 * run : 4 + 2 * run] = np.column_stack([np.ones(len(x)), x**3])
        designs.append(design)
        values = 0.7 * x + 0.2 * x * x + run + 0.05 * x**3 + rng.normal(0, 0.2, len(x))
        datasets.append(
            MuonDataset(
                time=x, asymmetry=values, error=np.full(len(x), 0.2), metadata={"run_number": run}
            )
        )
        seeds[run] = ParameterSet(
            [Parameter("m", 0.7), Parameter("q", 0.2), Parameter("b", run), Parameter("c", 0.05)]
        )
    weighted = np.concatenate(designs) / 0.2
    oracle = np.linalg.inv(weighted.T @ weighted)

    def model(t, **p):
        return p["m"] * t + p["q"] * t * t + p["b"] + p["c"] * t**3

    results, _ = FitEngine().global_fit(
        datasets, model, ["m", "q"], ["b", "c"], seeds, strategy=strategy
    )
    for run, result in results.items():
        assert result.success
        assert result.covariance_parameters == ["m", "q", "b", "c"]
        columns = [0, 1, 2 + 2 * run, 3 + 2 * run]
        np.testing.assert_allclose(
            result.covariance, oracle[np.ix_(columns, columns)], rtol=4e-3, atol=1e-8
        )


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
def test_fixed_globals_have_the_true_conditional_limit(strategy):
    x = np.linspace(-1, 3, 21)
    dataset = MuonDataset(
        time=x, asymmetry=0.7 * x + 2, error=np.full(len(x), 0.2), metadata={"run_number": 1}
    )
    seeds = {1: ParameterSet([Parameter("m", 0.7, fixed=True), Parameter("b", 1)])}
    result = FitEngine().global_fit(
        [dataset], lambda t, **p: p["m"] * t + p["b"], ["m"], ["b"], seeds, strategy=strategy
    )[0][1]
    assert result.success
    assert set(result.uncertainties) == {"b"}
    assert result.uncertainties["b"] == pytest.approx(0.2 / np.sqrt(len(x)), rel=1e-4)


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
def test_a_fixed_global_member_keeps_its_own_value(strategy):
    x = np.linspace(-1, 3, 31)
    datasets, seeds, designs = [], {}, []
    for run in (1, 2):
        slope = -0.3 if run == 1 else 0.7
        datasets.append(
            MuonDataset(
                time=x,
                asymmetry=slope * x + run,
                error=np.full(len(x), 0.2),
                metadata={"run_number": run},
            )
        )
        seeds[run] = ParameterSet(
            [Parameter("m", slope if run == 1 else 0.5, fixed=run == 1), Parameter("b", 0)]
        )
        design = np.zeros((len(x), 3))
        if run == 2:
            design[:, 0] = x
        design[:, run] = 1
        designs.append(design)
    weighted = np.concatenate(designs) / 0.2
    oracle = np.linalg.inv(weighted.T @ weighted)
    results, shared = FitEngine().global_fit(
        datasets, lambda t, **p: p["m"] * t + p["b"], ["m"], ["b"], seeds, strategy=strategy
    )
    assert shared["m"].value == pytest.approx(0.7, abs=1e-3)
    assert results[1].parameters["m"].value == -0.3
    assert results[1].parameters["m"].fixed
    for run, result in results.items():
        assert result.success
        columns = [0, run] if run == 2 else [run]
        assert result.covariance_parameters == (["m", "b"] if run == 2 else ["b"])
        np.testing.assert_allclose(
            result.covariance, oracle[np.ix_(columns, columns)], rtol=2e-3, atol=1e-8
        )

    from asymmetry.core.fitting.asymmetry_global import fit_global

    public = fit_global(
        dict(zip((1, 2), datasets)),
        lambda t, **p: p["m"] * t + p["b"],
        global_params=["m"],
        local_params=["b"],
        initial_params=seeds,
        strategy=strategy,
    )
    assert public.success
    assert public.dof == 2 * len(x) - 3
    assert public.global_uncertainties["m"] == pytest.approx(np.sqrt(oracle[0, 0]), rel=2e-3)


def test_failed_member_solution_does_not_inherit_outer_covariance(monkeypatch):
    from asymmetry.core.fitting.likelihood import Objective
    from asymmetry.core.fitting.results import FitResult

    x = np.linspace(-1, 3, 21)
    datasets = [
        MuonDataset(
            time=x, asymmetry=0.7 * x + run, error=np.ones(len(x)), metadata={"run_number": run}
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("m", 0.5), Parameter("b", run)]) for run in (1, 2)}

    def failed(*args, **kwargs):
        return FitResult(
            success=False, objective=Objective.GAUSSIAN, message="Synthetic member failure"
        )

    monkeypatch.setattr(FitEngine, "_fit_core", failed)
    results, _ = FitEngine().global_fit(
        datasets, lambda t, **p: p["m"] * t + p["b"], ["m"], ["b"], seeds, strategy="profiled"
    )
    for result in results.values():
        assert not result.success
        assert not result.covariance_accurate
        assert result.covariance is None
        assert not result.uncertainties
        assert any("invalid outer or member solution" in message for message in result.warnings)


def test_shared_slope_induces_cross_member_intercept_covariance():
    from asymmetry.core.fitting.engine import _build_coupled_global_problem
    from asymmetry.core.fitting.global_covariance import observed_covariance
    from asymmetry.core.fitting.likelihood import dataset_observations

    x = np.linspace(-1, 3, 21)
    datasets = [
        MuonDataset(
            time=x,
            asymmetry=0.7 * x + run,
            error=np.full(len(x), 0.2),
            metadata={"run_number": run},
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("m", 0.7), Parameter("b", run)]) for run in (1, 2)}
    problem = _build_coupled_global_problem(
        datasets=datasets,
        model_fn=lambda t, **p: p["m"] * t + p["b"],
        global_params=["m"],
        local_params=["b"],
        initial_params=seeds,
        free_global_params=["m"],
        local_group_key=lambda name, run: run,
        t_min=None,
        t_max=None,
        cancel_callback=None,
        observations={ds.run_number: dataset_observations(ds) for ds in datasets},
    )
    covariance = observed_covariance(problem, np.array([0.7, 1, 2]), problem.cancel_guard)
    design = (
        np.column_stack([np.tile(x, 2), np.repeat([1, 0], len(x)), np.repeat([0, 1], len(x))]) / 0.2
    )
    oracle = np.linalg.inv(design.T @ design)
    assert oracle[1, 2] > 0
    np.testing.assert_allclose(covariance.block([0, 1, 2]), oracle, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("defect", ["singular", "nonfinite", "bound", "unstable"])
def test_invalid_curvature_has_an_explicit_reason(defect):
    from asymmetry.core.fitting.global_covariance import (
        CurvatureUnavailableError,
        _information_inverse,
        _member_information,
    )
    from asymmetry.core.fitting.likelihood import GaussianObservations

    if defect == "singular":
        with pytest.raises(CurvatureUnavailableError, match="ill-conditioned"):
            _information_inverse(np.array([[1.0, 1.0 - 1e-12], [1.0 - 1e-12, 1.0]]))
        return
    record = GaussianObservations(np.arange(5.0), np.ones(5), np.ones(5))
    if defect == "bound":
        function, bounds, reason = lambda p: p[0] * record.time, [(0, 1)], "bound"
    elif defect == "nonfinite":
        function, bounds, reason = lambda p: np.full(5, np.nan), [(-1, 1)], "nonfinite"
    else:
        # A nondifferentiable model cannot supply a stable symmetric Hessian.
        function, bounds, reason = lambda p: np.full(5, np.sqrt(abs(p[0]))), [(-1, 1)], "unstable"
    with pytest.raises(CurvatureUnavailableError, match=reason):
        _member_information(function, np.array([0.0]), bounds, record)


@pytest.mark.parametrize("strategy", ["profiled", "least_squares"])
def test_identifiable_strongly_correlated_linear_fit_retains_errors(strategy):
    x = 10 + np.linspace(-0.01, 0.01, 31)
    datasets = [
        MuonDataset(
            time=x,
            asymmetry=0.7 * x + run,
            error=np.full(len(x), 0.2),
            metadata={"run_number": run},
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("m", 0.7), Parameter("b", run)]) for run in (1, 2)}
    design = (
        np.column_stack([np.tile(x, 2), np.repeat([1, 0], len(x)), np.repeat([0, 1], len(x))]) / 0.2
    )
    oracle = np.linalg.inv(design.T @ design)
    assert np.linalg.cond(design.T @ design) > 1e8
    results, _ = FitEngine().global_fit(
        datasets, lambda t, **p: p["m"] * t + p["b"], ["m"], ["b"], seeds, strategy=strategy
    )
    for run, result in results.items():
        assert result.success
        assert result.covariance_accurate
        np.testing.assert_allclose(result.covariance, oracle[np.ix_([0, run], [0, run])], rtol=2e-3)


def test_numerical_information_noise_is_checked_after_marginalisation(monkeypatch):
    from asymmetry.core.fitting import global_covariance

    x = 10 + np.linspace(-0.01, 0.01, 31)
    datasets = [
        MuonDataset(
            time=x,
            asymmetry=0.7 * x + run,
            error=np.full(len(x), 0.2),
            metadata={"run_number": run},
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("m", 0.7), Parameter("b", run)]) for run in (1, 2)}
    information = global_covariance._member_information
    attempts = []

    def noisy(*args):
        attempts.append(args[4])
        estimates = information(*args)
        # A 2e-8 relative derivative error is tiny before eliminating the nearly
        # collinear intercept but large relative to the slope's remaining information.
        estimates[1, 0, 0] *= 1 + 2e-8
        return estimates

    monkeypatch.setattr(global_covariance, "_member_information", noisy)
    results, _ = FitEngine().global_fit(
        datasets, lambda t, **p: p["m"] * t + p["b"], ["m"], ["b"], seeds, strategy="least_squares"
    )
    for result in results.values():
        assert result.success
        assert not result.covariance_accurate
        assert result.covariance is None
        assert not result.uncertainties
        assert any(
            "marginal covariance is numerically unstable" in reason for reason in result.warnings
        )

    assert attempts == [
        scale for scale in (1.0, 0.25, 0.0625, 0.015625, 0.00390625) for _ in range(2)
    ]


def test_bounded_refinement_recovers_a_smooth_neighbourhood_of_a_numerical_grid():
    x = np.linspace(-1, 3, 21)
    slope = 0.5 - 3e-6
    datasets = [
        MuonDataset(
            time=x,
            asymmetry=slope * x + run,
            error=np.full(len(x), 0.2),
            metadata={"run_number": run},
        )
        for run in (1, 2)
    ]
    seeds = {run: ParameterSet([Parameter("m", slope), Parameter("b", run)]) for run in (1, 2)}

    def model(t, **p):
        # A grid selection can change the numerical model across an otherwise smooth fit.
        return p["m"] * t + p["b"] + (1e-4 if p["m"] >= 0.5 else 0.0)

    results, _ = FitEngine().global_fit(
        datasets, model, ["m"], ["b"], seeds, strategy="least_squares"
    )
    design = (
        np.column_stack([np.tile(x, 2), np.repeat([1, 0], len(x)), np.repeat([0, 1], len(x))]) / 0.2
    )
    oracle = np.linalg.inv(design.T @ design)
    for run, result in results.items():
        assert result.success
        assert result.covariance_accurate
        np.testing.assert_allclose(
            result.covariance, oracle[np.ix_([0, run], [0, run])], rtol=2e-3, atol=1e-8
        )

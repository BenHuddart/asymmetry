"""Acceptance tests for the sparse least-squares global-fit strategy.

``strategy="least_squares"`` optimises exactly the vector the joint Minuit path
builds — the same free globals, grouped/per-run locals, limits and per-dataset
pinning — but hands it to a bounded trust-region solver over the concatenated
residual vector, with the coupled problem's arrow-shaped Jacobian pattern
declared. The whole point of the strategy is that it reaches the *same minimum*
far more cheaply, so these tests are a parity gate against ``"joint"``: χ² within
0.1, values within one joint-path uncertainty, uncertainties within 10 % on a
well-conditioned case. They also pin the block-sparse pattern itself, the bound
and fixed/grouped-tie handling, and the two requests the strategy refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import (
    POISSON_COST,
    FitEngine,
    _build_coupled_global_problem,
    _coupled_jacobian_sparsity,
)
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

_MODEL = MODELS["ExponentialRelaxation"].function


def _make_series(
    *,
    a0_true: float,
    lambdas: list[float],
    baseline_true: float = 0.0,
    seed: int = 0,
    sigma: float = 0.4,
    n_points: int = 250,
) -> tuple[list[MuonDataset], dict[int, ParameterSet]]:
    """A synthetic exponential-relaxation series with a shared A0 and per-run Λ."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.05, 8.0, n_points)
    datasets: list[MuonDataset] = []
    inits: dict[int, ParameterSet] = {}
    for index, lam in enumerate(lambdas):
        clean = _MODEL(time, A0=a0_true, Lambda=lam, baseline=baseline_true)
        asymmetry = clean + rng.normal(0.0, sigma, time.size)
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=asymmetry,
                error=np.full_like(time, sigma),
                metadata={"run_number": index},
            )
        )
        params = ParameterSet()
        params.add(Parameter("A0", 20.0, min=0.0))
        params.add(Parameter("Lambda", 0.5, min=0.0))
        params.add(Parameter("baseline", baseline_true, fixed=True))
        inits[index] = params
    return datasets, inits


def _total_chi2(results: dict) -> float:
    return float(sum(result.chi_squared for result in results.values()))


# --------------------------------------------------------------------------- #
# Parity with the joint Minuit path
# --------------------------------------------------------------------------- #


def test_least_squares_matches_joint_on_a_shared_parameter_series() -> None:
    """Two datasets, one shared A0: truth recovered and the joint minimum reached."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.35, 0.9], seed=3)
    engine = FitEngine()

    joint, joint_global = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="joint"
    )
    sparse, sparse_global = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )

    # The truth is recovered: the shared amplitude within a few σ of 22.
    a0_sigma = joint[0].uncertainties["A0"]
    assert sparse_global["A0"].value == pytest.approx(22.0, abs=5.0 * a0_sigma)

    # Same minimum as Minuit: χ² within 0.1 over the whole series.
    assert _total_chi2(sparse) == pytest.approx(_total_chi2(joint), abs=0.1)

    # Every fitted value within one joint-path uncertainty of the joint value.
    assert abs(sparse_global["A0"].value - joint_global["A0"].value) < a0_sigma
    for run in (0, 1):
        lambda_sigma = joint[run].uncertainties["Lambda"]
        assert (
            abs(sparse[run].parameters["Lambda"].value - joint[run].parameters["Lambda"].value)
            < lambda_sigma
        )
        # Well-conditioned case: the Gauss-Newton errors match HESSE within 10 %.
        assert sparse[run].uncertainties["Lambda"] == pytest.approx(lambda_sigma, rel=0.10)
        assert sparse[run].uncertainties["A0"] == pytest.approx(a0_sigma, rel=0.10)


def test_least_squares_reports_solver_status_calls_and_covariance_block() -> None:
    """The result carries the solver's own message and call counts, not Minuit's."""
    datasets, inits = _make_series(a0_true=18.0, lambdas=[0.4, 0.7, 1.2], seed=5)
    engine = FitEngine()

    results, _ = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )

    for result in results.values():
        assert result.success
        assert "least squares" in result.message
        assert "termination condition is satisfied" in result.message
        assert result.function_calls > 0
        assert result.gradient_calls > 0
        # Per-run covariance sub-block: the shared global plus this run's local.
        assert result.covariance_parameters == ["A0", "Lambda"]
        assert result.covariance is not None
        assert result.covariance.shape == (2, 2)
        for name, index in (("A0", 0), ("Lambda", 1)):
            assert result.uncertainties[name] == pytest.approx(
                float(np.sqrt(result.covariance[index, index]))
            )


def test_least_squares_matches_joint_with_two_shared_globals() -> None:
    """A second free global (the baseline) still lands on the joint minimum."""
    rng = np.random.default_rng(11)
    time = np.linspace(0.05, 8.0, 200)
    datasets: list[MuonDataset] = []
    inits: dict[int, ParameterSet] = {}
    for index, lam in enumerate([0.4, 0.7, 1.0]):
        clean = _MODEL(time, A0=18.0, Lambda=lam, baseline=1.5)
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=clean + rng.normal(0.0, 0.35, time.size),
                error=np.full_like(time, 0.35),
                metadata={"run_number": index},
            )
        )
        params = ParameterSet()
        params.add(Parameter("A0", 15.0, min=0.0))
        params.add(Parameter("Lambda", 0.6, min=0.0))
        params.add(Parameter("baseline", 0.0))
        inits[index] = params
    engine = FitEngine()

    joint, joint_global = engine.global_fit(
        datasets, _MODEL, ["A0", "baseline"], ["Lambda"], inits, strategy="joint"
    )
    sparse, sparse_global = engine.global_fit(
        datasets, _MODEL, ["A0", "baseline"], ["Lambda"], inits, strategy="least_squares"
    )

    assert _total_chi2(sparse) == pytest.approx(_total_chi2(joint), abs=0.1)
    for name in ("A0", "baseline"):
        assert (
            abs(sparse_global[name].value - joint_global[name].value) < joint[0].uncertainties[name]
        )


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #


def test_a_truth_outside_its_limit_lands_on_the_limit() -> None:
    """A bound the truth violates is respected, and the solve still reports success.

    The joint path reports ``success`` from Minuit's own validity flag, which
    stays true for a converged fit whose parameter sits at a limit; the sparse
    path reports the trust-region solver's convergence status the same way.
    """
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.30, 0.35], seed=7)
    capped = 0.20
    for params in inits.values():
        params.add(Parameter("Lambda", 0.15, min=0.0, max=capped))
    engine = FitEngine()

    results, _ = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )

    for result in results.values():
        assert result.success
        fitted = result.parameters["Lambda"].value
        assert fitted <= capped
        assert fitted == pytest.approx(capped, abs=1e-6)


def test_a_start_sitting_exactly_on_a_bound_is_still_fitted() -> None:
    """A seed pinned at its physical floor is nudged inside, not frozen there.

    Trust-region reflective scales its step by the distance to the nearer bound,
    so a start exactly *on* one has zero step scale and cannot move.
    """
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=13)
    for params in inits.values():
        params.add(Parameter("Lambda", 0.0, min=0.0))
    engine = FitEngine()

    results, _ = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )

    assert results[0].parameters["Lambda"].value > 0.1
    assert results[1].parameters["Lambda"].value > 0.1


# --------------------------------------------------------------------------- #
# Fixed, shared and grouped parameters
# --------------------------------------------------------------------------- #


def test_fixed_parameters_are_held_and_reported_fixed() -> None:
    """A pinned parameter keeps its value and its ``fixed`` flag in every result."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=17)
    engine = FitEngine()

    results, _ = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )

    for result in results.values():
        baseline = result.parameters["baseline"]
        assert baseline.fixed
        assert baseline.value == 0.0
        assert "baseline" not in result.uncertainties


def test_a_dataset_that_pins_a_shared_global_reports_its_own_value() -> None:
    """One run pinning the shared name keeps its measured value, not the fit's."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9, 1.4], seed=19)
    inits[1].add(Parameter("A0", 25.0, min=0.0, fixed=True))
    engine = FitEngine()

    results, fitted_global = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )

    assert results[1].parameters["A0"].value == 25.0
    assert results[1].parameters["A0"].fixed
    assert "A0" not in results[1].uncertainties
    for run in (0, 2):
        assert results[run].parameters["A0"].value == pytest.approx(fitted_global["A0"].value)
        assert not results[run].parameters["A0"].fixed


def test_grouped_local_ties_share_one_value_and_match_the_joint_path() -> None:
    """``local_param_groups`` gives the grouped runs one fitted Λ, as joint does."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.5, 0.5, 1.2, 1.2], seed=23)
    groups = {"Lambda": {0: "slow", 1: "slow", 2: "fast", 3: "fast"}}
    engine = FitEngine()

    joint, _ = engine.global_fit(
        datasets,
        _MODEL,
        ["A0"],
        ["Lambda"],
        inits,
        strategy="joint",
        local_param_groups=groups,
    )
    sparse, _ = engine.global_fit(
        datasets,
        _MODEL,
        ["A0"],
        ["Lambda"],
        inits,
        strategy="least_squares",
        local_param_groups=groups,
    )

    slow = sparse[0].parameters["Lambda"].value
    fast = sparse[2].parameters["Lambda"].value
    assert sparse[1].parameters["Lambda"].value == slow
    assert sparse[3].parameters["Lambda"].value == fast
    assert slow != fast
    assert _total_chi2(sparse) == pytest.approx(_total_chi2(joint), abs=0.1)
    assert slow == pytest.approx(joint[0].parameters["Lambda"].value, abs=1e-3)


def test_grouped_local_ties_alone_are_coupled_without_any_free_global() -> None:
    """A group tie couples the problem even with every global pinned."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.5, 0.5], seed=29)
    for params in inits.values():
        params.add(Parameter("A0", 22.0, min=0.0, fixed=True))
    groups = {"Lambda": {0: "both", 1: "both"}}
    engine = FitEngine()

    results, _ = engine.global_fit(
        datasets,
        _MODEL,
        ["A0"],
        ["Lambda"],
        inits,
        strategy="least_squares",
        local_param_groups=groups,
    )

    assert results[0].parameters["Lambda"].value == results[1].parameters["Lambda"].value
    assert results[0].parameters["Lambda"].value == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# Unsupported requests fail loudly
# --------------------------------------------------------------------------- #


def test_a_cost_factory_is_refused_by_name() -> None:
    """Only the least-squares cost has a residual vector to hand the solver."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=31)
    engine = FitEngine()

    with pytest.raises(NotImplementedError, match="least_squares"):
        engine.global_fit(
            datasets,
            _MODEL,
            ["A0"],
            ["Lambda"],
            inits,
            strategy="least_squares",
            cost_factory=POISSON_COST,
        )


def test_minos_is_refused_by_name() -> None:
    """MINOS is a Minuit scan; the trust-region solver has no equivalent."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=37)
    engine = FitEngine()

    with pytest.raises(NotImplementedError, match="least_squares"):
        engine.global_fit(
            datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares", minos=True
        )


def test_an_unknown_strategy_names_all_three() -> None:
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=41)
    engine = FitEngine()

    with pytest.raises(ValueError, match="least_squares"):
        engine.global_fit(datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="lsmr")


# --------------------------------------------------------------------------- #
# The sparsity pattern
# --------------------------------------------------------------------------- #


def _problem(datasets, inits, *, global_params, local_params, groups=None, t_min=None, t_max=None):
    def local_group_key(pname: str, run_number: int):
        if groups and pname in groups:
            return groups[pname].get(run_number, run_number)
        return run_number

    free_global_params = [
        pname
        for pname in global_params
        if any(not inits[ds.run_number][pname].fixed for ds in datasets)
    ]
    return _build_coupled_global_problem(
        datasets=datasets,
        model_fn=_MODEL,
        global_params=global_params,
        local_params=local_params,
        initial_params=inits,
        free_global_params=free_global_params,
        local_group_key=local_group_key,
        t_min=t_min,
        t_max=t_max,
        cancel_callback=None,
    )


def test_sparsity_pattern_is_the_arrow_of_globals_and_per_run_locals() -> None:
    """Column 0 is the shared global; each run's rows carry only its own local."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9, 1.4], seed=43, n_points=6)
    problem = _problem(datasets, inits, global_params=["A0"], local_params=["Lambda"])

    assert problem.param_names == ["A0", "Lambda_0", "Lambda_1", "Lambda_2"]
    assert problem.dataset_columns == ((0, 1), (0, 2), (0, 3))

    pattern = _coupled_jacobian_sparsity(problem).toarray()
    assert pattern.shape == (18, 4)
    # The global column is dense; each local column covers exactly its own run.
    assert pattern[:, 0].sum() == 18
    for run in range(3):
        expected = np.zeros(18, dtype=np.int8)
        expected[run * 6 : (run + 1) * 6] = 1
        assert np.array_equal(pattern[:, run + 1], expected)


def test_sparsity_pattern_shares_one_column_across_a_local_group() -> None:
    """A grouped local is one column spanning every dataset in its group."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.5, 0.5, 1.2, 1.2], seed=47, n_points=5)
    groups = {"Lambda": {0: "slow", 1: "slow", 2: "fast", 3: "fast"}}
    problem = _problem(
        datasets, inits, global_params=["A0"], local_params=["Lambda"], groups=groups
    )

    assert problem.param_names == ["A0", "Lambda_slow", "Lambda_fast"]
    assert problem.dataset_columns == ((0, 1), (0, 1), (0, 2), (0, 2))

    pattern = _coupled_jacobian_sparsity(problem).toarray()
    assert pattern.shape == (20, 3)
    assert np.array_equal(np.flatnonzero(pattern[:, 1]), np.arange(0, 10))
    assert np.array_equal(np.flatnonzero(pattern[:, 2]), np.arange(10, 20))


def test_sparsity_pattern_drops_a_global_the_dataset_pins_itself() -> None:
    """A run that pins the shared name does not depend on its column."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=53, n_points=4)
    inits[1].add(Parameter("A0", 25.0, min=0.0, fixed=True))
    problem = _problem(datasets, inits, global_params=["A0"], local_params=["Lambda"])

    assert problem.dataset_columns == ((0, 1), (2,))

    pattern = _coupled_jacobian_sparsity(problem).toarray()
    assert pattern[:4, 0].sum() == 4
    assert pattern[4:, 0].sum() == 0


# --------------------------------------------------------------------------- #
# Windowing and oversampling
# --------------------------------------------------------------------------- #


def test_t_min_and_t_max_window_the_fit_and_the_dof() -> None:
    """The window reaches the residual vector, the χ² and the reported dof."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=59)
    engine = FitEngine()

    full, _ = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )
    windowed, _ = engine.global_fit(
        datasets,
        _MODEL,
        ["A0"],
        ["Lambda"],
        inits,
        t_min=1.0,
        t_max=5.0,
        strategy="least_squares",
    )
    joint_windowed, _ = engine.global_fit(
        datasets,
        _MODEL,
        ["A0"],
        ["Lambda"],
        inits,
        t_min=1.0,
        t_max=5.0,
        strategy="joint",
    )

    in_window = int(np.sum((datasets[0].time >= 1.0) & (datasets[0].time <= 5.0)))
    assert windowed[0].dof == in_window - 2
    assert windowed[0].dof < full[0].dof
    assert windowed[0].residuals is not None
    assert windowed[0].residuals.size == in_window
    assert _total_chi2(windowed) == pytest.approx(_total_chi2(joint_windowed), abs=0.1)


def test_error_oversampling_scales_chi2_dof_and_uncertainties() -> None:
    """The zero-padding correction is applied exactly as on the joint path."""
    datasets, inits = _make_series(a0_true=22.0, lambdas=[0.4, 0.9], seed=61)
    engine = FitEngine()

    plain, _ = engine.global_fit(
        datasets, _MODEL, ["A0"], ["Lambda"], inits, strategy="least_squares"
    )
    corrected, _ = engine.global_fit(
        datasets,
        _MODEL,
        ["A0"],
        ["Lambda"],
        inits,
        strategy="least_squares",
        error_oversampling=4.0,
    )

    for run in (0, 1):
        assert corrected[run].chi_squared == pytest.approx(plain[run].chi_squared / 4.0)
        assert corrected[run].dof == max(len(datasets[run].time) // 4 - 2, 1)
        assert corrected[run].uncertainties["Lambda"] == pytest.approx(
            plain[run].uncertainties["Lambda"] * 2.0
        )
        assert corrected[run].warnings

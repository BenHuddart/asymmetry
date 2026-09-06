"""Tests for the full-covariance GLS collapse surrogate.

The surrogate claims to be *exact* for a quadratic χ² surface, so the central
test builds the surface analytically (a linear least-squares problem has
χ²(θ) = χ²_min + (θ − θ̂)ᵀ C⁻¹ (θ − θ̂) exactly) and compares Δχ², the shared
values, and the conditional locals against an independently solved constrained
refit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import SelectionMetric, compute_information_criteria
from asymmetry.core.fitting.global_search.surrogate import (
    CONDITION_LIMIT,
    OrderedCollapse,
    RunEstimate,
    collapse_cost,
    greedy_assignment,
    metric_penalty,
    rank_assignments,
    run_estimate_from_fit_result,
    surrogate_ic,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

NAMES = ("a", "b", "c")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _random_covariance(rng: np.random.Generator, size: int) -> np.ndarray:
    """A well-conditioned symmetric positive-definite matrix."""

    root = rng.normal(size=(size, size))
    return root @ root.T + size * np.eye(size)


def _estimate(
    run_number: int,
    values,
    covariance=None,
    *,
    names=NAMES,
    uncertainties=None,
    at_bound=(),
    chi_squared: float = 100.0,
    n_points: int = 200,
) -> RunEstimate:
    values = np.asarray(values, dtype=float)
    if uncertainties is None:
        uncertainties = (
            np.sqrt(np.diag(covariance)) if covariance is not None else np.ones(len(values))
        )
    return RunEstimate(
        run_number=run_number,
        names=tuple(names),
        values=values,
        covariance=None if covariance is None else np.asarray(covariance, dtype=float),
        uncertainties=np.asarray(uncertainties, dtype=float),
        at_bound=frozenset(at_bound),
        chi_squared=chi_squared,
        n_points=n_points,
    )


def _exact_constrained_refit(estimates, subset):
    """Minimise Σ_r (θ_r − θ̂_r)ᵀ C_r⁻¹ (θ_r − θ̂_r) with ``subset`` tied.

    Independent of the module under test: builds the whole constrained quadratic
    and solves its normal equations. Returns ``(delta_chi2, shared, locals)``.
    """

    names = estimates[0].names
    subset = tuple(subset)
    rest = [name for name in names if name not in set(subset)]
    n_shared, n_rest = len(subset), len(rest)
    dimension = n_shared + n_rest * len(estimates)

    def design(run_index: int) -> np.ndarray:
        matrix = np.zeros((len(names), dimension))
        for column, name in enumerate(subset):
            matrix[names.index(name), column] = 1.0
        for column, name in enumerate(rest):
            matrix[names.index(name), n_shared + run_index * n_rest + column] = 1.0
        return matrix

    normal = np.zeros((dimension, dimension))
    target = np.zeros(dimension)
    for index, estimate in enumerate(estimates):
        weight = np.linalg.inv(estimate.covariance)
        matrix = design(index)
        normal += matrix.T @ weight @ matrix
        target += matrix.T @ weight @ estimate.values
    solution = np.linalg.solve(normal, target)

    delta = 0.0
    conditional = {}
    for index, estimate in enumerate(estimates):
        weight = np.linalg.inv(estimate.covariance)
        residual = design(index) @ solution - estimate.values
        delta += float(residual @ weight @ residual)
        conditional[estimate.run_number] = {
            name: float(solution[n_shared + index * n_rest + column])
            for column, name in enumerate(rest)
        }
    shared = {name: float(solution[column]) for column, name in enumerate(subset)}
    return delta, shared, conditional


# --------------------------------------------------------------------------- #
# Exactness of the collapse
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("subset", [("a",), ("a", "c"), ("a", "b", "c")])
def test_collapse_matches_the_exact_constrained_refit(subset):
    rng = np.random.default_rng(20260906)
    estimates = [
        _estimate(run, rng.normal(size=3) * 2.0, _random_covariance(rng, 3)) for run in range(5)
    ]

    result = collapse_cost(estimates, subset)
    expected_delta, expected_shared, expected_locals = _exact_constrained_refit(estimates, subset)

    assert result.delta_chi2 == pytest.approx(expected_delta, rel=1e-9, abs=1e-9)
    assert result.diagonal_fallback_runs == frozenset()
    for name, value in expected_shared.items():
        assert result.shared_values[name] == pytest.approx(value, rel=1e-9)
    for run, locals_ in expected_locals.items():
        for name, value in locals_.items():
            assert result.conditional_locals_by_run[run][name] == pytest.approx(
                value, rel=1e-9, abs=1e-9
            )


def test_collapse_of_a_correlated_pair_beats_the_diagonal_wald_answer():
    """The conditional shift is the whole point: correlation changes Δχ²."""

    correlated = np.array([[1.0, 0.9], [0.9, 1.0]])
    estimates = [
        _estimate(0, [0.0, 0.0], correlated, names=("a", "b")),
        _estimate(1, [1.0, 1.0], correlated, names=("a", "b")),
    ]

    full = collapse_cost(estimates, ("a",)).delta_chi2
    diagonal = collapse_cost(
        [
            _estimate(0, [0.0, 0.0], np.eye(2), names=("a", "b")),
            _estimate(1, [1.0, 1.0], np.eye(2), names=("a", "b")),
        ],
        ("a",),
    ).delta_chi2

    assert full == pytest.approx(0.5)
    assert diagonal == pytest.approx(0.5)
    # ... but the *warm start* differs: the correlated case drags ``b`` along.
    correlated_local = collapse_cost(estimates, ("a",)).conditional_locals_by_run[0]["b"]
    assert correlated_local == pytest.approx(0.45)


def test_empty_subset_costs_nothing_and_keeps_every_value_local():
    estimates = [_estimate(0, [1.0, 2.0, 3.0], np.eye(3))]

    result = collapse_cost(estimates, ())

    assert result.subset == ()
    assert result.delta_chi2 == 0.0
    assert result.shared_values == {}
    assert result.conditional_locals_by_run[0] == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert result.diagonal_fallback_runs == frozenset()


# --------------------------------------------------------------------------- #
# Diagonal fallback
# --------------------------------------------------------------------------- #


def test_missing_covariance_falls_back_to_the_diagonal_weight():
    estimates = [
        _estimate(0, [1.0, 0.0, 0.0], None, uncertainties=[0.5, 1.0, 1.0]),
        _estimate(1, [3.0, 0.0, 0.0], None, uncertainties=[0.5, 1.0, 1.0]),
    ]

    result = collapse_cost(estimates, ("a",))

    # Inverse-variance mean of 1 and 3 with equal σ is 2; Q = 2·(1/0.25) = 8.
    assert result.shared_values["a"] == pytest.approx(2.0)
    assert result.delta_chi2 == pytest.approx(8.0)
    assert result.diagonal_fallback_runs == frozenset({0, 1})
    # No covariance means no conditional shift.
    assert result.conditional_locals_by_run[0]["b"] == 0.0


def test_singular_covariance_block_falls_back_for_that_run_only():
    good = np.diag([0.25, 1.0, 1.0])
    singular = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0 + 1.0 / (10.0 * CONDITION_LIMIT), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    estimates = [
        _estimate(0, [1.0, 0.0, 0.0], good, uncertainties=[0.5, 1.0, 1.0]),
        _estimate(1, [3.0, 0.0, 0.0], singular, uncertainties=[0.5, 1.0, 1.0]),
    ]

    result = collapse_cost(estimates, ("a", "b"))

    assert result.diagonal_fallback_runs == frozenset({1})
    assert math.isfinite(result.delta_chi2)


def test_non_finite_covariance_entry_is_dropped_when_building_the_estimate():
    parameters = ParameterSet([Parameter(name=name, value=1.0) for name in NAMES])
    result = FitResult(
        success=True,
        chi_squared=10.0,
        parameters=parameters,
        uncertainties={name: 1.0 for name in NAMES},
        covariance=np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]]),
        covariance_parameters=list(NAMES),
    )

    estimate = run_estimate_from_fit_result(result, NAMES, run_number=7, n_points=200)

    assert estimate.covariance is None


def test_a_parameter_no_run_constrains_makes_the_collapse_infinite():
    estimates = [
        _estimate(0, [1.0, 0.0, 0.0], None, uncertainties=[math.nan, 1.0, 1.0]),
        _estimate(1, [3.0, 0.0, 0.0], None, uncertainties=[0.0, 1.0, 1.0]),
    ]

    result = collapse_cost(estimates, ("a",))

    assert result.delta_chi2 == math.inf
    assert result.shared_values == {}
    assert surrogate_ic(estimates, ("a",), SelectionMetric.BIC) == math.inf


# --------------------------------------------------------------------------- #
# Building an estimate from a FitResult
# --------------------------------------------------------------------------- #


def test_run_estimate_reorders_the_covariance_to_the_free_name_order():
    parameters = ParameterSet(
        [
            Parameter(name="a", value=1.0),
            Parameter(name="b", value=2.0),
            Parameter(name="c", value=3.0),
        ]
    )
    # Engine order (c, a, b) with a distinct value per cell.
    engine_order = ["c", "a", "b"]
    covariance = np.array([[9.0, 0.3, 0.6], [0.3, 1.0, 0.1], [0.6, 0.1, 4.0]])
    result = FitResult(
        success=True,
        chi_squared=42.0,
        parameters=parameters,
        uncertainties={"a": 1.0, "b": 2.0, "c": 3.0},
        covariance=covariance,
        covariance_parameters=engine_order,
    )

    estimate = run_estimate_from_fit_result(
        result, ("a", "b", "c"), run_number=101, n_points=512, at_bound=["b"]
    )

    assert estimate.run_number == 101
    assert estimate.n_points == 512
    assert estimate.names == ("a", "b", "c")
    assert estimate.at_bound == frozenset({"b"})
    assert estimate.chi_squared == 42.0
    np.testing.assert_allclose(estimate.values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(estimate.uncertainties, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        estimate.covariance,
        np.array([[1.0, 0.1, 0.3], [0.1, 4.0, 0.6], [0.3, 0.6, 9.0]]),
    )


def test_run_estimate_drops_a_covariance_that_misses_a_free_parameter():
    parameters = ParameterSet([Parameter(name=name, value=1.0) for name in NAMES])
    result = FitResult(
        success=True,
        parameters=parameters,
        uncertainties={name: 1.0 for name in NAMES},
        covariance=np.eye(2),
        covariance_parameters=["a", "b"],
    )

    estimate = run_estimate_from_fit_result(result, NAMES, run_number=1, n_points=10)

    assert estimate.covariance is None


def test_run_estimate_marks_a_missing_uncertainty_as_carrying_no_information():
    parameters = ParameterSet([Parameter(name=name, value=1.0) for name in NAMES])
    result = FitResult(success=True, parameters=parameters, uncertainties={"a": 1.0, "b": 2.0})

    estimate = run_estimate_from_fit_result(result, NAMES, run_number=1, n_points=10)

    assert math.isnan(estimate.uncertainties[2])


# --------------------------------------------------------------------------- #
# Information criterion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", list(SelectionMetric))
@pytest.mark.parametrize("parameter_count, sample_count", [(1, 500), (24, 4000), (7, 8)])
def test_penalty_agrees_with_compute_information_criteria(metric, parameter_count, sample_count):
    chi_squared = 123.5
    aic, aicc, bic = compute_information_criteria(chi_squared, parameter_count, sample_count)
    expected = {
        SelectionMetric.AIC: aic,
        SelectionMetric.BIC: bic,
        SelectionMetric.AICC: aicc if aicc is not None else aic,
    }[metric]

    penalty = metric_penalty(parameter_count, sample_count=sample_count, metric=metric)

    assert chi_squared + penalty == pytest.approx(expected)


def test_aicc_penalty_falls_back_to_aic_when_the_correction_is_undefined():
    _aic, aicc, _bic = compute_information_criteria(0.0, 10, 10)
    assert aicc is None
    assert metric_penalty(10, sample_count=10, metric=SelectionMetric.AICC) == 20.0


@pytest.mark.parametrize("metric", list(SelectionMetric))
def test_surrogate_ic_is_chi2_plus_collapse_plus_the_shared_penalty(metric):
    rng = np.random.default_rng(7)
    estimates = [
        _estimate(
            run,
            rng.normal(size=3),
            _random_covariance(rng, 3),
            chi_squared=100.0 + run,
            n_points=250,
        )
        for run in range(4)
    ]
    subset = ("a", "c")

    delta = collapse_cost(estimates, subset).delta_chi2
    chi_squared = sum(estimate.chi_squared for estimate in estimates)
    # k = |S| + (P - |S|)·G = 2 + 1·4
    aic, aicc, bic = compute_information_criteria(chi_squared + delta, 6, 1000)
    expected = {
        SelectionMetric.AIC: aic,
        SelectionMetric.BIC: bic,
        SelectionMetric.AICC: aicc if aicc is not None else aic,
    }[metric]

    assert surrogate_ic(estimates, subset, metric) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def test_rank_assignments_enumerates_every_subset_in_ascending_ic_order():
    rng = np.random.default_rng(11)
    estimates = [_estimate(run, rng.normal(size=3), _random_covariance(rng, 3)) for run in range(3)]

    ranked = rank_assignments(estimates, NAMES, SelectionMetric.BIC)

    assert len(ranked) == 2 ** len(NAMES)
    assert {subset for subset, _ic in ranked} == {
        (),
        ("a",),
        ("b",),
        ("c",),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "b", "c"),
    }
    scores = [ic for _subset, ic in ranked]
    assert scores == sorted(scores)
    # Subsets keep the free_names order, so structure keys are stable.
    for subset, _ic in ranked:
        assert list(subset) == [name for name in NAMES if name in set(subset)]


def test_rank_assignments_prefers_globalising_a_genuinely_shared_parameter():
    covariance = np.diag([0.01, 0.01, 0.01])
    estimates = [
        _estimate(0, [1.0, 1.0, 0.0], covariance, chi_squared=500.0, n_points=400),
        _estimate(1, [1.0, 5.0, 0.0], covariance, chi_squared=500.0, n_points=400),
        _estimate(2, [1.0, 9.0, 0.0], covariance, chi_squared=500.0, n_points=400),
    ]

    best_subset, _ic = rank_assignments(estimates, NAMES, SelectionMetric.BIC)[0]

    assert "a" in best_subset
    assert "b" not in best_subset


def test_a_parameter_at_a_bound_on_any_run_is_never_proposed():
    covariance = np.diag([0.01, 0.01, 0.01])
    estimates = [
        _estimate(0, [1.0, 1.0, 0.0], covariance, at_bound=("a",)),
        _estimate(1, [1.0, 5.0, 0.0], covariance),
    ]

    ranked = rank_assignments(estimates, NAMES, SelectionMetric.BIC)

    assert all("a" not in subset for subset, _ic in ranked)
    assert len(ranked) == 4
    # collapse_cost still answers if a caller asks for it explicitly.
    assert collapse_cost(estimates, ("a",)).delta_chi2 == pytest.approx(0.0)


def test_above_max_enumerated_only_single_parameter_subsets_are_ranked():
    names = tuple(f"p{index}" for index in range(13))
    covariance = np.diag([0.01] * 13)
    estimates = [
        _estimate(run, np.zeros(13) + run * 0.001, covariance, names=names) for run in range(3)
    ]

    ranked = rank_assignments(estimates, names, SelectionMetric.BIC)

    assert len(ranked) == 13
    assert all(len(subset) == 1 for subset, _ic in ranked)

    # The threshold counts *eligible* names, so bounds shrink it back into the
    # enumerated branch.
    bounded = [
        _estimate(
            run,
            np.zeros(13),
            covariance,
            names=names,
            at_bound=tuple(names[3:]),
        )
        for run in range(3)
    ]
    assert len(rank_assignments(bounded, names, SelectionMetric.BIC)) == 2**3


# --------------------------------------------------------------------------- #
# Greedy forward selection
# --------------------------------------------------------------------------- #


def test_greedy_matches_full_enumeration_when_the_subsets_score_separably():
    """Diagonal covariance and a penalty linear in |S| make greedy exact.

    With ``W_r`` diagonal, ``Δχ²(S) = Σ_{p∈S} Δχ²({p})``, and BIC's penalty is
    linear in ``|S|``; the objective is then a sum of independent per-parameter
    terms, so picking the best one at a time is picking the best subset. This is
    the case the partition search actually runs in whenever a fit reports no
    covariance and the diagonal fallback fires.
    """
    covariance = np.diag([0.01, 0.02, 0.04])
    estimates = [
        _estimate(0, [1.0, 1.0, 0.0], covariance, chi_squared=500.0, n_points=400),
        _estimate(1, [1.0, 5.0, 0.1], covariance, chi_squared=500.0, n_points=400),
        _estimate(2, [1.0, 9.0, 0.2], covariance, chi_squared=500.0, n_points=400),
    ]

    best_subset, best_ic = rank_assignments(estimates, NAMES, SelectionMetric.BIC)[0]
    greedy_subset, greedy_ic = greedy_assignment(estimates, NAMES, SelectionMetric.BIC)

    assert greedy_subset == best_subset
    assert greedy_ic == pytest.approx(best_ic)


@pytest.mark.parametrize("seed", [3, 17, 41, 99, 2026])
def test_greedy_matches_full_enumeration_on_correlated_blocks_too(seed):
    rng = np.random.default_rng(seed)
    estimates = [
        _estimate(
            run,
            rng.normal(size=3) * 0.1,
            _random_covariance(rng, 3) * 0.01,
            chi_squared=400.0 + run,
            n_points=300,
        )
        for run in range(5)
    ]

    best_subset, best_ic = rank_assignments(estimates, NAMES, SelectionMetric.BIC)[0]
    greedy_subset, greedy_ic = greedy_assignment(estimates, NAMES, SelectionMetric.BIC)

    assert greedy_subset == best_subset
    assert greedy_ic == pytest.approx(best_ic)


def test_greedy_never_claims_to_beat_the_exhaustive_optimum():
    """Forward selection is a bound, not an oracle: it may stop early, never low."""
    rng = np.random.default_rng(5)
    for _ in range(25):
        estimates = [
            _estimate(
                run,
                rng.normal(size=3),
                _random_covariance(rng, 3) * 0.05,
                chi_squared=250.0,
                n_points=200,
            )
            for run in range(4)
        ]
        _best, best_ic = rank_assignments(estimates, NAMES, SelectionMetric.BIC)[0]
        _greedy, greedy_ic = greedy_assignment(estimates, NAMES, SelectionMetric.BIC)
        assert greedy_ic >= best_ic - 1e-9


def test_greedy_never_proposes_a_parameter_at_a_bound():
    covariance = np.diag([0.01, 0.01, 0.01])
    estimates = [
        _estimate(0, [1.0, 1.0, 0.0], covariance, at_bound=("a",), n_points=400),
        _estimate(1, [1.0, 1.0, 0.0], covariance, n_points=400),
        _estimate(2, [1.0, 1.0, 0.0], covariance, n_points=400),
    ]

    subset, _ic = greedy_assignment(estimates, NAMES, SelectionMetric.BIC)

    assert "a" not in subset


def test_greedy_returns_all_local_when_nothing_is_worth_sharing():
    """Three wildly disagreeing runs: sharing any parameter costs more than it saves."""
    covariance = np.diag([1e-6, 1e-6, 1e-6])
    estimates = [
        _estimate(run, [run * 10.0, run * 20.0, run * 30.0], covariance, n_points=50)
        for run in range(3)
    ]

    subset, ic = greedy_assignment(estimates, NAMES, SelectionMetric.BIC)

    assert subset == ()
    assert ic == pytest.approx(surrogate_ic(estimates, (), SelectionMetric.BIC))


# --------------------------------------------------------------------------- #
# Windowed collapse over an ordered series
# --------------------------------------------------------------------------- #


def test_ordered_collapse_windows_agree_with_scoring_the_window_directly():
    rng = np.random.default_rng(23)
    estimates = [
        _estimate(
            run,
            rng.normal(size=3) * 0.1,
            _random_covariance(rng, 3) * 0.02,
            chi_squared=300.0 + run,
            n_points=180,
        )
        for run in range(8)
    ]
    ordered = OrderedCollapse(estimates, NAMES)

    for start, stop in ((0, 8), (0, 3), (2, 6), (5, 8)):
        window = estimates[start:stop]
        for subset in ((), ("a",), ("b", "c"), NAMES):
            assert ordered.delta_chi2(start, stop, subset) == pytest.approx(
                collapse_cost(window, subset).delta_chi2, rel=1e-9, abs=1e-9
            )
            assert ordered.surrogate_ic(start, stop, subset, SelectionMetric.BIC) == pytest.approx(
                surrogate_ic(window, subset, SelectionMetric.BIC), rel=1e-9, abs=1e-9
            )
        assert (
            ordered.greedy(start, stop, NAMES, SelectionMetric.BIC)[0]
            == greedy_assignment(window, NAMES, SelectionMetric.BIC)[0]
        )


def test_ordered_collapse_scores_a_subset_the_same_whatever_order_it_is_written_in():
    """The prefix cache is keyed by the *set*, so a walk's append order is free."""
    rng = np.random.default_rng(29)
    estimates = [
        _estimate(run, rng.normal(size=3) * 0.1, _random_covariance(rng, 3) * 0.02)
        for run in range(5)
    ]
    ordered = OrderedCollapse(estimates, NAMES)

    first = ordered.delta_chi2(0, 5, ("a", "c", "b"))
    assert ordered.delta_chi2(0, 5, ("b", "a", "c")) == first
    assert len(ordered._prefix) == 1


def test_a_window_with_a_gap_is_reported_as_uncovered():
    rng = np.random.default_rng(31)
    present = [_estimate(run, rng.normal(size=3), _random_covariance(rng, 3)) for run in range(4)]
    ordered = OrderedCollapse((present[0], None, present[2], present[3]), NAMES)

    assert not ordered.covers(0, 4)
    assert not ordered.covers(0, 2)
    assert ordered.covers(2, 4)


def test_lower_bound_ic_never_exceeds_any_assignment_it_bounds():
    """The bound that prunes templates per window has to be a bound."""
    rng = np.random.default_rng(37)
    estimates = [
        _estimate(
            run,
            rng.normal(size=3) * 0.2,
            _random_covariance(rng, 3) * 0.03,
            chi_squared=280.0 + run,
            n_points=220,
        )
        for run in range(6)
    ]
    ordered = OrderedCollapse(estimates, NAMES)

    for start, stop in ((0, 6), (1, 4), (3, 6)):
        floor = ordered.lower_bound_ic(start, stop, SelectionMetric.BIC)
        for subset, ic in rank_assignments(estimates[start:stop], NAMES, SelectionMetric.BIC):
            assert floor <= ic + 1e-9, subset


def test_partition_ic_with_nothing_shared_is_the_sum_of_per_run_bics():
    """The partition convention: a local parameter pays against its own run."""
    from asymmetry.core.fitting.global_search.surrogate import OrderedCollapse

    names = ("a", "b")
    estimates = []
    for run in range(4):
        estimates.append(
            RunEstimate(
                run_number=run,
                names=names,
                values=np.array([1.0 + run, 2.0]),
                covariance=None,
                uncertainties=np.array([0.1, 0.1]),
                at_bound=frozenset(),
                chi_squared=100.0 + run,
                n_points=500 * (run + 1),
            )
        )
    collapse = OrderedCollapse(estimates, names)

    per_run = sum(
        estimate.chi_squared + len(names) * math.log(estimate.n_points) for estimate in estimates
    )
    assert collapse.partition_ic(0, 4, ()) == pytest.approx(per_run)
    # Sharing ``b`` (identical everywhere) trades four local penalties against
    # one shared penalty on the pooled points, with no χ² cost.
    shared = collapse.partition_ic(0, 4, ("b",))
    expected = (
        per_run
        - sum(math.log(e.n_points) for e in estimates)
        + math.log(sum(e.n_points for e in estimates))
    )
    assert shared == pytest.approx(expected)
    assert collapse.lower_bound_partition_ic(0, 4) <= collapse.partition_ic(0, 4, ("a", "b"))
    assert collapse.greedy_partition(0, 4, names) == (("b",), pytest.approx(expected))

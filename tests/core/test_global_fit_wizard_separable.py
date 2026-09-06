"""Tests for the separable global/local role-search engine.

The separable engine is the wizard's default. It differs from the exhaustive
wavefront in three ways this file pins:

* the all-local assignment is *assembled* from independent per-run fits and is
  never solved as one joint problem;
* the walk is backward elimination guided by the full-covariance GLS surrogate,
  with a monotonicity certificate that refits a parent the child proves
  mis-converged, and an escalation to the multi-start battery when a warm fit
  fails;
* the search may run at a coarser series resolution, but the winner and its
  flip-neighbourhood are always refitted at full resolution, so no assessment on
  the leaderboard mixes ``n``.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import replace

import numpy as np
import pytest

import asymmetry.core.fitting.fit_wizard as fit_wizard_module
import asymmetry.core.fitting.global_fit_wizard as global_fit_wizard_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import (
    _assessment_chi2,
    _fixed_param_names,
    _initial_parameter_sets_for_candidate,
    _local_names_for,
    _run_separable_anchor_task,
    _separable_backward_elimination,
    _separable_coupled_strategy,
    _SeparableAnchorTask,
    build_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import ParameterSet

# Every test here drives the real wizard end to end over a synthetic series.
pytestmark = [pytest.mark.integration]

_TEMPLATE_KEY = "exp_constant"


def _dataset(
    run_number: int,
    *,
    model: CompositeModel,
    params: dict[str, float],
    field: float = 0.0,
    temperature: float = 5.0,
    n_points: int = 120,
) -> MuonDataset:
    time = np.linspace(0.0, 8.0, n_points)
    return MuonDataset(
        time=time,
        asymmetry=model.function(time, **params),
        error=np.full_like(time, 0.01),
        metadata={
            "run_number": run_number,
            "field": field,
            "temperature": temperature,
            "run_label": str(run_number),
        },
    )


def _restrict_to_exp_constant(
    monkeypatch: pytest.MonkeyPatch,
    model: CompositeModel,
) -> CandidateTemplate:
    template = CandidateTemplate(
        key=_TEMPLATE_KEY,
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    monkeypatch.setattr(
        global_fit_wizard_module,
        "build_candidate_templates",
        lambda fingerprint, current_model=None: (template,),
    )
    return template


def _all_global_series(model: CompositeModel, *, n_points: int = 120) -> list[MuonDataset]:
    """Planted truth: every parameter identical across the series."""
    return [
        _dataset(
            800 + index,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
            temperature=5.0 * index,
            n_points=n_points,
        )
        for index in range(1, 5)
    ]


def _local_lambda_series(model: CompositeModel, *, n_points: int = 120) -> list[MuonDataset]:
    """Planted truth: shared amplitude and background, a rate that scans."""
    lambdas = (0.15, 0.25, 0.55, 0.9)
    return [
        _dataset(
            850 + index,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[index - 1], "A_bg": 0.01},
            temperature=5.0 * index,
            n_points=n_points,
        )
        for index in range(1, 5)
    ]


def _counters(instrumentation: dict[str, object]) -> dict[str, int]:
    counters = instrumentation.get("counters")
    assert isinstance(counters, dict)
    return {str(name): int(value) for name, value in counters.items()}


def _anchor_task(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    metric: SelectionMetric = SelectionMetric.AICC,
) -> _SeparableAnchorTask:
    """One template's all-local anchor task, built the way the search builds it."""
    fingerprints = {
        int(dataset.run_number): global_fit_wizard_module.fingerprint_spectrum(dataset)
        for dataset in datasets
    }
    fixed_param_names = _fixed_param_names(template, {})
    base_by_run = _initial_parameter_sets_for_candidate(
        datasets,
        fingerprints,
        template,
        current_values={},
        parameter_bounds={},
        fixed_param_names=fixed_param_names,
    )
    return _SeparableAnchorTask(
        template_key=template.key,
        template=template,
        datasets=datasets,
        base_by_run=base_by_run,
        fixed_param_names=fixed_param_names,
        axis_key="temperature",
        metric=metric,
        search_strategy="staged_v2",
        prescreen_results_by_run=None,
    )


# --------------------------------------------------------------------------- #
# Planted-role recovery
# --------------------------------------------------------------------------- #


def test_separable_engine_shares_every_parameter_on_a_uniform_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)

    recommendation = build_global_fit_wizard_recommendation(_all_global_series(model))

    assessment = recommendation.recommended_assessment
    assert assessment is not None
    assert assessment.template.key == _TEMPLATE_KEY
    assert assessment.local_param_names == ()


def test_separable_engine_localizes_only_the_scanning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)

    recommendation = build_global_fit_wizard_recommendation(_local_lambda_series(model))

    assessment = recommendation.recommended_assessment
    assert assessment is not None
    assert "Lambda" in assessment.local_param_names
    assert "A_1" not in assessment.local_param_names
    assert "A_bg" not in assessment.local_param_names


# --------------------------------------------------------------------------- #
# The all-local node is assembled, never fitted jointly
# --------------------------------------------------------------------------- #


def test_all_local_node_is_assembled_from_per_run_fits_never_fitted_jointly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No coupled fit is ever spent on the all-local assignment.

    It is the most expensive node the exhaustive wavefront solves — one joint
    problem over ``P * G`` parameters — and phase 1 already holds the answer.
    Every ``_fit_exact_assignment`` call is recorded here; none of them may carry
    the all-local role split, while the node still reaches the leaderboard.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)
    datasets = _local_lambda_series(model)
    free_names = tuple(model.param_names)
    all_local = _local_names_for(free_names, set(free_names))

    fitted_splits: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    real_fit = global_fit_wizard_module._fit_exact_assignment

    def _recording_fit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        fitted_splits.append((kwargs["global_param_names"], kwargs["local_param_names"]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(global_fit_wizard_module, "_fit_exact_assignment", _recording_fit)

    instrumentation: dict[str, object] = {}
    recommendation = build_global_fit_wizard_recommendation(
        datasets, instrumentation=instrumentation
    )

    assert ((), all_local) not in fitted_splits
    counters = _counters(instrumentation)
    # One independent single-run fit per dataset built the node instead.
    assert counters["separable_all_local_fits"] == len(datasets)
    optimized = recommendation.optimized_assessments()
    assert any(
        assessment.local_param_names == all_local and assessment.is_successful
        for assessment in optimized
    ), "the all-local node must still reach the leaderboard with a real score"


def test_separable_search_costs_far_fewer_coupled_fits_than_exhaustive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: O(P) coupled fits per template rather than O(2^P)."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)
    datasets = _local_lambda_series(model)

    separable: dict[str, object] = {}
    exhaustive: dict[str, object] = {}
    build_global_fit_wizard_recommendation(datasets, instrumentation=separable)
    build_global_fit_wizard_recommendation(
        datasets, instrumentation=exhaustive, search_engine="exhaustive"
    )

    assert _counters(separable)["global_fit_calls"] < _counters(exhaustive)["global_fit_calls"]


# --------------------------------------------------------------------------- #
# Flip neighbourhood and role recommendations
# --------------------------------------------------------------------------- #


def test_winner_flip_neighbourhood_is_fitted_and_justifies_every_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)
    datasets = _local_lambda_series(model)

    recommendation = build_global_fit_wizard_recommendation(datasets)
    winner = recommendation.recommended_assessment
    assert winner is not None

    present = {
        (assessment.global_param_names, assessment.local_param_names)
        for assessment in recommendation.optimized_assessments()
    }
    free_names = tuple(model.param_names)
    winner_local = set(winner.local_param_names)
    for name in free_names:
        if name in winner_local:
            flipped = _local_names_for(free_names, winner_local - {name})
        else:
            flipped = _local_names_for(free_names, winner_local | {name})
        flipped_global = tuple(n for n in free_names if n not in set(flipped))
        assert (flipped_global, flipped) in present, f"missing flip neighbour for {name}"

    recommended_names = {
        recommendation_.name for recommendation_ in winner.parameter_recommendations
    }
    assert recommended_names == set(free_names)


# --------------------------------------------------------------------------- #
# Monotonicity certificate and escalation
# --------------------------------------------------------------------------- #


def test_certificate_violation_refits_a_mis_converged_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that beats its parent's chi-squared indicts the *parent*.

    The child shares one more parameter, so it has fewer free parameters and
    cannot honestly fit better. Here the all-local anchor is handed a chi-squared
    (and IC) inflated by 500 — exactly what a parent stuck in a worse minimum
    reports — so the first elimination step must trip the certificate and refit
    it from the child's values.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = _restrict_to_exp_constant(monkeypatch, model)
    datasets = _all_global_series(model)

    anchor_result = _run_separable_anchor_task(_anchor_task(datasets, template))
    state = anchor_result.state
    anchor_key = ((), state.anchor_assessment.local_param_names)
    honest_chi2 = _assessment_chi2(state.anchor_assessment)

    damaged = replace(
        state.anchor_assessment,
        fit_results_by_run={
            run_number: replace(result, chi_squared=result.chi_squared + 500.0)
            for run_number, result in state.anchor_assessment.fit_results_by_run.items()
        },
        aic=state.anchor_assessment.aic + 500.0,
        aicc=state.anchor_assessment.aicc + 500.0,
        bic=state.anchor_assessment.bic + 500.0,
        selected_score=state.anchor_assessment.selected_score + 500.0,
    )
    state.anchor_assessment = damaged
    state.exact_cache[anchor_key] = damaged
    state.converged_assessments[anchor_key] = damaged
    state.best_assessment = damaged

    instrumentation: dict[str, object] = {"counters": {}}
    _separable_backward_elimination(
        datasets,
        state,
        anchor_result.estimates,
        axis_key="temperature",
        metric=SelectionMetric.AICC,
        search_strategy="staged_v2",
        instrumentation=instrumentation,
    )

    assert _counters(instrumentation)["separable_certificate_refits"] >= 1
    repaired = state.exact_cache[anchor_key]
    assert _assessment_chi2(repaired) < _assessment_chi2(damaged)
    # The refit lands at (or below) the honest minimum the anchor should have
    # reached, never merely somewhere between the two.
    assert _assessment_chi2(repaired) <= honest_chi2 + 1e-6


def test_a_failed_warm_fit_escalates_to_the_multi_start_battery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the single warm fit does not converge, the full battery runs.

    The warm collapse start is the fast path, not the only path: a node whose
    warm fit fails must still be given the staged multi-start attempts before the
    walk gives up on it.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)
    datasets = _all_global_series(model)

    real_warm_fit = global_fit_wizard_module._warm_certificate_fit
    state = {"failures": 0}

    def _first_warm_fit_fails(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        if state["failures"] == 0:
            state["failures"] += 1
            return None, ParameterSet(), float("inf"), {}
        return real_warm_fit(*args, **kwargs)

    monkeypatch.setattr(global_fit_wizard_module, "_warm_certificate_fit", _first_warm_fit_fails)

    instrumentation: dict[str, object] = {}
    recommendation = build_global_fit_wizard_recommendation(
        datasets, instrumentation=instrumentation
    )

    counters = _counters(instrumentation)
    assert counters["warm_only_escalations"] >= 1
    assert counters["separable_escalations"] >= 1
    # The escalation recovered: the planted all-global truth is still found.
    assert recommendation.recommended_assessment is not None
    assert recommendation.recommended_assessment.local_param_names == ()


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_search_resolution_never_reaches_the_leaderboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebinned search fits are refitted at full resolution before reporting.

    An IC computed over a rebinned ``n`` cannot be compared with one computed
    over the full record, so every returned assessment's per-run ``dof`` must
    reflect the original point count.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)
    datasets = _local_lambda_series(model)
    full_n_points = datasets[0].n_points
    # Zero field and no detected lines leave only the cost constraint, so the
    # series rebin factor is n // budget.
    monkeypatch.setattr(fit_wizard_module, "_FIT_SAMPLE_BUDGET", full_n_points // 3)

    instrumentation: dict[str, object] = {}
    recommendation = build_global_fit_wizard_recommendation(
        datasets, instrumentation=instrumentation
    )

    counters = _counters(instrumentation)
    assert instrumentation["separable_search_rebin_factor"] > 1
    assert counters["separable_search_rebin_applied"] >= 1
    assert counters["decimation_full_res_refits"] >= 1

    for assessment in recommendation.optimized_assessments():
        n_free = len(assessment.global_param_names) + len(assessment.local_param_names)
        for result in assessment.fit_results_by_run.values():
            assert result.dof == pytest.approx(full_n_points - n_free, abs=2)


def test_coupled_strategy_is_the_sparse_least_squares_solver_at_every_width() -> None:
    """Every coupled node goes to the sparse solver, short series and wide alike.

    A coupled node's Jacobian is arrow-shaped at any width, and the trust-region
    solver exploits that shape; neither Minuit architecture does. On a wide node
    the two Minuit paths are the difference between a search that finishes and
    one that keeps nothing, and on a short one the sparse solve is already at
    least as cheap, so there is no width at which they win back the node.
    """
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    short_series = _all_global_series(model)
    long_series = [
        _dataset(900 + index, model=model, params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01})
        for index in range(1, 15)
    ]

    for series in (short_series, long_series):
        assert _separable_coupled_strategy(series, ("A_1",), ("Lambda", "A_bg")) == "least_squares"


def test_coupled_nodes_are_counted_as_least_squares_fits() -> None:
    """The instrumentation records the solver each coupled node actually used."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = _all_global_series(model)

    instrumentation: dict[str, object] = {}
    build_global_fit_wizard_recommendation(datasets, instrumentation=instrumentation)

    counters = _counters(instrumentation)
    assert counters["separable_least_squares_fits"] >= 1
    assert "separable_profiled_fits" not in counters
    assert "separable_joint_fits" not in counters


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_cancel_mid_search_raises_and_leaves_no_pool_orphans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel is honoured inside the separable search, and the pool is reaped."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    _restrict_to_exp_constant(monkeypatch, model)
    datasets = _all_global_series(model)

    polls = {"calls": 0}

    def _cancel_after_first_poll() -> bool:
        polls["calls"] += 1
        return polls["calls"] > 1

    with pytest.raises(FitCancelledError):
        build_global_fit_wizard_recommendation(datasets, cancel_callback=_cancel_after_first_poll)

    assert polls["calls"] > 1
    assert multiprocessing.active_children() == []

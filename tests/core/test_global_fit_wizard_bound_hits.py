"""Wizard bridges preserve bound identities used by sharing proposals."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import global_fit_wizard as wizard
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric, _bound_hit_names
from asymmetry.core.fitting.global_search.surrogate import (
    OrderedCollapse,
    greedy_assignment,
    rank_assignments,
)
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


@pytest.mark.parametrize("bridge", ["prescreen", "separable"])
def test_wizard_bridges_exclude_bound_hits_from_sharing_proposals(bridge, monkeypatch):
    model = CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"])
    template = CandidateTemplate(
        key="biexp_constant",
        title="Two exponentials",
        category="General",
        rationale="test",
        model=model,
    )
    free_names = ("A_1", "Lambda_1", "A_2")
    datasets = []
    results = {}
    for run in (1, 2):
        time = np.linspace(0.0, 8.0, 100)
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=np.zeros_like(time),
                error=np.full_like(time, 0.01),
                metadata={"run_number": run},
            )
        )
        # Each side is hit in only one run; the other run is strictly interior.
        parameters = ParameterSet(
            [
                Parameter("A_1", 0.0 if run == 1 else 0.2, min=0.0, max=1.0),
                Parameter("Lambda_1", 1.0 if run == 2 else 0.8, min=0.0, max=1.0),
                Parameter("A_2", 0.3, min=0.0, max=1.0),
                Parameter("Lambda_2", 0.0, min=0.0, max=1.0, fixed=True),
                Parameter("A_bg", 0.0, min=0.0, max=1.0, expr="Lambda_2"),
            ]
        )
        results[run] = FitResult(
            success=True,
            objective=Objective.GAUSSIAN,
            chi_squared=100.0,
            parameters=parameters,
            uncertainties=dict.fromkeys(free_names, 0.1),
            covariance=np.eye(3) * 0.01,
            covariance_parameters=list(free_names),
        )

    assessment = wizard.GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=results,
        global_parameters=ParameterSet(),
        global_param_names=(),
        local_param_names=free_names,
        fixed_param_names=("Lambda_2", "A_bg"),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        sample_count=200,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )

    if bridge == "prescreen":
        _, by_template, _ = wizard._partition_inputs_from_prescreen(
            datasets, {template.key: assessment}, analysed_points_by_run={1: 100, 2: 100}
        )
        estimates = tuple(by_template[template.key].values())
    else:
        estimates = wizard._run_estimates_from_results(datasets, results, free_names)

    proposed = {
        subset for subset, _ in rank_assignments(estimates, free_names, SelectionMetric.BIC)
    }
    assert proposed == {(), ("A_2",)}
    assert greedy_assignment(estimates, free_names, SelectionMetric.BIC)[0] == ("A_2",)
    assert OrderedCollapse(estimates, free_names).greedy_partition(0, 2, free_names)[0] == ("A_2",)
    assert estimates[0].at_bound == frozenset({"A_1"})
    assert estimates[1].at_bound == frozenset({"Lambda_1"})
    assert _bound_hit_names(results[1].parameters) == ["A_1 at lower bound"]
    assert _bound_hit_names(results[2].parameters) == ["Lambda_1 at upper bound"]

    assignments = []

    def fit_assignment(datasets, state, local_names, **kwargs):
        assignments.append(local_names)
        return replace(assessment, fit_results_by_run={})

    monkeypatch.setattr(wizard, "_fit_separable_assignment", fit_assignment)
    state = wizard._HeuristicTemplateState(
        template=template,
        fixed_param_names=assessment.fixed_param_names,
        prefit_base_by_run={},
        free_param_names=free_names,
        exact_cache={},
        converged_assessments={},
        anchor_assessment=assessment,
    )
    assert (
        wizard._separable_backward_elimination(
            datasets,
            state,
            estimates,
            axis_key="temperature",
            metric=SelectionMetric.BIC,
            search_strategy="balanced",
            instrumentation={},
        )
        is assessment
    )
    assert assignments == [("A_1", "Lambda_1")]

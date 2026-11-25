"""Ranked sharing proposals must be checked against the exact incumbent."""

from dataclasses import replace

import pytest

from asymmetry.core.fitting import global_fit_wizard as wizard
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import ParameterSet


@pytest.mark.parametrize(
    "first", ["rejected", "failed", "accepted", "all_rejected", "parent_refit", "nonfinite"]
)
def test_elimination_tries_ranked_alternatives_and_terminates(monkeypatch, first):
    names = ("A_1", "Lambda", "A_bg")
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    anchor = wizard.GlobalCandidateAssessment(
        template=template,
        fit_results_by_run={
            1: FitResult(success=True, objective=Objective.GAUSSIAN, chi_squared=100.0)
        },
        global_parameters=ParameterSet(),
        global_param_names=(),
        local_param_names=names,
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        sample_count=100,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )
    state = wizard._HeuristicTemplateState(
        template=template,
        fixed_param_names=(),
        prefit_base_by_run={},
        free_param_names=names,
        exact_cache={},
        converged_assessments={},
        anchor_assessment=anchor,
    )
    monkeypatch.setattr(
        wizard, "surrogate_ic", lambda estimates, shared, metric: names.index(shared[-1])
    )
    monkeypatch.setattr(
        wizard, "_assessment_sort_key", lambda assessment, metric: assessment.sample_count
    )
    if first == "nonfinite":
        monkeypatch.setattr(wizard, "surrogate_ic", lambda *args: float("inf"))
    refitted = replace(anchor, sample_count=80)
    parent_refits = []

    def refit_parent(datasets, state, parent, child, **kwargs):
        parent_refits.append(parent)
        return refitted

    monkeypatch.setattr(wizard, "_refit_certificate_violating_parent", refit_parent)
    attempts = []
    seeds = []

    def fit_assignment(datasets, state, local_names, *, warm_start_source, **kwargs):
        shared = tuple(name for name in names if name not in local_names)
        attempts.append((shared, warm_start_source.global_param_names))
        seeds.append(warm_start_source)
        accepted = ("A_1",) if first == "accepted" else ("Lambda",)
        score = 90 if shared == accepted and first != "all_rejected" else 110
        results = anchor.fit_results_by_run
        if first == "parent_refit" and shared == ("A_1",):
            results = {1: replace(results[1], chi_squared=90.0)}
            score = 90
        if first == "parent_refit" and shared == ("Lambda",):
            score = 70
        return replace(
            anchor,
            global_param_names=shared,
            local_param_names=local_names,
            sample_count=score,
            fit_results_by_run={} if first == "failed" and shared == ("A_1",) else results,
        )

    monkeypatch.setattr(wizard, "_fit_separable_assignment", fit_assignment)
    result = wizard._separable_backward_elimination(
        [],
        state,
        (),
        axis_key="temperature",
        metric=SelectionMetric.AICC,
        search_strategy="staged_v2",
        instrumentation={},
    )
    if first == "nonfinite":
        assert result is anchor
        assert attempts == []
    elif first == "all_rejected":
        assert result is anchor
        assert attempts == [((name,), ()) for name in names]
    else:
        accepted = ("A_1",) if first == "accepted" else ("Lambda",)
        assert result.global_param_names == accepted
        prefix = [(("A_1",), ())]
        if first != "accepted":
            prefix.append((("Lambda",), ()))
        remaining = [name for name in names if name not in accepted]
        assert attempts == prefix + [
            (tuple(name for name in names if name in {*accepted, addition}), accepted)
            for addition in remaining
        ]
    if first == "parent_refit":
        assert parent_refits == [anchor]
        assert seeds[1] is refitted

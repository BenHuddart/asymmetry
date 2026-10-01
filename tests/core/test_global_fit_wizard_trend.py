"""The Global Fit Wizard under the trend objective (plan D1, D7–D9, D13–D14, D17).

The series are the sharing ladder's own (``tests/core/sharing_series.py``), run
through the whole wizard: all-local fits, the template band, one ladder per
competing template, and the recommendation.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalFitWizardRecommendation,
    build_global_fit_wizard_recommendation,
    deserialize_global_fit_wizard_recommendation,
    merge_global_fit_wizard_recommendations,
    rerank_global_fit_wizard_recommendation,
    serialize_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.trend_objective import (
    TEMPLATE_BAND,
    SelectionObjective,
    templates_within_band,
    trend_contenders,
)
from asymmetry.core.fitting.global_search.trend_search import (
    _LadderTask,
    _lines_vanish,
    _prescreen_fits_for_ladder,
    _run_ladder_task,
)
from tests.core.sharing_series import (
    TRANSITION,
    SimulatedSeries,
    glassy_series,
    hopping_series,
    two_line_series,
)


def _recommend(series: SimulatedSeries, *keys: str, **options) -> GlobalFitWizardRecommendation:
    return build_global_fit_wizard_recommendation(
        series.datasets, selected_template_keys=keys, **options
    )


@pytest.fixture(scope="module")
def hopping() -> tuple[SimulatedSeries, GlobalFitWizardRecommendation]:
    """Copper's case: a shared static width under a rising hop rate, with a poor rival."""
    series = hopping_series()
    return series, _recommend(series, "dynamic_gkt_constant", "exp_constant")


def test_trend_is_the_default_and_recommends_the_rung_that_trends_best(hopping) -> None:
    _series, recommendation = hopping
    recommended = recommendation.recommended_assessment

    assert recommendation.objective is SelectionObjective.TREND
    assert recommended.template.key == "dynamic_gkt_constant"
    assert recommended.global_param_names == ("A_bg", "A_1", "Delta")
    assert set(recommended.rung.trend.parameters) == {"nu"}
    assert recommended.rung.preselected
    assert recommended.rung.within_tolerance
    assert "A_bg, A_1, Delta shared" in recommendation.summary
    assert "nu varying from run to run" in recommendation.summary
    assert recommendation.sorted_optimized_assessments()[0] is recommended


def test_every_rung_is_a_candidate_with_its_cost_and_its_curves(hopping) -> None:
    series, recommendation = hopping
    rungs = [
        assessment
        for assessment in recommendation.assessments_for_template_key("dynamic_gkt_constant")
        if assessment.rung is not None
    ]

    assert [rung.global_param_names for rung in rungs] == [
        (),
        ("A_bg",),
        ("A_bg", "A_1"),
        ("A_bg", "A_1", "Delta"),
        ("A_bg", "A_1", "Delta", "nu"),
    ]
    assert rungs[0].rung.series_cost == 0.0
    assert [rung.rung.preselected for rung in rungs] == [False, False, False, True, False]
    # A hop rate that rises by two decades cannot be shared, and the rung says so.
    assert rungs[-1].rung.series_cost > 100.0
    assert not rungs[-1].rung.within_tolerance
    assert not rungs[-1].rung.trend.is_trend
    assert len({rung.selection_key for rung in rungs}) == len(rungs)
    assert len({rung.rung.all_local_chi2r for rung in rungs}) == 1
    runs = {int(dataset.run_number) for dataset in series.datasets}
    for rung in rungs:
        assert set(rung.fitted_curves_by_run) == set(rung.component_curves_by_run) == runs
        assert all(result.residuals is not None for result in rung.fit_results_by_run.values())
        assert math.isfinite(rung.aicc)


def test_template_outside_the_band_is_listed_and_does_not_contend(hopping) -> None:
    _series, recommendation = hopping
    rungs = [assessment for assessment in recommendation.assessments if assessment.rung]
    chi2r = {assessment.template.key: assessment.rung.all_local_chi2r for assessment in rungs}

    # The user ticked the exponential, so it was climbed; it fits far worse.
    assert chi2r["exp_constant"] > (1.0 + TEMPLATE_BAND) * chi2r["dynamic_gkt_constant"]
    assert templates_within_band(chi2r) == {"dynamic_gkt_constant"}
    assert recommendation.comparable_keys == ()
    # However well its parameters trend, a template outside the band cannot win.
    flattered = [
        replace(
            assessment,
            rung=replace(
                assessment.rung,
                trend=recommendation.recommended_assessment.rung.trend,
                hard_to_justify=(),
            ),
            aicc=-math.inf,
        )
        if assessment.template.key == "exp_constant"
        else assessment
        for assessment in rungs
    ]
    contenders = trend_contenders(flattered, SelectionMetric.AICC)
    assert [assessment.template.key for assessment in contenders] == ["dynamic_gkt_constant"]


def test_failed_residual_gate_is_a_caveat_on_the_recommendation(hopping) -> None:
    _series, recommendation = hopping
    recommended = recommendation.recommended_assessment
    failing = replace(
        recommended,
        run_diagnostics=(
            replace(recommended.run_diagnostics[0], gate_passed=False, gate_reasons=("bad",)),
            *recommended.run_diagnostics[1:],
        ),
    )
    reranked = rerank_global_fit_wizard_recommendation(
        replace(
            recommendation,
            assessments=tuple(
                failing if assessment is recommended else assessment
                for assessment in recommendation.assessments
            ),
        ),
        SelectionMetric.BIC,
    )

    assert not failing.residual_gate_passed
    assert reranked.recommended_key == recommended.selection_key
    assert f"structured residuals on runs {failing.run_diagnostics[0].run_label};" in (
        reranked.summary
    )


def test_trend_recommendation_survives_a_project_round_trip(hopping) -> None:
    _series, recommendation = hopping
    restored = deserialize_global_fit_wizard_recommendation(
        json.loads(
            json.dumps(serialize_global_fit_wizard_recommendation(recommendation, compact=True))
        )
    )

    assert restored.objective is SelectionObjective.TREND
    assert [assessment.rung for assessment in restored.assessments] == [
        assessment.rung for assessment in recommendation.assessments
    ]
    reranked = rerank_global_fit_wizard_recommendation(restored, recommendation.metric)
    assert reranked.recommended_key == recommendation.recommended_key
    assert reranked.summary == recommendation.summary


def test_changing_the_objective_replaces_the_optimised_candidates(hopping) -> None:
    series, recommendation = hopping
    statistical = _recommend(
        series, "dynamic_gkt_constant", objective=SelectionObjective.STATISTICAL
    )

    merged = merge_global_fit_wizard_recommendations(recommendation, statistical)

    assert merged.objective is SelectionObjective.STATISTICAL
    assert merged.optimized_assessments()
    assert all(assessment.rung is None for assessment in merged.optimized_assessments())


def test_role_search_engine_is_refused_under_the_trend_objective() -> None:
    with pytest.raises(ValueError, match="statistical objective"):
        build_global_fit_wizard_recommendation(hopping_series().datasets, search_engine="separable")


def test_ladder_stops_at_the_amplitudes_once_the_budget_has_run_out(hopping) -> None:
    series, recommendation = hopping
    anchor = next(
        assessment
        for assessment in recommendation.assessments_for_template_key("dynamic_gkt_constant")
        if assessment.rung is not None and not assessment.global_param_names
    )
    task = _LadderTask(
        template_key=anchor.template.key,
        template=anchor.template,
        datasets=series.datasets,
        anchor=replace(anchor, rung=None),
        base_by_run={run: result.parameters for run, result in anchor.fit_results_by_run.items()},
        fixed_param_names=(),
        axis_key=recommendation.series_axis_key,
        metric=recommendation.metric,
        budget_ends=0.0,
    )

    spent = _run_ladder_task(task)
    unlimited = _run_ladder_task(replace(task, budget_ends=math.inf))

    assert spent.restricted
    assert [rung.global_param_names for rung in spent.assessments] == [
        (),
        ("A_bg",),
        ("A_bg", "A_1"),
    ]
    assert not unlimited.restricted
    assert len(unlimited.assessments) == 5


def test_fits_restored_from_a_project_are_not_what_a_ladder_starts_from(hopping) -> None:
    series, recommendation = hopping
    anchor = next(assessment for assessment in recommendation.assessments if assessment.rung)
    screening_row = replace(anchor, prescreen_only=True, rung=None)
    restored = replace(
        screening_row,
        fit_results_by_run={
            run: replace(result, dof=0, covariance=None)
            for run, result in anchor.fit_results_by_run.items()
        },
    )

    assert _prescreen_fits_for_ladder(series.datasets, screening_row, resolution_matches=True)
    assert _prescreen_fits_for_ladder(series.datasets, restored, resolution_matches=True) is None


def test_isolated_anomalous_runs_are_exempt_and_counted() -> None:
    series = hopping_series(runs=12, amplitude_scale={3: 1.1, 8: 1.1})
    recommendation = _recommend(series, "dynamic_gkt_constant")
    recommended = recommendation.recommended_assessment

    assert recommended.global_param_names == ("A_bg", "A_1", "Delta")
    assert recommended.rung.exempt_runs == (3, 8)
    # ``A_1`` is Global, and the candidate says which runs it does not hold for.
    assert recommended.exemptions == {"A_1": (3, 8)}
    assert recommended.parameter_count == 3 + 2 + len(recommended.local_param_names) * 12
    assert "Runs 3, 8 keep their own A_1" in recommendation.summary


def test_end_block_the_amplitude_cannot_be_shared_through_is_reported() -> None:
    series = glassy_series(runs_with_lost_asymmetry=3)
    recommendation = _recommend(series, "stretched_constant")
    recommended = recommendation.recommended_assessment

    assert recommended.global_param_names == ("A_bg",)
    # A finding of the climb, carried by every rung of the template's ladder.
    assert {
        assessment.rung.amplitude_unshareable_runs
        for assessment in recommendation.assessments
        if assessment.rung
    } == {(1, 2, 3)}
    assert "could not be shared through runs 1, 2, 3" in recommendation.summary


def test_shared_total_rung_carries_the_model_it_was_fitted_in() -> None:
    series = two_line_series(TRANSITION, fixed=())
    recommendation = build_global_fit_wizard_recommendation(
        series.datasets,
        series.model,
        current_values={parameter.name: parameter.value for parameter in series.base_by_run[1]},
        current_parameter_types={"phase_1": "Fixed", "phase_3": "Fixed"},
        selected_template_keys=("current_model",),
    )
    recommended = recommendation.recommended_assessment
    grouped = recommended.template.model

    # The same template key, written with both lines under one total.
    assert recommended.template.key == "current_model"
    assert grouped.fraction_groups
    assert grouped.param_names != series.model.param_names
    assert recommended.global_param_names[:2] == ("A_bg", "A_1")
    assert {"f_Oscillatory", "sigma_1", "sigma_2"} <= set(recommended.rung.trend.parameters)
    # The pinned phases are named as the fraction form names them.
    assert recommended.fixed_param_names == ("phase_1", "phase_2")
    assert set(recommended.applied_roles) == set(grouped.param_names)
    for run, (time, curve) in recommended.fitted_curves_by_run.items():
        values = {
            parameter.name: parameter.value
            for parameter in recommended.fit_results_by_run[run].parameters
        }
        assert curve == pytest.approx(grouped.function(time, **values))

    restored = deserialize_global_fit_wizard_recommendation(
        json.loads(json.dumps(serialize_global_fit_wizard_recommendation(recommendation)))
    )
    assert restored.recommended_assessment.template.model.param_names == grouped.param_names
    assert restored.recommended_assessment.rung == recommended.rung


def test_rung_whose_lines_have_vanished_on_a_run_is_recognised_in_either_form() -> None:
    """The oscillatory rule reads a fraction-form rung through its amplitudes."""
    template = CandidateTemplate(
        key="oscillatory2_gaussian_constant",
        title="Two lines",
        category="Oscillatory",
        rationale="test",
        model=two_line_series(TRANSITION).model,
    )
    present = two_line_series(TRANSITION).climb(further=0)
    shared_total = next(rung for rung in present.rungs if rung.model.fraction_groups)

    assert shared_total.adequate
    assert not _lines_vanish(template, present.rungs[0])
    assert not _lines_vanish(template, shared_total)

    # The same rungs with every line's amplitude left without an error on one
    # run: nothing there is a measured line.
    def unmeasured(rung, names):
        results = dict(rung.results_by_run)
        results[1] = replace(
            results[1],
            uncertainties={
                name: error for name, error in results[1].uncertainties.items() if name not in names
            },
        )
        return replace(rung, results_by_run=results)

    assert _lines_vanish(template, unmeasured(present.rungs[0], {"A_1", "A_3"}))
    assert _lines_vanish(template, unmeasured(shared_total, {"f_Oscillatory"}))

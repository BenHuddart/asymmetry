"""Core consumers agree on availability and count actual fitted columns."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import CandidateAssessment, CandidateTemplate
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    RunResidualDiagnostic,
)
from asymmetry.core.fitting.global_search.surrogate import metric_penalty
from asymmetry.core.fitting.global_search.types import ModelScore
from asymmetry.core.fitting.information_criteria import InformationCriteria, SelectionMetric
from asymmetry.core.fitting.parameters import AffineTie, Parameter, ParameterSet


def _single_statistic(statistic, k, n):
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    parameters = ParameterSet(
        [Parameter(name, 1.0, fixed=i >= k) for i, name in enumerate(model.param_names)]
    )
    return CandidateAssessment(
        template=CandidateTemplate("criteria", "Criteria", "General", "", model),
        fit_result=FitResult(success=True, chi_squared=statistic, parameters=parameters),
        sample_count=n,
        residual_rms=0.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=np.array([]),
        fitted_curve=np.array([]),
        component_curves=(),
    )


@pytest.mark.parametrize("statistic,k,n", [(1.0, 3, 4), (1.0, 3, 5), (10.0, 1, 4)])
@pytest.mark.parametrize("metric", list(SelectionMetric))
def test_core_consumers_agree(statistic, k, n, metric):
    assessment = _single_statistic(statistic, k, n)
    criteria = InformationCriteria(statistic, k, n)
    score = ModelScore(criteria=criteria, primary_metric=metric)
    expected = criteria.value(metric)
    penalty = metric_penalty(k, sample_count=n, metric=metric)
    assert assessment.criteria == criteria
    if expected is None:
        assert assessment.metric_value(metric) == float("inf")
        assert score.primary_value == float("inf")
        assert penalty is None
        assert assessment.unavailable_reason(metric) == criteria.unavailable_reason(metric)
    else:
        assert assessment.metric_value(metric) == expected
        assert score.primary_value == expected
        assert statistic + penalty == pytest.approx(expected)


def test_prescreen_matches_coupled_all_local_totals():
    single = _single_statistic(10.0, 3, 10)
    names = tuple(single.template.model.param_names)
    diagnostics = tuple(
        RunResidualDiagnostic(
            run_number=run,
            run_label=str(run),
            axis_value=float(run),
            residual_rms=0.0,
            runs_z_score=0.0,
            max_abs_autocorrelation=0.0,
            residual_fft_peak_snr=0.0,
            gate_passed=True,
            gate_reasons=(),
        )
        for run in (1, 2, 3)
    )
    coupled = GlobalCandidateAssessment(
        template=single.template,
        fit_results_by_run={run: single.fit_result for run in (1, 2, 3)},
        global_parameters=ParameterSet(),
        global_param_names=(),
        local_param_names=names,
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=diagnostics,
        series_warnings=(),
        sample_count=30,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )
    prescreen = replace(
        coupled, prescreen_only=True, global_param_names=names, local_param_names=()
    )
    assert coupled.parameter_count == prescreen.parameter_count == 9
    for row in (coupled, prescreen):
        assert [row.criteria.value(metric) for metric in SelectionMetric] == pytest.approx(
            [48.0, 57.0, 60.6107764349594], abs=1e-12
        )


def test_single_count_excludes_fixed_link_and_tie_followers_and_keeps_free_auxiliary():
    single = _single_statistic(10.0, 3, 100)
    parameters = ParameterSet(
        [
            Parameter("a", 1.0, link_group=1),
            Parameter("b", 1.0, link_group=1),
            Parameter("fixed", 0.0, fixed=True),
            Parameter("auxiliary", 0.1),
            Parameter("tied", 1.1, tie=AffineTie(main="a", offset="auxiliary")),
        ]
    )
    row = replace(single, fit_result=replace(single.fit_result, parameters=parameters))
    assert row.parameter_count == 2
    assert row.criteria.value(SelectionMetric.AIC) == 14.0


def test_cross_group_unit_weight_scale_invariance_and_x_error_exclusion():
    from asymmetry.core.fitting.cross_group_roles import CrossGroupCandidate
    from asymmetry.core.fitting.information_criteria import rank
    from asymmetry.core.fitting.parameter_models import CrossGroupFitResult, ErrorMode

    deltas = []
    for scale in (1.0, 0.1, 1e3):
        candidates = [
            CrossGroupCandidate(
                tuple(f"p{i}" for i in range(k)),
                (),
                (),
                CrossGroupFitResult(
                    success=True,
                    chi_squared=rss * scale**2,
                    reduced_chi_squared=1.0,
                    n_points=20,
                    error_mode=ErrorMode.NONE.value,
                    global_parameters=ParameterSet([Parameter(f"p{i}", 0.0) for i in range(k)]),
                ),
            )
            for rss, k in [(2.0, 3), (5.0, 2)]
        ]
        deltas.append(candidates[1].aicc - candidates[0].aicc)
    assert deltas == pytest.approx([15.159147970816434] * 3)
    excluded = replace(
        candidates[0], result=replace(candidates[0].result, uses_effective_variance=True)
    )
    eligible, reasons = rank([excluded], SelectionMetric.AICC)
    assert not eligible
    assert reasons == [
        (excluded, "effective-variance objective omits its likelihood normalization")
    ]


def test_single_wizard_has_no_recommendation_when_aicc_is_unavailable():
    from asymmetry.core.fitting.fit_wizard import (
        FitWizardRecommendation,
        rerank_fit_wizard_recommendation,
    )
    from tests.core.test_fit_wizard_tiered import _plain_fingerprint

    row = _single_statistic(1.0, 3, 4)
    recommendation = FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=(row.template,),
        assessments=(row,),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
    )
    ranked = rerank_fit_wizard_recommendation(recommendation, SelectionMetric.AICC)
    assert ranked.recommended_key is None


def test_restored_single_recommendation_cannot_publish_an_undefined_score():
    from asymmetry.core.fitting.fit_wizard import (
        FitWizardRecommendation,
        deserialize_fit_wizard_recommendation,
        serialize_fit_wizard_recommendation,
    )
    from tests.core.test_fit_wizard_tiered import _plain_fingerprint

    row = _single_statistic(1.0, 3, 4)
    saved = FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=(row.template,),
        assessments=(row,),
        metric=SelectionMetric.AICC,
        recommended_key=row.template.key,
        comparable_keys=(),
        summary="cached recommendation",
    )
    restored = deserialize_fit_wizard_recommendation(serialize_fit_wizard_recommendation(saved))
    assert restored.recommended_key is None
    assert restored.assessments[0].unavailable_reason(SelectionMetric.AICC) == "n − k − 1 = 0"


def test_shared_columns_count_once_plus_exempt_run_columns():
    from asymmetry.core.fitting.global_fit_wizard import _assignment_parameter_count
    from asymmetry.core.fitting.global_search.trend_objective import CandidateRung
    from asymmetry.core.fitting.trend_quality import CandidateTrend

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    rung = CandidateRung(
        exempt_runs=(2,),
        series_cost=0.0,
        run_costs={},
        trend=CandidateTrend(parameters={}),
        hard_to_justify=(),
        pass_disagreements={},
        preselected=True,
        amplitude_unshareable_runs=(),
        all_local_chi2r=1.0,
    )
    assert _assignment_parameter_count(model, ("A_1",), ("Lambda",), run_count=3, rung=rung) == 5
    assert _assignment_parameter_count(model, ("A_1",), ("Lambda",), run_count=3, rung=None) == 4


@pytest.mark.parametrize("restore", [False, True])
def test_global_wizard_cannot_recommend_an_undefined_score(restore):
    from asymmetry.core.fitting.global_fit_wizard import (
        GlobalFitWizardRecommendation,
        deserialize_global_fit_wizard_recommendation,
        rerank_global_fit_wizard_recommendation,
        serialize_global_fit_wizard_recommendation,
    )
    from asymmetry.core.fitting.global_search.trend_objective import SelectionObjective
    from tests.core.test_model_comparison import _assessment

    row = _assessment("undefined", statistic=1.0)
    row = replace(row, sample_count=row.parameter_count + 1)
    recommendation = GlobalFitWizardRecommendation(
        series_axis_key="field",
        series_axis_label="Field",
        mixed_axes_warning=None,
        fingerprints_by_run={},
        dataset_order=tuple(row.fit_results_by_run),
        templates=(row.template,),
        assessments=(row,),
        metric=SelectionMetric.AICC,
        recommended_key=row.selection_key,
        comparable_keys=(),
        summary="cached recommendation",
        objective=SelectionObjective.STATISTICAL,
    )
    ranked = (
        deserialize_global_fit_wizard_recommendation(
            serialize_global_fit_wizard_recommendation(recommendation)
        )
        if restore
        else rerank_global_fit_wizard_recommendation(recommendation, SelectionMetric.AICC)
    )
    assert ranked.recommended_key is None
    assert ranked.summary == "No fitted candidate can be ranked by AICc."


def test_defined_null_can_anchor_an_unrankable_candidate_without_claiming_structure():
    from asymmetry.core.fitting.fit_wizard import (
        FitWizardRecommendation,
        RecommendationVerdict,
        rerank_fit_wizard_recommendation,
    )
    from tests.core.test_fit_wizard_tiered import _plain_fingerprint

    row = _single_statistic(1.0, 3, 4)
    null = replace(
        _single_statistic(10.0, 1, 4),
        template=replace(row.template, key="null"),
        is_null_baseline=True,
    )
    recommendation = FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=(row.template, null.template),
        assessments=(row, null),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
    )
    ranked = rerank_fit_wizard_recommendation(recommendation, SelectionMetric.AICC)
    assert ranked.recommended_key == "null"
    assert ranked.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE
    assert "criterion availability checks" in ranked.caveat
    assert "disqualified" not in ranked.caveat

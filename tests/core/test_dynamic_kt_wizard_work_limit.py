"""A bounded numerical evaluation failure belongs to one wizard attempt."""

from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.fit_wizard import build_fit_wizard_recommendation_for_templates
from asymmetry.core.fitting.strong_collision import EvaluationWorkLimitError
from tests.core.test_wizard_applied_field import _lkt_template, _record


def test_wizard_continues_after_one_work_limited_attempt(monkeypatch):
    original = FitEngine.fit
    calls = []

    def fit(self, dataset, model, parameters, **kwargs):
        calls.append(parameters)
        if len(calls) == 1:
            raise EvaluationWorkLimitError("Required samples exceed the work budget.")
        return original(self, dataset, model, parameters, **kwargs)

    monkeypatch.setattr(FitEngine, "fit", fit)
    recommendation = build_fit_wizard_recommendation_for_templates(
        _record(metadata={"field": 0, "field_direction": "ZF"}), [_lkt_template()]
    )
    (assessment,) = recommendation.assessments
    assert len(calls) > 1
    assert assessment.fit_result.success
    assert "work budget" not in assessment.fit_result.message


def test_wizard_preserves_all_work_limited_attempt_diagnostic(monkeypatch):
    calls = []

    def fit(self, dataset, model, parameters, **kwargs):
        calls.append(parameters)
        raise EvaluationWorkLimitError("Required samples exceed the work budget.")

    monkeypatch.setattr(FitEngine, "fit", fit)
    recommendation = build_fit_wizard_recommendation_for_templates(
        _record(metadata={"field": 0, "field_direction": "ZF"}), [_lkt_template()]
    )
    (assessment,) = recommendation.assessments
    assert len(calls) > 1
    result = assessment.fit_result
    assert not result.success
    assert "work budget" in result.message
    assert result.chi_squared == float("inf")
    assert [(p.name, p.value, p.fixed) for p in result.parameters] == [
        (p.name, p.value, p.fixed) for p in calls[0]
    ]


def test_actual_oversized_seeds_have_no_eager_or_deferred_curve():
    from asymmetry.core.fitting.fit_wizard import assessment_with_curves

    dataset = _record(metadata={"field": 1e6, "field_direction": "LF"})
    recommendation = build_fit_wizard_recommendation_for_templates(dataset, [_lkt_template()])
    (assessment,) = recommendation.assessments
    assert not assessment.fit_result.success
    assert "work budget" in assessment.fit_result.message
    assert assessment.fit_result.parameters["B_L"].value == 1e6
    assert assessment.fit_result.parameters["B_L"].fixed
    assert assessment.fitted_time.size == assessment.fitted_curve.size == 0
    rebuilt = assessment_with_curves(recommendation, assessment.template.key, dataset)
    assert rebuilt.fit_result is assessment.fit_result
    assert rebuilt.fitted_time.size == rebuilt.fitted_curve.size == 0
    assert rebuilt.component_curves == ()

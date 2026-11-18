"""A bounded numerical evaluation failure belongs to one wizard attempt."""

import pytest

from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.fit_wizard import build_fit_wizard_recommendation_for_templates
from asymmetry.core.fitting.strong_collision import EvaluationWorkLimitError
from tests.core.test_wizard_applied_field import _lkt_template, _record, _template


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


@pytest.mark.parametrize(
    "template",
    [_lkt_template(), _template(["GaussianBroadenedKT", "Constant"], ["+"], "gbkt_constant")],
)
def test_actual_oversized_seeds_have_no_eager_or_deferred_curve(template):
    from asymmetry.core.fitting.fit_wizard import assessment_with_curves

    dataset = _record(metadata={"field": 1e7, "field_direction": "LF"})
    recommendation = build_fit_wizard_recommendation_for_templates(dataset, [template])
    (assessment,) = recommendation.assessments
    assert not assessment.fit_result.success
    assert "work budget" in assessment.fit_result.message
    assert assessment.fit_result.parameters["B_L"].value == 1e7
    assert assessment.fit_result.parameters["B_L"].fixed
    assert assessment.residual_gate_reasons == (assessment.fit_result.message,)
    assert assessment.fitted_time.size == assessment.fitted_curve.size == 0
    rebuilt = assessment_with_curves(recommendation, assessment.template.key, dataset)
    assert rebuilt.fit_result is assessment.fit_result
    assert rebuilt.fitted_time.size == rebuilt.fitted_curve.size == 0
    assert rebuilt.component_curves == ()


def test_failed_optimizer_keeps_diagnostics_from_computed_residuals():
    import numpy as np

    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.fit_wizard import _residual_gate_reasons
    from asymmetry.core.fitting.likelihood import Objective
    from asymmetry.core.fitting.spectral_residual import _residual_diagnostics

    dataset = _record()
    result = FitResult(
        success=False,
        objective=Objective.GAUSSIAN,
        residuals=20 * np.sin(2 * np.pi * 0.8 * dataset.time),
        message="Optimizer did not converge.",
    )
    rms, runs, autocorrelation, snr = _residual_diagnostics(dataset, result)
    reasons = _residual_gate_reasons(
        fit_result=result,
        residual_rms=rms,
        runs_z_score=runs,
        max_abs_autocorrelation=autocorrelation,
        residual_fft_peak_snr=snr,
        bound_hits=[],
    )
    assert result.message in reasons
    assert any("RMS" in reason for reason in reasons)
    assert any("FFT" in reason for reason in reasons)
    assert all("inf" not in reason for reason in reasons)

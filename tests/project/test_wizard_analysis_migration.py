"""Both standalone recommendation codecs preserve pending and constrained history."""

from copy import deepcopy

import pytest

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    deserialize_fit_wizard_recommendation,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_fit_wizard import (
    _deserialize_global_candidate_assessment,
    deserialize_global_fit_wizard_recommendation,
    serialize_global_fit_wizard_recommendation,
)
from tests.core.test_fit_wizard import _minimal_fingerprint_payload


def wizard_payload(global_fit, *, missing=False, constrained=False):
    template = {
        "key": "old",
        "title": "Saved model",
        "model": {
            "component_names": ["Exponential", "AbsentScale" if missing else "Gaussian"],
            "operators": ["*"],
            "open_parentheses": [1, 0],
            "close_parentheses": [0, 1],
            "fraction_groups": [],
        },
    }
    rows = [
        {"name": "A_1", "value": 2.0},
        {"name": "A_2", "value": 3.0},
        {"name": "Lambda", "value": 0.5},
        {"name": "sigma", "value": 0.3},
    ]
    if constrained:
        rows[0]["tie"] = {"main": "x", "scale": 2.0}
        rows.append({"name": "x", "value": 1.0, "fixed": True})
    result = {
        "success": True,
        "parameters": rows,
        "uncertainties": {},
        "computation_revision": None,
    }
    assessment = {"template": template, "rung": None}
    if global_fit:
        assessment.update(
            fit_results_by_run={"7": result},
            global_parameters=rows,
            global_param_names=["A_1", "A_2"],
            local_param_names=["Lambda", "sigma"],
        )
    else:
        assessment["fit_result"] = result
    return {
        "templates": [template],
        "assessments": [assessment],
        "objective": "statistical",
        "metric": "aicc",
        "fingerprint": _minimal_fingerprint_payload(),
        "computation_revision": "older",
    }


@pytest.mark.parametrize("global_fit", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_recommendation_codec_round_trip_preserves_constraint_archive(global_fit, constrained):
    reader = (
        deserialize_global_fit_wizard_recommendation
        if global_fit
        else deserialize_fit_wizard_recommendation
    )
    writer = (
        serialize_global_fit_wizard_recommendation
        if global_fit
        else serialize_fit_wizard_recommendation
    )
    payload = wizard_payload(global_fit, constrained=constrained)
    original = deepcopy(payload)
    recommendation = reader(payload)
    assert recommendation is not None
    assert payload == original
    row = recommendation.assessments[0]
    result = row.fit_results_by_run[7] if global_fit else row.fit_result
    assert result.parameters["A_1"].value == 6.0
    assert result.computation_revision is None
    if constrained:
        assert row.analysis_history["original"] == original["assessments"][0]
        assert not row.is_successful
    stored = writer(recommendation)
    assert stored["analysis_payload_version"] == 1
    assert writer(reader(stored)) == stored
    assert reader(stored).computation_revision == "older"


@pytest.mark.parametrize("global_fit", [False, True])
def test_recommendation_missing_model_resolves_after_real_codec_round_trip(global_fit, monkeypatch):
    reader = (
        deserialize_global_fit_wizard_recommendation
        if global_fit
        else deserialize_fit_wizard_recommendation
    )
    writer = (
        serialize_global_fit_wizard_recommendation
        if global_fit
        else serialize_fit_wizard_recommendation
    )
    first = reader(wizard_payload(global_fit, missing=True))
    assert first.assessments[0].analysis_history["reason"] == "missing_model"
    stored = writer(first)
    monkeypatch.setitem(COMPONENTS, "AbsentScale", COMPONENTS["Gaussian"])
    second = reader(stored)
    row = second.assessments[0]
    assert row.analysis_history is None
    result = row.fit_results_by_run[7] if global_fit else row.fit_result
    assert result.parameters["A_1"].value == 6.0
    assert result.computation_revision is None


def test_deserialize_global_candidate_assessment_migrates_legacy_fraction_params() -> None:
    """Global values, errors and role names share the stored template context."""
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    template_payload = {
        "key": "cand",
        "title": "",
        "category": "",
        "rationale": "",
        "model": model.to_dict(),
    }
    fit_result_payload = {
        "success": True,
        "chi_squared": 1.0,
        "reduced_chi_squared": 1.0,
        "parameters": [
            {"name": "A_1", "value": 20.0, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "Lambda", "value": 0.5, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "fraction_1", "value": 2.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "sigma", "value": 0.3, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "fraction_2", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "fraction_3", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
        ],
        "uncertainties": {"fraction_1": 0.01, "fraction_2": 0.02, "fraction_3": 0.03},
    }
    payload = {
        "template": template_payload,
        "fit_results_by_run": {"1001": fit_result_payload},
        "global_parameters": [
            {"name": "A_1", "value": 20.0, "min": 0.0, "max": 1e9, "fixed": False},
            {"name": "fraction_1", "value": 2.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "fraction_2", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
            {"name": "fraction_3", "value": 1.0, "min": 0.0, "max": 1.0, "fixed": False},
        ],
        "global_param_names": ["A_1", "fraction_1", "fraction_2", "fraction_3"],
        "local_param_names": ["Lambda"],
        "fixed_param_names": [],
        "aic": 1.0,
        "bic": 1.0,
        "selected_score": 1.0,
        "rung": None,
    }

    from asymmetry.core.project.analysis_migration import migrate_wizard_payload

    canonical = migrate_wizard_payload({"assessments": [payload]})
    assessment = _deserialize_global_candidate_assessment(canonical["assessments"][0])

    assert assessment is not None

    # Per-run fit result parameters migrated.
    run_result = assessment.fit_results_by_run[1001]
    run_names = {parameter.name for parameter in run_result.parameters}
    assert "f_Exponential" in run_names
    assert "f_Gaussian" in run_names
    assert not any(name.startswith("fraction_") for name in run_names)
    assert "f_Exponential" in run_result.uncertainties
    assert "f_Gaussian" in run_result.uncertainties
    assert not any(key.startswith("fraction_") for key in run_result.uncertainties)

    # Standalone global_parameters ParameterSet migrated.
    global_names = set(assessment.global_parameters.names)
    assert "f_Exponential" in global_names
    assert "f_Gaussian" in global_names
    assert not any(name.startswith("fraction_") for name in global_names)

    # Parameter-role tuples migrated (legacy names renamed/dropped in place).
    assert "f_Exponential" in assessment.global_param_names
    assert "f_Gaussian" in assessment.global_param_names
    assert not any(name.startswith("fraction_") for name in assessment.global_param_names)
    assert not any(name.startswith("fraction_") for name in assessment.local_param_names)
    assert not any(name.startswith("fraction_") for name in assessment.fixed_param_names)
